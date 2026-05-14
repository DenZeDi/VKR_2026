# prepare_vkr_data.py
# Единый скрипт проверки и предобработки данных для MVP ВКР.
# Ожидаемые входные файлы:
#   data/processed/purchases.parquet
#   data/processed/clients.parquet
#   data/processed/policies.parquet
#
# Выходные файлы:
#   data/model_input/purchases_prepared.parquet
#   data/model_input/clients_prepared.parquet
#   data/model_input/policies_prepared.parquet
#   data/model_input/preprocessing_report.json
#   data/model_input/preprocessing_report.xlsx

from pathlib import Path
import json
import argparse
import warnings

import numpy as np
import pandas as pd


DEFAULT_SNAPSHOT_DATE = pd.Timestamp("2026-01-31")

EXCLUDE_KPZN_DEFAULT = False


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def rename_if_exists(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df = df.copy()
    real_mapping = {}
    for old, new in mapping.items():
        if old in df.columns and new not in df.columns:
            real_mapping[old] = new
    return df.rename(columns=real_mapping)


def to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


def mode_or_first(s: pd.Series):
    s2 = s.dropna()
    if len(s2) == 0:
        return np.nan
    mode = s2.mode(dropna=True)
    if len(mode) > 0:
        return mode.iloc[0]
    return s2.iloc[0]


def numeric_summary(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    if cols is None:
        cols = df.select_dtypes(include=["number"]).columns.tolist()

    rows = []
    for col in cols:
        if col not in df.columns:
            continue

        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().sum() == 0:
            rows.append({
                "column": col,
                "count": 0,
                "missing": int(x.isna().sum()),
            })
            continue

        rows.append({
            "column": col,
            "count": int(x.notna().sum()),
            "missing": int(x.isna().sum()),
            "min": float(x.min()),
            "p01": float(x.quantile(0.01)),
            "p05": float(x.quantile(0.05)),
            "p25": float(x.quantile(0.25)),
            "p50": float(x.quantile(0.50)),
            "p75": float(x.quantile(0.75)),
            "p95": float(x.quantile(0.95)),
            "p99": float(x.quantile(0.99)),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "std": float(x.std()) if x.notna().sum() > 1 else None,
        })
    return pd.DataFrame(rows)


def value_counts_df(df: pd.DataFrame, col: str, top_n: int = 50) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()

    res = (
        df[col]
        .astype("string")
        .fillna("NULL")
        .value_counts(dropna=False)
        .head(top_n)
        .reset_index()
    )
    res.columns = [col, "count"]
    res["share"] = res["count"] / len(df) if len(df) else 0
    return res


def basic_profile(df: pd.DataFrame, name: str, key_col: str = None) -> dict:
    profile = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "columns_list": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_count": df.isna().sum().astype(int).to_dict(),
        "missing_share": df.isna().mean().round(6).to_dict(),
        "nunique": df.nunique(dropna=True).astype(int).to_dict(),
    }
    if key_col and key_col in df.columns:
        profile[f"unique_{key_col}"] = int(df[key_col].nunique(dropna=True))
        profile[f"duplicated_{key_col}"] = int(df.duplicated(key_col).sum())
    return profile


def save_report(report: dict, tables: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with pd.ExcelWriter(out_dir / "preprocessing_report.xlsx", engine="openpyxl") as writer:
        # summary sheet
        summary_rows = []
        for k, v in report.items():
            if isinstance(v, dict) and "before_rows" in v:
                summary_rows.append({
                    "dataset": k,
                    "before_rows": v.get("before_rows"),
                    "after_rows": v.get("after_rows"),
                    "rows_removed": v.get("rows_removed"),
                    "key_duplicates_before": v.get("key_duplicates_before"),
                    "key_duplicates_after": v.get("key_duplicates_after"),
                })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)

        for sheet_name, table in tables.items():
            if table is None or len(table) == 0:
                continue
            safe_name = sheet_name[:31]
            table.to_excel(writer, sheet_name=safe_name, index=False)


def clean_product_universe(df: pd.DataFrame, exclude_kpzn: bool) -> pd.DataFrame:
    if not exclude_kpzn or "product_code" not in df.columns:
        return df
    return df[df["product_code"] != "Коробочное страхование в КПЗН"].copy()


# ---------------------------------------------------------------------
# Purchases
# ---------------------------------------------------------------------

def prepare_purchases(data_dir: Path, out_dir: Path, report: dict, tables: dict, exclude_kpzn: bool):
    path = data_dir / "purchases.parquet"
    if not path.exists():
        warnings.warn(f"Файл не найден: {path}")
        report["purchases"] = {"error": f"file not found: {str(path)}"}
        return None

    raw = pd.read_parquet(path)
    df = normalize_columns(raw)

    df = rename_if_exists(df, {
        "client_pin": "client_id",
        "value_day": "purchase_date",
        "typeproduct_ccode": "product_code",
        "trn_cur_amt": "purchase_amount",
    })

    before_rows = len(df)
    key_dups_before = int(df.duplicated("policy_id").sum()) if "policy_id" in df.columns else None

    required = ["client_id", "purchase_date", "purchase_amount", "product_code", "is_renewal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"purchases.parquet: нет обязательных колонок: {missing}")

    df["purchase_date"] = to_datetime_safe(df["purchase_date"])
    df["purchase_amount"] = to_numeric_safe(df["purchase_amount"])
    df["is_renewal"] = pd.to_numeric(df["is_renewal"], errors="coerce").fillna(0).astype(int)

    # В текущем профиле policy_id не является уникальным ключом покупки, поэтому создаем технический ID.
    if "policy_id" not in df.columns:
        df["policy_id"] = np.nan

    df = df.dropna(subset=["client_id", "purchase_date", "purchase_amount", "product_code"])
    df = df[df["purchase_amount"] > 0].copy()
    df = clean_product_universe(df, exclude_kpzn=exclude_kpzn)

    df = df.sort_values(["client_id", "purchase_date", "product_code", "purchase_amount"]).reset_index(drop=True)
    df["purchase_event_id"] = np.arange(len(df), dtype=np.int64)

    # Для LTV / ARPPU оставляем исходную сумму и модельную сумму с отсечением выбросов по продукту.
    q99 = df.groupby("product_code")["purchase_amount"].transform(lambda x: x.quantile(0.99))
    df["purchase_amount_model"] = df["purchase_amount"].clip(upper=q99)

    # Полезные календарные признаки для агрегаций.
    df["purchase_month"] = df["purchase_date"].dt.to_period("M").astype(str)

    out_path = out_dir / "purchases_prepared.parquet"
    df.to_parquet(out_path, index=False)

    report["purchases"] = {
        "before_rows": int(before_rows),
        "after_rows": int(len(df)),
        "rows_removed": int(before_rows - len(df)),
        "key": "purchase_event_id",
        "policy_id_note": "policy_id in source is not unique; purchase_event_id was created",
        "key_duplicates_before": key_dups_before,
        "key_duplicates_after": int(df.duplicated("purchase_event_id").sum()),
        "output": str(out_path),
    }

    tables["purchases_numeric"] = numeric_summary(df, ["purchase_amount", "purchase_amount_model"])
    tables["purchases_is_renewal"] = value_counts_df(df, "is_renewal")
    tables["purchases_product"] = value_counts_df(df, "product_code")
    tables["purchases_month"] = value_counts_df(df, "purchase_month", top_n=200)

    return df


# ---------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------

def aggregate_duplicate_clients(df: pd.DataFrame) -> pd.DataFrame:
    # Стандартизированные колонки
    id_col = "client_id"

    binary_cols = [
        "active_cc_flag",
        "active_loan_flag",
        "active_dc_flag",
        "prod_client_flag",
        "ever_bought_insurance",
        "bought_insurance_in_next_90d",
        "has_txn_90d",
    ]

    txn_max_cols = [
        "txn_count_30d",
        "txn_count_90d",
        "txn_amount_30d",
        "txn_amount_90d",
        "avg_txn_amount_90d",
        "active_txn_days_90d",
    ]

    median_cols = [
        "age_years",
        "tenure_bank_months",
    ]

    min_cols = [
        "days_since_last_txn",
    ]

    agg = {}

    for col in df.columns:
        if col == id_col:
            continue
        if col in binary_cols:
            agg[col] = "max"
        elif col in txn_max_cols:
            agg[col] = "max"
        elif col in median_cols:
            agg[col] = "median"
        elif col in min_cols:
            agg[col] = "min"
        elif col == "products_count_min":
            agg[col] = "max"
        elif col == "sample_weight":
            # Потом пересчитаем ниже, здесь просто берем первое значение.
            agg[col] = "first"
        elif pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = "first"
        else:
            agg[col] = mode_or_first

    return df.groupby(id_col, as_index=False).agg(agg)


def prepare_clients(data_dir: Path, out_dir: Path, report: dict, tables: dict):
    path = data_dir / "clients.parquet"
    if not path.exists():
        warnings.warn(f"Файл не найден: {path}")
        report["clients"] = {"error": f"file not found: {str(path)}"}
        return None

    raw = pd.read_parquet(path)
    df = normalize_columns(raw)

    df = rename_if_exists(df, {
        "client_pin": "client_id",
        "client_pin.1": "client_id_joined",
        "gender_ccode": "gender_code",
        "country_ccode": "country_code",
        "addref_city_name": "city_name",
        "addrref_city_name": "city_name",
        "clientsegment_grace_ccode": "client_segment",
        "tnx_count_30d": "txn_count_30d",
        "tnx_count_90d": "txn_count_90d",
        "tnx_amount_30d": "txn_amount_30d",
        "tnx_amount_90d": "txn_amount_90d",
        "avg_tnx_amound_90d": "avg_txn_amount_90d",
        "avg_tnx_amount_90d": "avg_txn_amount_90d",
        "active_trx_days_90d": "active_txn_days_90d",
    })

    before_rows = len(df)
    key_dups_before = int(df.duplicated("client_id").sum()) if "client_id" in df.columns else None

    required = ["client_id", "bought_insurance_in_next_90d"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"clients.parquet: нет обязательных колонок: {missing}")

    # Удаляем технический дубль ключа после join.
    if "client_id_joined" in df.columns:
        df = df.drop(columns=["client_id_joined"])

    # Даты
    if "snapshot_date" not in df.columns:
        df["snapshot_date"] = DEFAULT_SNAPSHOT_DATE
    else:
        df["snapshot_date"] = to_datetime_safe(df["snapshot_date"]).fillna(DEFAULT_SNAPSHOT_DATE)

    if "birth_date" in df.columns:
        df["birth_date"] = to_datetime_safe(df["birth_date"])

    if "client_start_date" in df.columns:
        df["client_start_date"] = to_datetime_safe(df["client_start_date"])

    # Числовые и бинарные признаки
    int_cols = [
        "active_cc_flag",
        "active_loan_flag",
        "active_dc_flag",
        "products_count_min",
        "prod_client_flag",
        "ever_bought_insurance",
        "bought_insurance_in_next_90d",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Пересчитываем age / tenure, если даты есть.
    if "birth_date" in df.columns:
        df["age_years"] = (df["snapshot_date"] - df["birth_date"]).dt.days / 365.25
        df.loc[(df["age_years"] < 14) | (df["age_years"] > 100), "age_years"] = np.nan
    elif "age_years" in df.columns:
        df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
        df.loc[(df["age_years"] < 14) | (df["age_years"] > 100), "age_years"] = np.nan

    if "client_start_date" in df.columns:
        df["tenure_bank_months"] = (df["snapshot_date"] - df["client_start_date"]).dt.days / 30.44
        df.loc[df["tenure_bank_months"] < 0, "tenure_bank_months"] = np.nan
    elif "tenure_bank_months" in df.columns:
        df["tenure_bank_months"] = pd.to_numeric(df["tenure_bank_months"], errors="coerce")
        df.loc[df["tenure_bank_months"] < 0, "tenure_bank_months"] = np.nan

    # Транзакционные признаки
    txn_zero_cols = [
        "txn_count_30d",
        "txn_count_90d",
        "txn_amount_30d",
        "txn_amount_90d",
        "avg_txn_amount_90d",
        "active_txn_days_90d",
    ]

    for col in txn_zero_cols:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col]) if df[col].dtype == "object" else pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].clip(lower=0)

    if "days_since_last_txn" in df.columns:
        df["days_since_last_txn"] = pd.to_numeric(df["days_since_last_txn"], errors="coerce")
        df["days_since_last_txn"] = df["days_since_last_txn"].clip(lower=0)

    # Флаг наличия транзакций за 90 дней.
    if "has_txn_90d" not in df.columns:
        if "txn_count_90d" in df.columns:
            df["has_txn_90d"] = np.where(df["txn_count_90d"].fillna(0) > 0, 1, 0)
        else:
            df["has_txn_90d"] = 0

    # Агрегируем до 1 строки на клиента. Это обязательно: по профилю были дубли client_pin.
    if df.duplicated("client_id").sum() > 0:
        df = aggregate_duplicate_clients(df)

    # Заполняем отсутствующие транзакции уже после агрегации.
    for col in txn_zero_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    if "days_since_last_txn" in df.columns:
        df["days_since_last_txn"] = pd.to_numeric(df["days_since_last_txn"], errors="coerce").fillna(9999).clip(lower=0)

    # Если у клиента нет транзакций, фиксируем это явно.
    if "txn_count_90d" in df.columns:
        df["has_txn_90d"] = np.where(df["txn_count_90d"] > 0, 1, 0)

    # Если products_count_min не пришел, считаем по доступным флагам.
    if "products_count_min" not in df.columns:
        product_flags = [c for c in ["active_cc_flag", "active_loan_flag", "active_dc_flag"] if c in df.columns]
        if product_flags:
            df["products_count_min"] = df[product_flags].sum(axis=1)

    # Веса. Сохраняем пришедшие, но если после агрегации они сломались — восстанавливаем под текущий сэмпл:
    # positive 20% => weight 5; negative 0.5% => weight 200.
    if "sample_weight" not in df.columns:
        df["sample_weight"] = np.where(df["bought_insurance_in_next_90d"] == 1, 5.0, 200.0)
    else:
        df["sample_weight"] = pd.to_numeric(df["sample_weight"], errors="coerce")
        bad_weight = df["sample_weight"].isna() | (df["sample_weight"] <= 0)
        df.loc[bad_weight, "sample_weight"] = np.where(
            df.loc[bad_weight, "bought_insurance_in_next_90d"] == 1,
            5.0,
            200.0,
        )

    # Категориальные признаки
    cat_cols = [
        "gender_code",
        "country_code",
        "city_name",
        "client_segment",
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("unknown")

    # Отсечение выбросов + log-признаки по транзакциям.
    log_cols = [
        "txn_count_30d",
        "txn_count_90d",
        "txn_amount_30d",
        "txn_amount_90d",
        "avg_txn_amount_90d",
    ]
    for col in log_cols:
        if col in df.columns:
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=q99)
            df[col + "_log"] = np.log1p(df[col])

    key_dups_after = int(df.duplicated("client_id").sum())

    out_path = out_dir / "clients_prepared.parquet"
    df.to_parquet(out_path, index=False)

    report["clients"] = {
        "before_rows": int(before_rows),
        "after_rows": int(len(df)),
        "rows_removed": int(before_rows - len(df)),
        "key": "client_id",
        "key_duplicates_before": key_dups_before,
        "key_duplicates_after": key_dups_after,
        "note": "duplicates were aggregated to one row per client_id; transaction nulls mean no transactions in 90d window",
        "output": str(out_path),
    }

    tables["clients_numeric"] = numeric_summary(df)
    tables["clients_target"] = value_counts_df(df, "bought_insurance_in_next_90d")
    tables["clients_sample_weight"] = value_counts_df(df, "sample_weight")
    tables["clients_has_txn"] = value_counts_df(df, "has_txn_90d")
    tables["clients_segment"] = value_counts_df(df, "client_segment")
    tables["clients_city"] = value_counts_df(df, "city_name")
    tables["clients_txn_summary"] = numeric_summary(
        df,
        [
            "txn_count_30d",
            "txn_count_90d",
            "txn_amount_30d",
            "txn_amount_90d",
            "avg_txn_amount_90d",
            "active_txn_days_90d",
            "days_since_last_txn",
            "txn_count_90d_log",
            "txn_amount_90d_log",
        ],
    )

    return df


# ---------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------

def prepare_policies(data_dir: Path, out_dir: Path, report: dict, tables: dict, exclude_kpzn: bool, max_duration_days: int):
    path = data_dir / "policies.parquet"
    if not path.exists():
        warnings.warn(f"Файл не найден: {path}")
        report["policies"] = {"error": f"file not found: {str(path)}"}
        return None

    raw = pd.read_parquet(path)
    df = normalize_columns(raw)

    before_rows = len(df)
    key_dups_before = int(df.duplicated("policy_id").sum()) if "policy_id" in df.columns else None

    required = [
        "client_id",
        "policy_id",
        "product_code",
        "policy_start_date",
        "policy_end_date",
        "duration_days",
        "purchase_amount",
        "period_type",
        "is_renewed",
        "event_observed",
        "sample_weight",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"policies.parquet: нет обязательных колонок: {missing}")

    df["policy_start_date"] = to_datetime_safe(df["policy_start_date"])
    df["policy_end_date"] = to_datetime_safe(df["policy_end_date"])
    df["duration_days"] = pd.to_numeric(df["duration_days"], errors="coerce")
    df["purchase_amount"] = to_numeric_safe(df["purchase_amount"])
    df["is_renewed"] = pd.to_numeric(df["is_renewed"], errors="coerce").fillna(0).astype(int)
    df["event_observed"] = pd.to_numeric(df["event_observed"], errors="coerce").fillna(1).astype(int)
    df["sample_weight"] = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0)

    df["period_type"] = (
        df["period_type"]
        .astype("string")
        .str.lower()
        .str.replace("annualy", "yearly", regex=False)
        .str.replace("annually", "yearly", regex=False)
        .fillna("unknown")
    )

    df = clean_product_universe(df, exclude_kpzn=exclude_kpzn)

    # Базовая очистка
    df = df.dropna(subset=[
        "client_id",
        "policy_id",
        "product_code",
        "policy_start_date",
        "policy_end_date",
        "duration_days",
        "purchase_amount",
    ])
    df = df[df["purchase_amount"] > 0].copy()
    df = df[df["duration_days"] > 0].copy()
    df = df[df["policy_end_date"] >= df["policy_start_date"]].copy()

    # Мягкий фильтр на технически невозможную длительность.
    # По умолчанию 4000 дней ~ 11 лет. Если надо отключить, передай --max-duration-days 0.
    if max_duration_days and max_duration_days > 0:
        df = df[df["duration_days"] <= max_duration_days].copy()

    # Убираем возможные leakage-поля, если они вдруг есть.
    leakage_cols = [
        "payment_cnt",
        "payment_periods_cnt",
        "first_payment_date",
        "last_payment_date",
        "du_num_start",
        "du_num_end",
        "prolong_flag",
    ]
    existing_leakage_cols = [c for c in leakage_cols if c in df.columns]
    if existing_leakage_cols:
        df = df.drop(columns=existing_leakage_cols)

    # Если есть дубли policy_id, оставляем первый после сортировки.
    if df.duplicated("policy_id").sum() > 0:
        df = df.sort_values(["policy_id", "policy_start_date"]).drop_duplicates("policy_id", keep="first")

    key_dups_after = int(df.duplicated("policy_id").sum())

    out_path = out_dir / "policies_prepared.parquet"
    df.to_parquet(out_path, index=False)

    report["policies"] = {
        "before_rows": int(before_rows),
        "after_rows": int(len(df)),
        "rows_removed": int(before_rows - len(df)),
        "key": "policy_id",
        "key_duplicates_before": key_dups_before,
        "key_duplicates_after": key_dups_after,
        "output": str(out_path),
    }

    tables["policies_numeric"] = numeric_summary(df, ["duration_days", "purchase_amount"])
    tables["policies_is_renewed"] = value_counts_df(df, "is_renewed")
    tables["policies_event_observed"] = value_counts_df(df, "event_observed")
    tables["policies_sample_weight"] = value_counts_df(df, "sample_weight")
    tables["policies_product"] = value_counts_df(df, "product_code")
    tables["policies_period_type"] = value_counts_df(df, "period_type")

    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--out-dir", type=str, default="data/model_input")
    parser.add_argument("--exclude-kpzn", action="store_true", default=EXCLUDE_KPZN_DEFAULT)
    parser.add_argument("--max-duration-days", type=int, default=4000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    tables = {}

    print("=== Preparing purchases ===")
    try:
        purchases = prepare_purchases(data_dir, out_dir, report, tables, exclude_kpzn=args.exclude_kpzn)
        if purchases is not None:
            print("purchases_prepared:", purchases.shape)
    except Exception as e:
        report["purchases"] = {"error": str(e)}
        print("Purchases error:", e)

    print("\n=== Preparing clients ===")
    try:
        clients = prepare_clients(data_dir, out_dir, report, tables)
        if clients is not None:
            print("clients_prepared:", clients.shape)
            print(clients["bought_insurance_in_next_90d"].value_counts(dropna=False))
    except Exception as e:
        report["clients"] = {"error": str(e)}
        print("Clients error:", e)

    print("\n=== Preparing policies ===")
    try:
        policies = prepare_policies(
            data_dir,
            out_dir,
            report,
            tables,
            exclude_kpzn=args.exclude_kpzn,
            max_duration_days=args.max_duration_days,
        )
        if policies is not None:
            print("policies_prepared:", policies.shape)
            print(policies["is_renewed"].value_counts(dropna=False))
    except Exception as e:
        report["policies"] = {"error": str(e)}
        print("Policies error:", e)

    save_report(report, tables, out_dir)

    print("\n=== Done ===")
    print("Outputs:")
    print(f"  {out_dir / 'purchases_prepared.parquet'}")
    print(f"  {out_dir / 'clients_prepared.parquet'}")
    print(f"  {out_dir / 'policies_prepared.parquet'}")
    print(f"  {out_dir / 'preprocessing_report.json'}")
    print(f"  {out_dir / 'preprocessing_report.xlsx'}")


if __name__ == "__main__":
    main()
