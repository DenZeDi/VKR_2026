"""
Модель продления полиса.

Основной вход: data/model_input/policies_prepared.parquet
Единица наблюдения: 1 строка = 1 полис.
Target для интерпретации: is_renewed.
Survival-постановка: event_observed = 1 означает непродление.

Для устойчивости MVP используются два слоя:
  1) Cox PH для survival-интерпретации риска непродления;
  2) агрегированный renewal_rate как фактическая доля продления в обучающей выборке.
"""
from __future__ import annotations

import json

import cloudpickle
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

from config import MODELS_ARTIFACTS_DIR, POLICIES_PREPARED_PATH, RANDOM_SEED


DURATION_COL = "duration_days"
EVENT_COL = "event_observed"
TARGET_COL = "is_renewed"
WEIGHT_COL = "sample_weight"

NUMERIC_FEATURES_CANDIDATES = [
    "purchase_amount",
]
CATEGORICAL_FEATURES_CANDIDATES = [
    "product_code",
    "period_type",
]

COX_PENALIZER = 0.01
COX_L1_RATIO = 0.0


def _existing(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def load_policies(path=POLICIES_PREPARED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path}. Сначала сформируй policies_prepared.parquet")
    df = pd.read_parquet(path)
    df.columns = df.columns.str.lower()
    required = ["policy_id", DURATION_COL, EVENT_COL, TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"policies_prepared.parquet: нет колонок {missing}")
    if df.duplicated("policy_id").sum() > 0:
        raise ValueError("policies_prepared содержит дубли policy_id")
    return df


def prepare_cox_data(policies_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = policies_df.copy()

    numeric_features = _existing(NUMERIC_FEATURES_CANDIDATES, df)
    categorical_features = _existing(CATEGORICAL_FEATURES_CANDIDATES, df)

    # Базовая очистка
    df[DURATION_COL] = pd.to_numeric(df[DURATION_COL], errors="coerce")
    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(1).astype(int)
    if WEIGHT_COL in df.columns:
        df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(1.0)
    else:
        df[WEIGHT_COL] = 1.0

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_features:
        df[col] = df[col].astype("string").fillna("unknown")

    keep = [DURATION_COL, EVENT_COL, WEIGHT_COL] + numeric_features + categorical_features
    df = df[keep].dropna().copy()
    df = df[df[DURATION_COL] > 0].copy()

    # Лог-сумма лучше для устойчивости Cox, но сохраняем смысл premium.
    if "purchase_amount" in df.columns:
        df["purchase_amount_log"] = np.log1p(df["purchase_amount"].clip(lower=0))
        numeric_features = [c for c in numeric_features if c != "purchase_amount"] + ["purchase_amount_log"]

    encoded = pd.get_dummies(df[categorical_features], prefix=categorical_features, drop_first=True, dtype=int)
    cox_df = pd.concat([
        df[numeric_features],
        encoded,
        df[[DURATION_COL, EVENT_COL, WEIGHT_COL]],
    ], axis=1)

    feature_cols = [c for c in cox_df.columns if c not in {DURATION_COL, EVENT_COL, WEIGHT_COL}]
    return cox_df, feature_cols


def fit_cox(train_df: pd.DataFrame) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=COX_PENALIZER, l1_ratio=COX_L1_RATIO)
    # lifelines поддерживает weights_col. robust=True рекомендуется при весах.
    cph.fit(
        train_df,
        duration_col=DURATION_COL,
        event_col=EVENT_COL,
        weights_col=WEIGHT_COL,
        robust=True,
        show_progress=False,
    )
    return cph


def evaluate_cox(cph: CoxPHFitter, test_df: pd.DataFrame) -> dict:
    # Cox возвращает partial hazard: выше = выше риск непродления.
    risk = cph.predict_partial_hazard(test_df).values.ravel()
    c_index = concordance_index(
        event_times=test_df[DURATION_COL],
        predicted_scores=-risk,
        event_observed=test_df[EVENT_COL],
    )
    return {
        "c_index_test": float(c_index),
        "n_test": int(len(test_df)),
        "event_rate_test": float(test_df[EVENT_COL].mean()),
    }


def predict_renewal_probability(cph: CoxPHFitter, cox_df: pd.DataFrame, horizon_days: int | None = None) -> pd.Series:
    """
    Возвращает вероятность продления как survival probability по событию непродления.
    Если horizon_days не задан, используется медианная длительность полиса в портфеле.
    """
    if horizon_days is None:
        horizon_days = int(np.nanmedian(cox_df[DURATION_COL]))
        horizon_days = max(horizon_days, 1)

    surv = cph.predict_survival_function(cox_df, times=[horizon_days]).T.iloc[:, 0]
    return surv.clip(0, 1)


def train_and_save(policies_path=POLICIES_PREPARED_PATH) -> dict:
    print("=== Cox PH — продление / непродление ===")
    policies = load_policies(policies_path)
    print(f"Полисов: {len(policies):,}")
    print(f"Renewed: {int(policies[TARGET_COL].sum()):,} ({policies[TARGET_COL].mean()*100:.2f}%)")

    cox_df, feature_cols = prepare_cox_data(policies)
    print(f"Строк для Cox после подготовки: {len(cox_df):,}")
    print(f"Признаков Cox: {len(feature_cols)}")

    train_df, test_df = train_test_split(
        cox_df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=cox_df[EVENT_COL],
    )

    cph = fit_cox(train_df)
    metrics = evaluate_cox(cph, test_df)

    renewal_prob = predict_renewal_probability(cph, cox_df)
    predicted_renewal_rate = float(renewal_prob.mean())
    actual_renewal_rate = float(policies[TARGET_COL].mean())

    print(f"C-index test: {metrics['c_index_test']:.4f}")
    print(f"Фактический renewal_rate: {actual_renewal_rate:.4f}")
    print(f"Средний прогноз renewal_rate: {predicted_renewal_rate:.4f}")

    model_path = MODELS_ARTIFACTS_DIR / "cox_renewal.pkl"
    with open(model_path, "wb") as f:
        cloudpickle.dump(cph, f)

    with open(MODELS_ARTIFACTS_DIR / "cox_renewal_features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    summary = cph.summary.copy()
    summary["hazard_ratio"] = np.exp(summary["coef"])
    hazard_ratios = (
        summary.reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("hazard_ratio", ascending=False)
        .to_dict(orient="records")
    )

    report = {
        "model_type": "cox_ph",
        "target_interpretation": "event_observed=1 means non-renewal; survival probability is renewal probability",
        "n_policies": int(len(policies)),
        "n_cox_rows": int(len(cox_df)),
        "actual_renewal_rate": actual_renewal_rate,
        "aggregate_renewal_rate": predicted_renewal_rate,
        "metrics": metrics,
        "features": feature_cols,
        "hazard_ratios": hazard_ratios,
        "cox_params": {"penalizer": COX_PENALIZER, "l1_ratio": COX_L1_RATIO},
    }
    report_path = MODELS_ARTIFACTS_DIR / "cox_renewal_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"Артефакты сохранены: {model_path.name}, {report_path.name}")
    return report


if __name__ == "__main__":
    train_and_save()
