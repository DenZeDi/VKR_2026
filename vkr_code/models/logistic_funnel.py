"""
Логистическая регрессия для 4 шагов воронки первой покупки.

Если реального data/model_input/funnel_sessions_prepared.parquet нет, модуль
создает proxy-датасет на основе clients_prepared.parquet. Это нужно только для
демонстрации технической работоспособности блока CR первой покупки.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CLIENTS_PREPARED_PATH,
    FUNNEL_BASE_RATES,
    FUNNEL_PREPARED_PATH,
    FUNNEL_PROXY_PATH,
    MODELS_ARTIFACTS_DIR,
    RANDOM_SEED,
)

FUNNEL_STEPS = [
    {"name": "ctr_vitrine", "target": "had_click", "precondition": None},
    {"name": "cr_calc", "target": "had_calc", "precondition": "had_click"},
    {"name": "cr_form", "target": "had_form", "precondition": "had_calc"},
    {"name": "cr_payment", "target": "had_payment", "precondition": "had_form"},
]

NUMERIC_FEATURES = [
    "age_years",
    "tenure_bank_months",
    "products_count_min",
    "txn_count_90d_log",
    "txn_amount_90d_log",
    "avg_txn_amount_90d_log",
    "active_txn_days_90d",
    "days_since_last_txn",
    "day_of_week",
    "hour_of_day",
    "offer_price_rub",
]
CATEGORICAL_FEATURES = [
    "client_segment",
    "device_type",
    "offer_version",
]
BINARY_FEATURES = [
    "active_cc_flag",
    "active_loan_flag",
    "active_dc_flag",
    "ever_bought_insurance",
    "has_txn_90d",
    "is_weekend",
    "has_promo_banner",
]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def _series(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0
    return (s - s.mean()) / std


def generate_proxy_funnel_sessions(
    clients_path=CLIENTS_PREPARED_PATH,
    out_path=FUNNEL_PROXY_PATH,
    max_clients: int = 120_000,
) -> pd.DataFrame:
    """Создает proxy-сессии воронки из clients_prepared."""
    if not clients_path.exists():
        raise FileNotFoundError(f"Не найден {clients_path}. Нужен clients_prepared.parquet")

    clients = pd.read_parquet(clients_path).copy()
    if len(clients) > max_clients:
        clients = clients.sample(max_clients, random_state=RANDOM_SEED).copy()
    clients = clients.reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_SEED)
    df = clients.copy()
    df["session_id"] = np.arange(len(df), dtype=np.int64)
    df["device_type"] = rng.choice(["ios", "android", "web"], size=len(df), p=[0.35, 0.5, 0.15])
    df["offer_version"] = rng.choice(["base", "discount", "bundle"], size=len(df), p=[0.55, 0.3, 0.15])
    df["day_of_week"] = rng.integers(0, 7, size=len(df))
    df["hour_of_day"] = rng.integers(8, 23, size=len(df))
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["has_promo_banner"] = rng.binomial(1, 0.35, size=len(df))

    # Цена оффера приблизительно вокруг среднего чека клиента/продукта, но без использования target.
    base_price = np.maximum(_series(df, "avg_txn_amount_90d", 5000), 500)
    df["offer_price_rub"] = np.clip(base_price * rng.lognormal(mean=0.0, sigma=0.35, size=len(df)), 500, 100000)

    score = (
        0.55 * _z(_series(df, "txn_amount_90d_log", 0))
        + 0.35 * _z(_series(df, "txn_count_90d_log", 0))
        + 0.25 * _z(_series(df, "products_count_min", 0))
        - 0.20 * _z(_series(df, "days_since_last_txn", 0))
        + 0.25 * _series(df, "has_txn_90d", 0).astype(float)
        + 0.15 * _series(df, "ever_bought_insurance", 0).astype(float)
        + 0.15 * df["has_promo_banner"]
    )
    # Небольшое согласование с реальным target: положительные клиенты в среднем активнее воронки.
    if "bought_insurance_in_next_90d" in df.columns:
        score = score + 0.55 * df["bought_insurance_in_next_90d"].astype(float)

    p_click = sigmoid(logit(FUNNEL_BASE_RATES["ctr_vitrine"]) + 0.55 * score)
    df["had_click"] = rng.binomial(1, p_click)

    p_calc = sigmoid(logit(FUNNEL_BASE_RATES["cr_calc"]) + 0.35 * score)
    df["had_calc"] = np.where(df["had_click"] == 1, rng.binomial(1, p_calc), 0)

    p_form = sigmoid(logit(FUNNEL_BASE_RATES["cr_form"]) + 0.30 * score)
    df["had_form"] = np.where(df["had_calc"] == 1, rng.binomial(1, p_form), 0)

    p_payment = sigmoid(logit(FUNNEL_BASE_RATES["cr_payment"]) + 0.25 * score)
    df["had_payment"] = np.where(df["had_form"] == 1, rng.binomial(1, p_payment), 0)

    # Гарантируем наличие всех фичей.
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
    for col in BINARY_FEATURES:
        if col not in df.columns:
            df[col] = 0

    keep = ["session_id", "client_id"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + [s["target"] for s in FUNNEL_STEPS]
    df = df[keep].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Proxy funnel sessions saved: {out_path} ({len(df):,} rows)")
    return df


def load_sessions(sessions_path=None) -> pd.DataFrame:
    if sessions_path is None:
        if FUNNEL_PREPARED_PATH.exists():
            sessions_path = FUNNEL_PREPARED_PATH
        elif FUNNEL_PROXY_PATH.exists():
            sessions_path = FUNNEL_PROXY_PATH
        else:
            return generate_proxy_funnel_sessions()
    return pd.read_parquet(sessions_path)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
            ("bin", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), BINARY_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED)),
    ])


def safe_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_proba))


def evaluate(y_true, y_proba):
    y_label = (y_proba >= 0.5).astype(int)
    return {
        "auc_roc": safe_auc(y_true, y_proba),
        "f1": float(f1_score(y_true, y_label, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
        "base_rate": float(np.mean(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_label).tolist(),
    }


def train_step(sessions_df: pd.DataFrame, step: dict) -> dict:
    if step["precondition"] is None:
        df_step = sessions_df.copy()
    else:
        df_step = sessions_df[sessions_df[step["precondition"]] == 1].copy()

    if len(df_step) < 100 or df_step[step["target"]].nunique() < 2:
        return {"error": f"Недостаточно наблюдений или один класс: n={len(df_step)}, classes={df_step[step['target']].nunique()}"}

    X = df_step[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES]
    y = df_step[step["target"]].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test.to_numpy(), y_proba)

    # Средняя предсказанная вероятность по применимой популяции = forecast узла.
    p_all = pipeline.predict_proba(X)[:, 1]
    forecast_rate = float(np.mean(p_all))

    classifier = pipeline.named_steps["classifier"]
    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        coef = classifier.coef_[0]
        coefs = pd.DataFrame({"feature": feature_names, "coefficient": coef}).sort_values("coefficient", key=abs, ascending=False)
    except Exception:
        coefs = pd.DataFrame()

    return {
        "step_name": step["name"],
        "metrics": metrics,
        "forecast_rate": forecast_rate,
        "pipeline": pipeline,
        "coefficients": coefs,
    }


def train_all_funnel_models(sessions_path=None) -> dict:
    sessions_df = load_sessions(sessions_path)
    print(f"Загружено/сформировано сессий воронки: {len(sessions_df):,}")

    results = {}
    for step in FUNNEL_STEPS:
        print(f"\n--- Обучение: {step['name']} ---")
        result = train_step(sessions_df, step)
        if "error" in result:
            print(f"Ошибка: {result['error']}")
            results[step["name"]] = {"error": result["error"]}
            continue

        m = result["metrics"]
        print(f"base rate={m['base_rate']:.3f}, AUC={m['auc_roc']}, F1={m['f1']:.4f}, forecast={result['forecast_rate']:.4f}")

        artifact_path = MODELS_ARTIFACTS_DIR / f"logreg_{step['name']}.joblib"
        joblib.dump(result["pipeline"], artifact_path)

        results[step["name"]] = {
            "metrics": result["metrics"],
            "forecast_rate": result["forecast_rate"],
            "coefficients": result["coefficients"].head(100).to_dict(orient="records"),
            "artifact_path": str(artifact_path),
            "data_source": "real" if FUNNEL_PREPARED_PATH.exists() else "proxy_generated_from_clients",
        }

    report_path = MODELS_ARTIFACTS_DIR / "logreg_funnel_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Сводный отчёт сохранён: {report_path}")
    return results


if __name__ == "__main__":
    train_all_funnel_models()
