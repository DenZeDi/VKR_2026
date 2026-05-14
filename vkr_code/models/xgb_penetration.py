"""
XGBoost-модель коэффициента проникновения.

Вход: data/model_input/clients_prepared.parquet
Единица наблюдения: 1 строка = 1 клиент на дату среза.
Target: bought_insurance_in_next_90d.

Код не ожидает synthetic transactions_agg: все транзакционные агрегаты уже
встроены в clients_prepared.parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import CLIENTS_PREPARED_PATH, MODELS_ARTIFACTS_DIR, RANDOM_SEED


TARGET_COL = "bought_insurance_in_next_90d"
WEIGHT_COL = "sample_weight"

ID_COLS = {
    "client_id", "client_uk", "birth_date", "client_start_date", "snapshot_date",
    TARGET_COL, WEIGHT_COL,
}

NUMERIC_FEATURES_CANDIDATES = [
    "age_years",
    "tenure_bank_months",
    "active_cc_flag",
    "active_loan_flag",
    "active_dc_flag",
    "products_count_min",
    "prod_client_flag",
    "ever_bought_insurance",
    "has_txn_90d",
    "txn_count_30d_log",
    "txn_count_90d_log",
    "txn_amount_30d_log",
    "txn_amount_90d_log",
    "avg_txn_amount_90d_log",
    "active_txn_days_90d",
    "days_since_last_txn",
]

CATEGORICAL_FEATURES_CANDIDATES = [
    "gender_code",
    "country_code",
    "city_name",
    "client_segment",
]

XGB_PARAMS = {
    "n_estimators": 350,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 20,
    "gamma": 0.1,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
}


def _existing(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def load_clients(path: Path = CLIENTS_PREPARED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path}. Сначала сформируй clients_prepared.parquet")
    df = pd.read_parquet(path)
    df.columns = df.columns.str.lower()
    if TARGET_COL not in df.columns:
        raise ValueError(f"В clients_prepared.parquet нет target-колонки {TARGET_COL}")
    if df.duplicated("client_id").sum() > 0:
        raise ValueError("clients_prepared содержит дубли client_id. Проверь предобработку.")
    return df


def select_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = _existing(NUMERIC_FEATURES_CANDIDATES, df)
    categorical = _existing(CATEGORICAL_FEATURES_CANDIDATES, df)
    if not numeric and not categorical:
        raise ValueError("Не найдено ни одного признака для XGB-модели")
    return numeric, categorical


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    clf = xgb.XGBClassifier(**XGB_PARAMS)

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", clf),
    ])


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.02, 0.98, 49)
    scores = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def safe_auc(y_true, y_score) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    best_threshold, best_f1 = find_best_threshold(y_true, y_proba)
    y_pred_best = (y_proba >= best_threshold).astype(int)
    return {
        "auc_roc": safe_auc(y_true, y_proba),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "threshold_05": 0.5,
        "f1_at_05": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_at_05": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_05": float(recall_score(y_true, y_pred, zero_division=0)),
        "best_threshold": best_threshold,
        "f1_best": best_f1,
        "precision_best": float(precision_score(y_true, y_pred_best, zero_division=0)),
        "recall_best": float(recall_score(y_true, y_pred_best, zero_division=0)),
        "confusion_matrix_at_05": confusion_matrix(y_true, y_pred).tolist(),
        "confusion_matrix_best": confusion_matrix(y_true, y_pred_best).tolist(),
        "n_test": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
        "base_rate": float(np.mean(y_true)),
    }


def get_feature_importance(pipeline: Pipeline, numeric_features: list[str], categorical_features: list[str]) -> list[dict]:
    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocess"]
    try:
        feature_names = pre.get_feature_names_out().tolist()
    except Exception:
        feature_names = numeric_features + categorical_features

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(100)
        .to_dict(orient="records")
    )


def train_and_save(clients_path: Path = CLIENTS_PREPARED_PATH) -> dict:
    print("=== XGBoost — коэффициент проникновения ===")
    df = load_clients(clients_path)
    numeric_features, categorical_features = select_features(df)

    y = df[TARGET_COL].astype(int)
    X = df[numeric_features + categorical_features]
    sample_weight = df[WEIGHT_COL].astype(float) if WEIGHT_COL in df.columns else None

    print(f"Клиентов: {len(df):,}")
    print(f"Target positive: {int(y.sum()):,} ({y.mean()*100:.2f}%)")
    print(f"Числовых признаков: {len(numeric_features)}, категориальных: {len(categorical_features)}")

    split_kwargs = dict(test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    if sample_weight is not None:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, sample_weight, **split_kwargs)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
        w_train = w_test = None

    pipeline = build_pipeline(numeric_features, categorical_features)
    fit_kwargs = {}
    if w_train is not None:
        fit_kwargs["model__sample_weight"] = w_train
    pipeline.fit(X_train, y_train, **fit_kwargs)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test.to_numpy(), y_proba)

    historical_penetration = float(y.mean())
    calibrated_threshold = float(np.quantile(y_proba, 1 - historical_penetration))
    predicted_penetration_at_calibrated = float((y_proba >= calibrated_threshold).mean())

    print(f"AUC-ROC: {metrics['auc_roc']:.4f}" if metrics["auc_roc"] is not None else "AUC-ROC: n/a")
    print(f"PR-AUC:  {metrics['pr_auc']:.4f}")
    print(f"F1 best: {metrics['f1_best']:.4f} при threshold={metrics['best_threshold']:.3f}")
    print(f"Калиброванный threshold: {calibrated_threshold:.4f}")

    model_path = MODELS_ARTIFACTS_DIR / "xgb_penetration.pkl"
    with open(model_path, "wb") as f:
        cloudpickle.dump(pipeline, f)

    features_payload = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET_COL,
    }
    with open(MODELS_ARTIFACTS_DIR / "xgb_penetration_features.json", "w", encoding="utf-8") as f:
        json.dump(features_payload, f, indent=2, ensure_ascii=False)

    report = {
        "model_type": "xgboost_pipeline",
        "n_clients": int(len(df)),
        "n_features_raw": int(len(numeric_features) + len(categorical_features)),
        "target": TARGET_COL,
        "target_positive": int(y.sum()),
        "historical_penetration": historical_penetration,
        "calibrated_threshold": calibrated_threshold,
        "predicted_penetration_at_calibrated": predicted_penetration_at_calibrated,
        "metrics": metrics,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_importance": get_feature_importance(pipeline, numeric_features, categorical_features),
        "xgb_params": XGB_PARAMS,
    }
    report_path = MODELS_ARTIFACTS_DIR / "xgb_penetration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Артефакты сохранены: {model_path.name}, {report_path.name}")
    return report


if __name__ == "__main__":
    train_and_save()
