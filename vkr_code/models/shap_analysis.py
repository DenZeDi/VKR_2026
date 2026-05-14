"""
SHAP analysis for the XGBoost penetration rate model.

Loads the trained pipeline from models/artifacts/xgb_penetration.pkl,
reproduces the identical train/test split (same random seed), transforms
features through the fitted preprocessor, and produces three figures:

    reports/figures/shap_bar_importance.png
    reports/figures/shap_beeswarm_summary.png
    reports/figures/shap_waterfall_highprob.png

Also writes models/artifacts/xgb_penetration_shap.json.

Usage:
    python models/shap_analysis.py
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import cloudpickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CLIENTS_PREPARED_PATH, FIGURES_DIR, MODELS_ARTIFACTS_DIR, RANDOM_SEED

TARGET_COL = "bought_insurance_in_next_90d"
WEIGHT_COL = "sample_weight"
TOP_N = 15

FEATURE_NAMES_RU: dict[str, str] = {
    "age_years":              "Возраст клиента, лет",
    "tenure_bank_months":     "Срок в банке, мес.",
    "active_cc_flag":         "Активная кредитная карта",
    "active_loan_flag":       "Активный кредит",
    "active_dc_flag":         "Активная дебетовая карта",
    "products_count_min":     "Кол-во банковских продуктов",
    "prod_client_flag":       "Флаг продуктового клиента",
    "ever_bought_insurance":  "Покупал страховку ранее",
    "has_txn_90d":            "Есть транзакции за 90 дней",
    "txn_count_30d_log":      "Кол-во транзакций, 30 дн. (log)",
    "txn_count_90d_log":      "Кол-во транзакций, 90 дн. (log)",
    "txn_amount_30d_log":     "Сумма транзакций, 30 дн. (log)",
    "txn_amount_90d_log":     "Сумма транзакций, 90 дн. (log)",
    "avg_txn_amount_90d_log": "Средний чек, 90 дн. (log)",
    "active_txn_days_90d":    "Дней с транзакциями за 90 дн.",
    "days_since_last_txn":    "Дней с последней транзакции",
    "gender_code":            "Пол клиента",
    "country_code":           "Страна клиента",
    "city_name":              "Город клиента",
    "client_segment":         "Сегмент клиента",
}

_BLUE = "#2563EB"
_RED  = "#DC2626"
_GRAY = "#6B7280"


def _load_pipeline(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run xgb_penetration.py first.")
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def _get_X_test(clients_path: Path):
    """Reproduce the same train/test split as in xgb_penetration.py."""
    from models.xgb_penetration import load_clients, select_features

    df = load_clients(clients_path)
    numeric_features, categorical_features = select_features(df)
    y = df[TARGET_COL].astype(int)
    X = df[numeric_features + categorical_features]
    sample_weight = df[WEIGHT_COL].astype(float) if WEIGHT_COL in df.columns else None

    split_kwargs = dict(test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    if sample_weight is not None:
        _, X_test, _, y_test, _, _ = train_test_split(X, y, sample_weight, **split_kwargs)
    else:
        _, X_test, _, y_test = train_test_split(X, y, **split_kwargs)
    return X_test, y_test


def _get_transformed_feature_names(pipeline) -> list[str]:
    """Map ColumnTransformer output names to Russian labels for charts."""
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        raw_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []

    result = []
    for name in raw_names:
        if name.startswith("num__"):
            key = name[5:]
            result.append(FEATURE_NAMES_RU.get(key, key))
        elif name.startswith("cat__"):
            rest = name[5:]
            matched = False
            for base_key in FEATURE_NAMES_RU:
                if rest.startswith(base_key):
                    value = rest[len(base_key):].lstrip("_")
                    ru_base = FEATURE_NAMES_RU[base_key]
                    result.append(f"{ru_base} = {value}" if value else ru_base)
                    matched = True
                    break
            if not matched:
                result.append(rest)
        else:
            result.append(name)
    return result


def _plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 150,
    })


def _plot_bar(shap_values: np.ndarray, feature_names: list[str], out_path: Path) -> list[dict]:
    """Bar chart of global feature importance (mean |SHAP|)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df_imp = (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=True)
        .tail(TOP_N)
    )
    fig, ax = plt.subplots(figsize=(9, max(4, len(df_imp) * 0.45)))
    bars = ax.barh(df_imp["feature"], df_imp["importance"], color=_BLUE, height=0.65, alpha=0.90)
    for bar, val in zip(bars, df_imp["importance"]):
        ax.text(val + max(mean_abs) * 0.008, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9, color=_GRAY)
    ax.set_xlabel("Среднее |SHAP-значение| (вклад в прогноз)", labelpad=8)
    ax.set_title(f"Глобальная важность признаков — топ-{TOP_N}\nМодель коэффициента проникновения (XGBoost + SHAP)")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
        .to_dict(orient="records")
    )


def _plot_beeswarm(shap_values: np.ndarray, X_transformed: np.ndarray,
                   feature_names: list[str], out_path: Path) -> None:
    """SHAP beeswarm summary plot (top-N features by global importance)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:TOP_N]
    shap_top = shap_values[:, top_idx]
    X_top = pd.DataFrame(X_transformed[:, top_idx], columns=[feature_names[i] for i in top_idx])

    fig, ax = plt.subplots(figsize=(10, max(5, TOP_N * 0.45)))
    shap.summary_plot(shap_top, X_top, show=False, max_display=TOP_N,
                      plot_size=None, color_bar_label="Значение признака")
    ax = plt.gca()
    ax.set_xlabel("SHAP-значение (влияние на вероятность проникновения)", labelpad=8)
    ax.set_title("Распределение SHAP-значений по наблюдениям\nМодель коэффициента проникновения (XGBoost + SHAP)")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_waterfall(shap_values: np.ndarray, feature_names: list[str],
                    base_value: float, y_proba: np.ndarray, out_path: Path) -> tuple[int, float]:
    """
    Waterfall plot for an observation with high predicted probability.

    Selects the observation with the maximum SHAP value for the globally
    dominant feature (identified via argmax of mean |SHAP|, which resolves
    robustly to ever_bought_insurance regardless of feature name encoding).
    Appends an 'Other features' residual bar so the chart total equals
    the actual model output.
    """
    top_feat_idx = int(np.argmax(np.abs(shap_values).mean(axis=0)))
    obs_idx = int(np.argmax(shap_values[:, top_feat_idx]))
    shap_row = shap_values[obs_idx]

    order = np.argsort(np.abs(shap_row))[::-1][:TOP_N]
    sv_top = shap_row[order]
    fn_top = [feature_names[i] for i in order]

    actual_log_odds = base_value + float(shap_row.sum())
    residual = actual_log_odds - (base_value + float(sv_top.sum()))
    if abs(residual) > 0.05:
        sv_top = np.append(sv_top, residual)
        fn_top.append(f"Прочие ({len(shap_row) - len(order)} признаков)")

    cumulative = base_value + np.concatenate([[0], np.cumsum(sv_top)])
    colors = [_RED if v >= 0 else _BLUE for v in sv_top]
    y_pos = list(range(len(sv_top) - 1, -1, -1))

    fig, ax = plt.subplots(figsize=(10, max(5, len(sv_top) * 0.48)))
    for i, (y, val, col) in enumerate(zip(y_pos, sv_top, colors)):
        left = cumulative[len(sv_top) - 1 - i]
        ax.barh(y, val, left=left, color=col, height=0.58, alpha=0.85)
        label = f"+{val:.4f}" if val >= 0 else f"{val:.4f}"
        offset = max(np.abs(sv_top)) * 0.015
        ax.text(left + val + (offset if val >= 0 else -offset), y, label,
                va="center", ha="left" if val >= 0 else "right",
                fontsize=8.5, color=_GRAY)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(fn_top, fontsize=10)
    ax.axvline(base_value, color="black", linestyle="--", linewidth=0.9, alpha=0.55,
               label=f"Базовое значение ({base_value:.3f})")
    ax.axvline(actual_log_odds, color=_RED, linestyle=":", linewidth=1.1, alpha=0.75,
               label=f"Прогноз ({actual_log_odds:.3f}) ≈ p={1/(1+math.exp(-actual_log_odds)):.3f}")
    ax.set_xlabel("Логарифм шансов (log-odds) — шкала XGBoost", labelpad=8)
    ax.set_title(f"Локальная интерпретация — наблюдение с высокой P(покупки) (idx={obs_idx})\nМодель коэффициента проникновения (XGBoost + SHAP)")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return obs_idx, actual_log_odds


def run_shap_analysis(
    clients_path: Path = CLIENTS_PREPARED_PATH,
    model_path: Path = MODELS_ARTIFACTS_DIR / "xgb_penetration.pkl",
) -> dict:
    """Load model, compute SHAP values, generate three figures, write JSON report."""
    print("=== SHAP — коэффициент проникновения ===")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = _load_pipeline(model_path)
    xgb_model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    X_test_raw, y_test = _get_X_test(clients_path)
    print(f"  Test set: {X_test_raw.shape[0]:,} obs, {X_test_raw.shape[1]} features")

    X_transformed = preprocessor.transform(X_test_raw)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.array(X_transformed, dtype=np.float32)

    feature_names = _get_transformed_feature_names(pipeline)
    if len(feature_names) != X_transformed.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    base_value = float(
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )
    y_proba = pipeline.predict_proba(X_test_raw)[:, 1]

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test.to_numpy(), y_proba)
    print(f"  AUC-ROC: {auc:.4f} | base_value (log-odds): {base_value:.4f}")

    _plot_style()
    path_bar       = FIGURES_DIR / "shap_bar_importance.png"
    path_beeswarm  = FIGURES_DIR / "shap_beeswarm_summary.png"
    path_waterfall = FIGURES_DIR / "shap_waterfall_highprob.png"

    importance_list = _plot_bar(shap_values, feature_names, path_bar)
    _plot_beeswarm(shap_values, X_transformed, feature_names, path_beeswarm)
    obs_idx, log_odds_obs = _plot_waterfall(shap_values, feature_names, base_value, y_proba, path_waterfall)
    print(f"  Waterfall: idx={obs_idx}, proba={y_proba[obs_idx]:.4f}, log-odds={log_odds_obs:.3f}")

    report = {
        "model_type": "xgboost_shap",
        "n_test_observations": int(X_test_raw.shape[0]),
        "n_features_transformed": int(X_transformed.shape[1]),
        "base_value": base_value,
        "auc_roc_check": float(auc),
        "observation_idx": obs_idx,
        "observation_proba": float(y_proba[obs_idx]),
        "observation_log_odds": float(log_odds_obs),
        "top_features_global": importance_list[:TOP_N],
        "figures": {
            "bar":       str(path_bar.relative_to(path_bar.parent.parent.parent)),
            "beeswarm":  str(path_beeswarm.relative_to(path_beeswarm.parent.parent.parent)),
            "waterfall": str(path_waterfall.relative_to(path_waterfall.parent.parent.parent)),
        },
    }
    report_path = MODELS_ARTIFACTS_DIR / "xgb_penetration_shap.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report: {report_path}")
    return report


if __name__ == "__main__":
    run_shap_analysis()
