"""
LTV-модель: BG/NBD + Gamma-Gamma с fallback на эмпирический средний чек.

Вход: data/model_input/purchases_prepared.parquet
Единица наблюдения: покупочное событие.
Для денежной части используется purchase_amount_model, если колонка есть.
"""
from __future__ import annotations

import json
from datetime import timedelta

import cloudpickle
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.metrics import mean_absolute_error

from config import (
    LTV_DISCOUNT_RATE_MONTHLY,
    LTV_HOLDOUT_DAYS,
    LTV_HORIZONS_DAYS,
    MODELS_ARTIFACTS_DIR,
    PURCHASES_PREPARED_PATH,
)

BGNBD_PENALIZER = 0.001
GAMMA_PENALIZER = 0.001


def load_purchases(path=PURCHASES_PREPARED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path}. Сначала сформируй purchases_prepared.parquet")
    df = pd.read_parquet(path)
    df.columns = df.columns.str.lower()
    required = ["client_id", "purchase_date", "purchase_amount", "is_renewal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"purchases_prepared.parquet: нет колонок {missing}")
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    amount_col = "purchase_amount_model" if "purchase_amount_model" in df.columns else "purchase_amount"
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    df["is_renewal"] = pd.to_numeric(df["is_renewal"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["client_id", "purchase_date", amount_col])
    df = df[df[amount_col] > 0].copy()
    return df


def get_observation_end(df: pd.DataFrame) -> pd.Timestamp:
    return pd.to_datetime(df["purchase_date"].max()).normalize()


def build_rfm(purchases_df: pd.DataFrame, observation_period_end: pd.Timestamp | None = None) -> pd.DataFrame:
    df = purchases_df.copy()
    amount_col = "purchase_amount_model" if "purchase_amount_model" in df.columns else "purchase_amount"
    if observation_period_end is None:
        observation_period_end = get_observation_end(df)

    # Для LTV берем все страховые покупки как клиентские покупки. Это ближе к ARPPU/LTV
    # в текущем MVP, чем жесткое исключение renewals, потому что renewal тоже денежное событие.
    rfm = summary_data_from_transaction_data(
        transactions=df,
        customer_id_col="client_id",
        datetime_col="purchase_date",
        monetary_value_col=amount_col,
        observation_period_end=observation_period_end,
        freq="D",
    )
    # monetary_value для frequency=0 у lifetimes обычно 0; дальше имитируем медианой.
    return rfm


def fit_bgnbd(rfm: pd.DataFrame, penalizer: float = BGNBD_PENALIZER) -> BetaGeoFitter:
    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])
    return bgf


def fit_gamma_gamma(rfm: pd.DataFrame, penalizer: float = GAMMA_PENALIZER) -> GammaGammaFitter | None:
    repeat_buyers = rfm[(rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)].copy()
    if len(repeat_buyers) < 100:
        print(f"[ВНИМАНИЕ] Слишком мало repeat buyers для Gamma-Gamma: {len(repeat_buyers)}")
        return None

    corr = repeat_buyers[["frequency", "monetary_value"]].corr().iloc[0, 1]
    if abs(corr) > 0.3:
        print(f"[ВНИМАНИЕ] corr(frequency, monetary_value)={corr:.3f}; Gamma-Gamma может быть смещена")

    try:
        ggf = GammaGammaFitter(penalizer_coef=penalizer)
        ggf.fit(repeat_buyers["frequency"], repeat_buyers["monetary_value"])
        return ggf
    except Exception as e:
        print(f"[ВНИМАНИЕ] Gamma-Gamma не обучилась: {e}. Используем эмпирический средний чек.")
        return None


def monetary_fallback(rfm: pd.DataFrame) -> pd.Series:
    positive = rfm.loc[rfm["monetary_value"] > 0, "monetary_value"]
    median_monetary = positive.median() if len(positive) else 0.0
    return rfm["monetary_value"].where(rfm["monetary_value"] > 0, median_monetary)


def expected_monetary_value(rfm: pd.DataFrame, ggf: GammaGammaFitter | None) -> pd.Series:
    fallback = monetary_fallback(rfm)
    if ggf is None:
        return fallback
    try:
        expected = ggf.conditional_expected_average_profit(rfm["frequency"], fallback)
        if (expected <= 0).any() or expected.isna().any():
            print("[ВНИМАНИЕ] Gamma-Gamma дала неположительные/NaN значения. Используем fallback.")
            return fallback
        return expected
    except Exception as e:
        print(f"[ВНИМАНИЕ] Ошибка Gamma-Gamma predict: {e}. Используем fallback.")
        return fallback


def predict_clv(bgf: BetaGeoFitter, ggf: GammaGammaFitter | None, rfm: pd.DataFrame, horizon_days: int) -> pd.Series:
    expected_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
        horizon_days,
        rfm["frequency"],
        rfm["recency"],
        rfm["T"],
    )
    expected_monetary = expected_monetary_value(rfm, ggf)
    horizon_months = horizon_days / 30.0
    discount_factor = 1.0 / ((1 + LTV_DISCOUNT_RATE_MONTHLY) ** (horizon_months / 2))
    clv = expected_purchases * expected_monetary * discount_factor
    clv.name = f"clv_{horizon_days}d"
    return clv.clip(lower=0)


def evaluate_holdout(purchases_df: pd.DataFrame, holdout_days: int = LTV_HOLDOUT_DAYS) -> dict:
    obs_end = get_observation_end(purchases_df)
    cal_end = obs_end - timedelta(days=holdout_days)

    amount_col = "purchase_amount_model" if "purchase_amount_model" in purchases_df.columns else "purchase_amount"
    cal = purchases_df[purchases_df["purchase_date"] <= cal_end].copy()
    holdout = purchases_df[(purchases_df["purchase_date"] > cal_end) & (purchases_df["purchase_date"] <= obs_end)].copy()

    if len(cal) < 1000 or len(holdout) == 0:
        return {"error": "Недостаточно данных для holdout", "calibration_end": str(cal_end.date())}

    cal_rfm = build_rfm(cal, observation_period_end=cal_end)
    bgf = fit_bgnbd(cal_rfm)
    ggf = fit_gamma_gamma(cal_rfm)

    pred_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
        holdout_days,
        cal_rfm["frequency"],
        cal_rfm["recency"],
        cal_rfm["T"],
    )
    holdout_actual = holdout.groupby("client_id").agg(
        actual_purchases=("purchase_date", "count"),
        actual_amount=(amount_col, "sum"),
    )
    comparison = pd.DataFrame({"predicted_purchases": pred_purchases}).join(holdout_actual, how="left").fillna(0)
    predicted_amount = predict_clv(bgf, ggf, cal_rfm, holdout_days)
    comparison["predicted_amount"] = predicted_amount

    total_actual = float(comparison["actual_purchases"].sum())
    total_pred = float(comparison["predicted_purchases"].sum())
    return {
        "calibration_end": str(cal_end.date()),
        "observation_end": str(obs_end.date()),
        "holdout_days": int(holdout_days),
        "n_clients_calibration": int(len(cal_rfm)),
        "mae_purchases": float(mean_absolute_error(comparison["actual_purchases"], comparison["predicted_purchases"])),
        "mae_amount_rub": float(mean_absolute_error(comparison["actual_amount"], comparison["predicted_amount"])),
        "total_actual_purchases": int(total_actual),
        "total_predicted_purchases": total_pred,
        "total_bias_pct": float((total_pred - total_actual) / total_actual * 100) if total_actual > 0 else None,
    }


def train_and_predict(purchases_path=PURCHASES_PREPARED_PATH) -> dict:
    print("=== BG/NBD + Gamma-Gamma — LTV ===")
    purchases = load_purchases(purchases_path)
    obs_end = get_observation_end(purchases)
    print(f"Загружено покупок: {len(purchases):,}")
    print(f"Observation end: {obs_end.date()}")

    rfm = build_rfm(purchases, observation_period_end=obs_end)
    print(f"RFM клиентов: {len(rfm):,}")
    print(f"Repeat buyers frequency>0: {int((rfm['frequency'] > 0).sum()):,}")

    bgf = fit_bgnbd(rfm)
    ggf = fit_gamma_gamma(rfm)

    clv_results = {}
    clv_frame = rfm.copy()
    for horizon in LTV_HORIZONS_DAYS:
        clv = predict_clv(bgf, ggf, rfm, horizon)
        clv_frame[f"clv_{horizon}d"] = clv
        clv_results[f"clv_{horizon}d_rub"] = {
            "mean": float(clv.mean()),
            "median": float(clv.median()),
            "p25": float(clv.quantile(0.25)),
            "p75": float(clv.quantile(0.75)),
            "p95": float(clv.quantile(0.95)),
        }
        print(f"CLV {horizon}d: mean={clv.mean():,.0f}, median={clv.median():,.0f}")

    val_metrics = evaluate_holdout(purchases)
    if "error" in val_metrics:
        print(f"Holdout: {val_metrics['error']}")
    else:
        print(f"Holdout MAE purchases: {val_metrics['mae_purchases']:.4f}")
        print(f"Holdout bias: {val_metrics['total_bias_pct']:+.2f}%")

    bgnbd_path = MODELS_ARTIFACTS_DIR / "bgnbd_model.pkl"
    gamma_path = MODELS_ARTIFACTS_DIR / "gamma_gamma_model.pkl"
    rfm_path = MODELS_ARTIFACTS_DIR / "ltv_rfm_calibrated.parquet"

    with open(bgnbd_path, "wb") as f:
        cloudpickle.dump(bgf, f)
    if ggf is not None:
        with open(gamma_path, "wb") as f:
            cloudpickle.dump(ggf, f)
    clv_frame.to_parquet(rfm_path)

    report = {
        "model_type": "bgnbd_gamma_gamma_with_empirical_fallback",
        "observation_end": str(obs_end.date()),
        "n_clients_total": int(len(rfm)),
        "n_repeat_buyers": int((rfm["frequency"] > 0).sum()),
        "share_repeat_buyers": float((rfm["frequency"] > 0).mean()),
        "bgnbd_params": {k: float(v) for k, v in bgf.params_.items()},
        "gamma_gamma_used": ggf is not None,
        "gamma_gamma_params": {k: float(v) for k, v in ggf.params_.items()} if ggf is not None else None,
        "clv_predictions": clv_results,
        "holdout_validation": val_metrics,
    }
    report_path = MODELS_ARTIFACTS_DIR / "ltv_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Артефакты сохранены: {bgnbd_path.name}, {rfm_path.name}, {report_path.name}")
    return report


if __name__ == "__main__":
    train_and_predict()
