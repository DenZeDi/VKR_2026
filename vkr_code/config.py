"""
Project-wide paths and constants.

Expects three prepared parquet files in data/model_input/:
    purchases_prepared.parquet
    clients_prepared.parquet
    policies_prepared.parquet

The funnel block falls back to a proxy dataset built from clients_prepared
when funnel_sessions_prepared.parquet is unavailable.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR             = PROJECT_ROOT / "data"
SYNTHETIC_DIR        = DATA_DIR / "synthetic"
PROCESSED_DIR        = DATA_DIR / "processed"
MODEL_INPUT_DIR      = DATA_DIR / "model_input"
REPORTS_DIR          = PROJECT_ROOT / "reports"
FIGURES_DIR          = REPORTS_DIR / "figures"
MODELS_ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

for path in (SYNTHETIC_DIR, PROCESSED_DIR, MODEL_INPUT_DIR, FIGURES_DIR, MODELS_ARTIFACTS_DIR):
    path.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

CUTOFF_DATE   = "2026-01-31"
SNAPSHOT_DATE = "2026-01-31"

PURCHASES_PREPARED_PATH = MODEL_INPUT_DIR / "purchases_prepared.parquet"
CLIENTS_PREPARED_PATH   = MODEL_INPUT_DIR / "clients_prepared.parquet"
POLICIES_PREPARED_PATH  = MODEL_INPUT_DIR / "policies_prepared.parquet"
FUNNEL_PREPARED_PATH    = MODEL_INPUT_DIR / "funnel_sessions_prepared.parquet"
FUNNEL_PROXY_PATH       = MODEL_INPUT_DIR / "funnel_sessions_proxy.parquet"

LTV_HORIZONS_DAYS        = [90, 180, 365]
LTV_DISCOUNT_RATE_MONTHLY = 0.01
LTV_HOLDOUT_DAYS          = 90

FUNNEL_BASE_RATES = {
    "ctr_vitrine": 0.12,
    "cr_calc":     0.55,
    "cr_form":     0.45,
    "cr_payment":  0.70,
}
