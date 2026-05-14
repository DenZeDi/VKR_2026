# Product Metrics Tree — Forecasting Prototype

Research prototype for the master's thesis:
**"Development of a tool for forecasting metric changes and generating recommendations in product metrics trees"**
HSE — Higher School of Business, programme "E-Business and Digital Innovations", 2026.

## Overview

Forecasts key drivers of a product metrics tree for a digital non-credit insurance sales channel and aggregates them into a top-level target metric. Supports scenario analysis and generates hypothesis-based managerial recommendations.

## Repository Structure

```
├── config.py                   # Paths and constants
├── run_all.py                  # Full pipeline runner
├── export_for_outside.py       # Safe artefact export (no personal data)
├── check_environment.py        # Dependency checker
├── models/
│   ├── xgb_penetration.py      # XGBoost penetration rate model
│   ├── shap_analysis.py        # SHAP interpretation (3 figures)
│   ├── cox_renewal.py          # Cox PH renewal model
│   ├── bgnbd_ltv.py            # BG/NBD + Gamma-Gamma LTV
│   └── logistic_funnel.py      # Logistic regression funnel models
├── tree/
│   ├── metric_tree.py          # Metrics tree structure and computation
│   └── aggregator.py           # Forecast aggregation pipeline
├── recommendations/
│   └── engine.py               # Recommendation generation
└── data/
    ├── processed/
    │   ├── preprocessing.py    # Raw → prepared parquet transformation
    │   └── csv2parquet.py      # CSV ingestion utility
    └── synthetic/              # Synthetic data generators (demo only)
```

## Quick Start

**1. Check dependencies:**
```bash
python check_environment.py
```

**2. Install (offline, recommended for closed environments):**
```bash
pip install --no-index --find-links=wheels_py310/ -r requirements_py310.txt
```

**3. Place prepared parquet files in `data/model_input/`:**
- `clients_prepared.parquet`
- `purchases_prepared.parquet`
- `policies_prepared.parquet`

**4. Run:**
```bash
python run_all.py

# Selective execution:
python run_all.py --skip-funnel --skip-ltv                         # XGBoost + SHAP + Cox + tree
python run_all.py --skip-funnel --skip-ltv --skip-renewal ...      # XGBoost + SHAP only
```

Flags: `--skip-funnel` `--skip-ltv` `--skip-xgb` `--skip-shap` `--skip-renewal` `--skip-tree` `--skip-recommendations`

**5. Export artefacts (no personal data):**
```bash
python export_for_outside.py
```

## Data Schema

**clients_prepared** — one row per client; target: `bought_insurance_in_next_90d`
**purchases_prepared** — one row per purchase event; used for RFM + BG/NBD
**policies_prepared** — one row per policy; survival target: `event_observed`, `duration_days`

## Notes

- Numeric values referencing specific clients use scaled figures per data confidentiality requirements.
- The funnel block uses a proxy dataset from `clients_prepared` when session-level data is unavailable.
- Random seed: `RANDOM_SEED = 42` in `config.py`.

## Requirements

Python 3.8 or 3.10 (Windows x64). See `requirements_py310.txt` / `requirements_py38.txt`.

Core: `pandas` `numpy` `scikit-learn` `xgboost` `lifelines` `lifetimes` `shap` `cloudpickle` `pyarrow` `joblib` `matplotlib`
