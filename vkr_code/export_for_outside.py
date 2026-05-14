"""
Safe export of training artifacts from a closed banking contour.

Copies only artefacts that contain no personal data (model weights, metrics,
JSON reports, SHAP figures) into safe_export/. Parquet files and any table
with client_id are explicitly excluded.

Run after a full pipeline execution:
    python export_for_outside.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from config import MODELS_ARTIFACTS_DIR, PROJECT_ROOT

SAFE_FILES = [
    ("models/artifacts/pipeline_summary.json",       "Pipeline summary: all model metrics and timing"),
    ("models/artifacts/logreg_funnel_report.json",    "LogReg funnel: AUC, F1, feature coefficients"),
    ("models/artifacts/ltv_report.json",              "BG/NBD + Gamma-Gamma: distribution params, holdout metrics"),
    ("models/artifacts/xgb_penetration_report.json",  "XGBoost: AUC, PR-AUC, feature importance"),
    ("models/artifacts/xgb_penetration_shap.json",    "SHAP: global importances, base value, figure metadata"),
    ("models/artifacts/cox_renewal_report.json",       "Cox PH: C-index, hazard ratios, p-values"),
    ("models/artifacts/logreg_ctr_vitrine.joblib",    "LogReg model — CTR vitrine"),
    ("models/artifacts/logreg_cr_calc.joblib",         "LogReg model — CR calculation"),
    ("models/artifacts/logreg_cr_form.joblib",         "LogReg model — CR form"),
    ("models/artifacts/logreg_cr_payment.joblib",      "LogReg model — CR payment"),
    ("models/artifacts/bgnbd_model.pkl",               "BG/NBD model — 4 distribution parameters"),
    ("models/artifacts/gamma_gamma_model.pkl",         "Gamma-Gamma model — 3 distribution parameters"),
    ("models/artifacts/xgb_penetration.pkl",           "XGBoost pipeline — trees + preprocessor"),
    ("models/artifacts/xgb_penetration_features.json", "XGBoost feature list for reproducibility"),
    ("models/artifacts/cox_renewal.pkl",               "Cox PH model — coefficients + baseline hazard"),
    ("models/artifacts/cox_renewal_features.json",     "Cox PH feature list for reproducibility"),
    ("models/artifacts/tree_baseline.json",            "Metrics tree: baseline node values"),
    ("models/artifacts/recommendations_demo.json",     "Demo recommendations with threshold rules"),
    ("reports/figures/shap_bar_importance.png",        "SHAP bar chart: global feature importance (top-15)"),
    ("reports/figures/shap_beeswarm_summary.png",      "SHAP beeswarm: per-observation value distribution"),
    ("reports/figures/shap_waterfall_highprob.png",    "SHAP waterfall: local explanation for high-prob observation"),
]

NEVER_EXPORT = [
    "models/artifacts/ltv_rfm_calibrated.parquet",
    "data/synthetic/clients.parquet",
    "data/synthetic/funnel_sessions.parquet",
    "data/synthetic/purchases.parquet",
    "data/synthetic/policies.parquet",
    "data/synthetic/transactions_agg.parquet",
    "data/processed",
]


def export_for_outside(export_dir: Path = PROJECT_ROOT / "safe_export") -> dict:
    """Copy allowed artefacts to export_dir. Raises if any forbidden file appears."""
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    (export_dir / "models" / "artifacts").mkdir(parents=True)
    (export_dir / "reports" / "figures").mkdir(parents=True)

    print(f"=== Exporting to {export_dir} ===\n")
    report = {"copied": [], "skipped_missing": []}

    for relative_path, description in SAFE_FILES:
        src = PROJECT_ROOT / relative_path
        dst = export_dir / relative_path
        if not src.exists():
            print(f"  [-] {relative_path}  (missing, skipped)")
            report["skipped_missing"].append(relative_path)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size_kb = src.stat().st_size / 1024
        print(f"  [+] {relative_path}  ({size_kb:.1f} KB)")
        report["copied"].append({"path": relative_path, "size_kb": round(size_kb, 1), "description": description})

    readme_path = export_dir / "README.md"
    readme_path.write_text(_build_readme(report), encoding="utf-8")
    print("\n  [+] README.md")

    manifest_path = export_dir / "export_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Security check ===")
    forbidden = [str(export_dir / p) for p in NEVER_EXPORT if (export_dir / p).exists()]
    if forbidden:
        raise RuntimeError(f"Forbidden files found in export: {forbidden}")
    print("  [ok] No forbidden files.")

    total_kb = sum(c["size_kb"] for c in report["copied"])
    print(f"\n  Copied: {len(report['copied'])} files ({total_kb:.1f} KB)")
    print(f"  Skipped: {len(report['skipped_missing'])} missing files")
    return report


def _build_readme(report: dict) -> str:
    lines = [
        "# Model Training Artefacts — Safe Export",
        "",
        "Contains trained model weights, quality metrics, aggregated results,",
        "and SHAP figures. No personal data (no client_id, no raw transactions).",
        "",
        "## Contents",
        "",
    ]
    for item in report["copied"]:
        lines.append(f"- **{item['path']}** ({item['size_kb']} KB) — {item['description']}")
    lines += [
        "",
        "## Dependencies",
        "",
        "pandas, numpy, scikit-learn, xgboost, lifelines, lifetimes, shap",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    export_for_outside()
