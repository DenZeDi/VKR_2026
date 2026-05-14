"""
Единый запуск финального MVP-пайплайна на подготовленных реальных данных.

Перед запуском должны лежать:
    data/model_input/purchases_prepared.parquet
    data/model_input/clients_prepared.parquet
    data/model_input/policies_prepared.parquet

Запуск:
    python run_all.py

Опционально:
    python run_all.py --skip-funnel
    python run_all.py --skip-ltv
    python run_all.py --skip-xgb
    python run_all.py --skip-renewal
    python run_all.py --skip-tree
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_step(name: str, callable_fn, *args, **kwargs):
    print("\n" + "=" * 80)
    print(f"ШАГ: {name}")
    print("=" * 80)
    start = time.time()
    try:
        result = callable_fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"\n[{name}] OK за {elapsed:.1f} сек.")
        return {"status": "ok", "result": result, "elapsed_sec": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n[{name}] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e), "elapsed_sec": elapsed}


def extract_key_metrics(step_name: str, report: dict) -> dict:
    if not isinstance(report, dict):
        return {}
    if step_name == "logreg_funnel":
        return {
            step: {
                "base_rate": payload.get("metrics", {}).get("base_rate"),
                "auc_roc": payload.get("metrics", {}).get("auc_roc"),
                "forecast_rate": payload.get("forecast_rate"),
            }
            for step, payload in report.items()
            if isinstance(payload, dict)
        }
    if step_name == "ltv":
        return {
            "n_clients_total": report.get("n_clients_total"),
            "n_repeat_buyers": report.get("n_repeat_buyers"),
            "clv_365_mean": report.get("clv_predictions", {}).get("clv_365d_rub", {}).get("mean"),
            "mae_purchases": report.get("holdout_validation", {}).get("mae_purchases"),
            "total_bias_pct": report.get("holdout_validation", {}).get("total_bias_pct"),
        }
    if step_name == "xgb_penetration":
        m = report.get("metrics", {})
        return {
            "historical_penetration": report.get("historical_penetration"),
            "auc_roc": m.get("auc_roc"),
            "pr_auc": m.get("pr_auc"),
            "f1_best": m.get("f1_best"),
            "best_threshold": m.get("best_threshold"),
        }
    if step_name == "cox_renewal":
        return {
            "actual_renewal_rate": report.get("actual_renewal_rate"),
            "aggregate_renewal_rate": report.get("aggregate_renewal_rate"),
            "c_index_test": report.get("metrics", {}).get("c_index_test"),
        }
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-funnel", action="store_true")
    parser.add_argument("--skip-ltv", action="store_true")
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-renewal", action="store_true")
    parser.add_argument("--skip-tree", action="store_true")
    parser.add_argument("--skip-recommendations", action="store_true")
    args = parser.parse_args()

    from config import MODELS_ARTIFACTS_DIR

    pipeline_results = {}

    if not args.skip_funnel:
        from models.logistic_funnel import train_all_funnel_models
        pipeline_results["logreg_funnel"] = run_step(
            "Logistic Regression — proxy/real воронка первой покупки",
            train_all_funnel_models,
        )

    if not args.skip_ltv:
        from models.bgnbd_ltv import train_and_predict
        pipeline_results["ltv"] = run_step(
            "BG/NBD + Gamma-Gamma/fallback — LTV",
            train_and_predict,
        )

    if not args.skip_xgb:
        from models.xgb_penetration import train_and_save
        pipeline_results["xgb_penetration"] = run_step(
            "XGBoost — коэффициент проникновения",
            train_and_save,
        )

    if not args.skip_shap:
        _pkl = MODELS_ARTIFACTS_DIR / "xgb_penetration.pkl"
        if args.skip_xgb and not _pkl.exists():
            print("[shap_analysis] Skipped: xgb_penetration.pkl not found and --skip-xgb is set.")
        else:
            from models.shap_analysis import run_shap_analysis
            pipeline_results["shap_analysis"] = run_step(
                "SHAP-анализ — коэффициент проникновения",
                run_shap_analysis,
            )

    if not args.skip_renewal:
        from models.cox_renewal import train_and_save
        pipeline_results["cox_renewal"] = run_step(
            "Cox PH — коэффициент продления",
            train_and_save,
        )

    if not args.skip_tree:
        from tree.aggregator import run_default_pipeline
        pipeline_results["tree_aggregation"] = run_step(
            "Дерево метрик — агрегация прогнозов",
            run_default_pipeline,
        )

    if not args.skip_recommendations:
        from recommendations.engine import demo
        pipeline_results["recommendations"] = run_step(
            "Демо-рекомендации",
            demo,
        )

    print("\n" + "=" * 80)
    print("СВОДКА")
    print("=" * 80)

    summary = {"steps": {}, "key_metrics": {}}
    total_time = 0.0
    for step_name, info in pipeline_results.items():
        total_time += info["elapsed_sec"]
        summary["steps"][step_name] = {
            "status": info["status"],
            "elapsed_sec": round(info["elapsed_sec"], 1),
        }
        if info["status"] == "ok":
            summary["key_metrics"][step_name] = extract_key_metrics(step_name, info.get("result"))
        print(f"{step_name:<35s} {info['status']:<8s} {info['elapsed_sec']:>8.1f} сек")

    summary["total_time_sec"] = round(total_time, 1)

    out_path = MODELS_ARTIFACTS_DIR / "pipeline_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nСводный отчёт: {out_path}")
    print("Готово.")


if __name__ == "__main__":
    main()
