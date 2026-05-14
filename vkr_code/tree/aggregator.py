"""
Агрегатор дерева метрик под финальный MVP.

Источники прогнозов:
  - logreg_funnel_report.json: CTR/CR шагов воронки;
  - xgb_penetration_report.json: penetration_rate;
  - cox_renewal_report.json: renewal_rate;
  - ltv_report.json: LTV 365d.

Такой режим не требует повторно применять модели к персональным строкам при
сценарном анализе и безопаснее для выноса результатов из контура.
"""
from __future__ import annotations

import json
from pathlib import Path

from config import MODELS_ARTIFACTS_DIR
from tree.metric_tree import ALL_NODES, get_node, nodes_by_type


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_models(artifacts_dir: Path = MODELS_ARTIFACTS_DIR) -> dict[str, float]:
    forecasts: dict[str, float] = {}

    # Воронка
    funnel_report = _read_json(artifacts_dir / "logreg_funnel_report.json")
    for node_name in ["ctr_vitrine", "cr_calc", "cr_form", "cr_payment"]:
        payload = funnel_report.get(node_name, {})
        forecasts[node_name] = payload.get("forecast_rate") or payload.get("metrics", {}).get("base_rate")

    # Повторный вход пока не обучается отдельной моделью в MVP.
    forecasts["coef_repeat_entry"] = 0.25

    # Проникновение
    xgb_report = _read_json(artifacts_dir / "xgb_penetration_report.json")
    forecasts["penetration_rate"] = xgb_report.get("historical_penetration")

    # Продление
    renewal_report = _read_json(artifacts_dir / "cox_renewal_report.json")
    forecasts["renewal_rate"] = renewal_report.get("aggregate_renewal_rate") or renewal_report.get("actual_renewal_rate")

    # LTV
    ltv_report = _read_json(artifacts_dir / "ltv_report.json")
    forecasts["ltv"] = ltv_report.get("clv_predictions", {}).get("clv_365d_rub", {}).get("mean")

    return forecasts


def compute_tree(observed_inputs: dict[str, float], forecasts: dict[str, float]) -> dict[str, float]:
    values: dict[str, float] = {}

    for node in nodes_by_type("observed"):
        if node.name not in observed_inputs:
            raise ValueError(f"Не задан наблюдаемый узел {node.name}: {node.display_name}")
        values[node.name] = float(observed_inputs[node.name])

    for node in nodes_by_type("forecasted"):
        if node.name not in forecasts or forecasts[node.name] is None:
            raise ValueError(f"Не задан прогнозный узел {node.name}: {node.display_name}")
        values[node.name] = float(forecasts[node.name])

    for level in [4, 3, 2, 1]:
        remaining = [n for n in ALL_NODES if n.level == level and n.node_type == "computed"]
        max_iterations = max(1, len(remaining) ** 2)
        it = 0
        while remaining and it < max_iterations:
            it += 1
            progressed = False
            for node in list(remaining):
                if all(dep in values for dep in node.formula_inputs):
                    values[node.name] = float(node.formula(values))
                    remaining.remove(node)
                    progressed = True
            if not progressed:
                break
        if remaining:
            raise ValueError(f"Не удалось посчитать уровень {level}: {[n.name for n in remaining]}")

    return values


def scenario_analysis(observed_inputs: dict[str, float], forecasts: dict[str, float], overrides: dict[str, float]) -> dict[str, dict]:
    baseline = compute_tree(observed_inputs, forecasts)
    obs2 = dict(observed_inputs)
    fc2 = dict(forecasts)
    for node_name, value in overrides.items():
        node = get_node(node_name)
        if node.node_type == "observed":
            obs2[node_name] = value
        elif node.node_type == "forecasted":
            fc2[node_name] = value
        else:
            raise ValueError(f"Расчётный узел {node_name} нельзя менять напрямую; меняй зависимости {node.formula_inputs}")
    scenario = compute_tree(obs2, fc2)
    delta_abs = {k: scenario[k] - baseline[k] for k in baseline}
    delta_pct = {k: (scenario[k] - baseline[k]) / baseline[k] * 100 if baseline[k] else 0 for k in baseline}
    return {"baseline": baseline, "scenario": scenario, "delta_abs": delta_abs, "delta_pct": delta_pct, "overrides": overrides}


# Наблюдаемые узлы для демонстрации MVP. Их можно заменить на актуальные агрегаты периода.
DEFAULT_OBSERVED_INPUTS = {
    "communications_reach": 500_000,
    "communications_ctr": 0.025,
    "bank_clients": 2_000_000,
    "previously_active_clients": 50_000,
    "active_policies": 150_000,
    "resurrection_rate": 0.04,
    "coef_cross_sell": 0.08,
}


def run_default_pipeline() -> dict:
    print("=== Агрегация прогнозов моделей ===")
    forecasts = apply_models()
    for name, val in forecasts.items():
        if val is None:
            print(f"  {name:<25s}: НЕТ ДАННЫХ")
        elif abs(val) < 1:
            print(f"  {name:<25s}: {val:.4f}")
        else:
            print(f"  {name:<25s}: {val:,.0f}")

    print("\n=== Расчёт дерева ===")
    tree_values = compute_tree(DEFAULT_OBSERVED_INPUTS, forecasts)
    for level in range(1, 6):
        print(f"\nУровень {level}")
        for node in [n for n in ALL_NODES if n.level == level]:
            val = tree_values[node.name]
            fmt = f"{val:.4f}" if abs(val) < 1 else f"{val:,.0f}"
            print(f"  {node.name:<28s} {node.display_name:<38s} = {fmt}")

    scenarios = [
        ("CR оформления +5 п.п.", {"cr_form": min(forecasts["cr_form"] + 0.05, 1.0)}),
        ("Коэффициент продления -3 п.п.", {"renewal_rate": max(forecasts["renewal_rate"] - 0.03, 0.0)}),
        ("Охват коммуникаций +20%", {"communications_reach": DEFAULT_OBSERVED_INPUTS["communications_reach"] * 1.2}),
    ]
    scenario_results = {}
    print("\n=== Сценарный анализ ===")
    for label, overrides in scenarios:
        result = scenario_analysis(DEFAULT_OBSERVED_INPUTS, forecasts, overrides)
        scenario_results[label] = result
        b = result["baseline"]["nsm"]
        s = result["scenario"]["nsm"]
        print(f"{label}: NSM {b:,.0f} -> {s:,.0f} ({result['delta_pct']['nsm']:+.2f}%)")

    out = {
        "observed_inputs": DEFAULT_OBSERVED_INPUTS,
        "forecasts": forecasts,
        "tree_values": tree_values,
        "scenario_results": scenario_results,
    }
    out_path = MODELS_ARTIFACTS_DIR / "tree_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Baseline дерева сохранён: {out_path}")
    return tree_values


if __name__ == "__main__":
    run_default_pipeline()
