"""
Структура дерева продуктовых метрик.

Дерево содержит 25 узлов на 5 уровнях иерархии. Из них:
    - 8 прогнозируемых узлов (CTR витрины, CR расчёта, CR оформления, CR оплаты,
      коэффициент повторного входа, коэффициент проникновения, коэффициент
      продления, LTV)
    - Остальные — расчётные (агрегаты по формулам) и наблюдаемые (входные
      данные из систем).

Каждый узел представлен NodeSpec — спецификацией с типом, формулой агрегации
(если узел расчётный), уровнем в иерархии, и ссылкой на модель (если прогнозный).

Формулы расчётных узлов:
    NSM            = Клиенты × Общая_конверсия
    Клиенты        = Новые + Повторные
    Новые          = Привлечённые_коммуникациями + Привлечённые_банка
    Повторные      = Возвращённые + Продлённые
    Общая_конверсия = CR_первой × доля_новых + CR_повторной × доля_повторных
    CR_первой      = CTR_витрины × CR_расчёта × CR_оформления × CR_оплаты
    CR_повторной   = 1 - (1 - Коэф_повт_входа) × (1 - Коэф_кросс_продаж)

    Привлечённые_коммуникациями = Охват × CTR_коммуникаций
    Привлечённые_банка          = Клиенты_банка × Коэф_проникновения
    Возвращённые                = Ранее_активные × Коэф_возврата
    Продлённые                  = Активные × Коэф_продления

LTV — самостоятельный блок, не входит в формулу NSM напрямую (используется
для оценки ценности клиентской базы, см. Главу 2 раздел 2.4.1).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class NodeSpec:
    """
    Спецификация узла дерева метрик.

    name           — внутренний идентификатор (snake_case, на латинице)
    display_name   — читаемое название для текстов и отчётов
    node_type      — observed (входные данные), forecasted (прогнозируется
                     моделью), computed (вычисляется по формуле)
    level          — уровень в иерархии: 1 (NSM, верх) до 5 (наблюдаемые, низ)
    formula_inputs — для computed: список имён узлов, на которых строится формула
    formula        — для computed: функция (dict[str, float]) -> float
    model_name     — для forecasted: имя модели в models/artifacts (без расширения)
    description    — пояснение для документации
    """
    name: str
    display_name: str
    node_type: str             # "observed" | "forecasted" | "computed"
    level: int
    formula_inputs: list[str] = field(default_factory=list)
    formula: Optional[Callable] = None
    model_name: Optional[str] = None
    description: str = ""


# === Уровень 1: NSM ===
NSM = NodeSpec(
    name="nsm",
    display_name="NSM (оформленные полисы)",
    node_type="computed",
    level=1,
    formula_inputs=["clients_total", "overall_conversion"],
    formula=lambda v: v["clients_total"] * v["overall_conversion"],
    description="Верхнеуровневая метрика: число оформленных страховых полисов "
                "за период. Произведение клиентов воронки и общей конверсии.",
)

# === Уровень 2: структурные блоки ===
CLIENTS_TOTAL = NodeSpec(
    name="clients_total",
    display_name="Клиенты страхового канала",
    node_type="computed",
    level=2,
    formula_inputs=["new_clients", "returning_clients"],
    formula=lambda v: v["new_clients"] + v["returning_clients"],
)

OVERALL_CONVERSION = NodeSpec(
    name="overall_conversion",
    display_name="Общая конверсия",
    node_type="computed",
    level=2,
    formula_inputs=["cr_first_purchase", "cr_repeat_purchase",
                     "share_new_clients", "share_returning_clients"],
    formula=lambda v: (
        v["cr_first_purchase"] * v["share_new_clients"]
        + v["cr_repeat_purchase"] * v["share_returning_clients"]
    ),
    description="Взвешенная конверсия: CR первичных покупок взвешен долей "
                "новых клиентов, CR повторных — долей повторных.",
)

LTV = NodeSpec(
    name="ltv",
    display_name="LTV (пожизненная ценность)",
    node_type="forecasted",
    level=2,
    model_name="bgnbd_gamma",
    description="Оценивается отдельным блоком (BG/NBD + Gamma-Gamma). "
                "Не входит в формулу NSM напрямую — используется для оценки "
                "ценности клиентской базы.",
)

# === Уровень 3: декомпозиция структурных блоков ===
NEW_CLIENTS = NodeSpec(
    name="new_clients",
    display_name="Новые клиенты",
    node_type="computed",
    level=3,
    formula_inputs=["acquired_via_communications", "acquired_via_bank"],
    formula=lambda v: v["acquired_via_communications"] + v["acquired_via_bank"],
)

RETURNING_CLIENTS = NodeSpec(
    name="returning_clients",
    display_name="Повторные клиенты",
    node_type="computed",
    level=3,
    formula_inputs=["resurrected_clients", "renewed_clients"],
    formula=lambda v: v["resurrected_clients"] + v["renewed_clients"],
)

CR_FIRST_PURCHASE = NodeSpec(
    name="cr_first_purchase",
    display_name="CR первой покупки",
    node_type="computed",
    level=3,
    formula_inputs=["ctr_vitrine", "cr_calc", "cr_form", "cr_payment"],
    formula=lambda v: v["ctr_vitrine"] * v["cr_calc"] * v["cr_form"] * v["cr_payment"],
    description="Произведение четырёх последовательных конверсий воронки "
                "первичной покупки.",
)

CR_REPEAT_PURCHASE = NodeSpec(
    name="cr_repeat_purchase",
    display_name="CR повторной покупки",
    node_type="computed",
    level=3,
    formula_inputs=["coef_repeat_entry", "coef_cross_sell"],
    formula=lambda v: 1 - (1 - v["coef_repeat_entry"]) * (1 - v["coef_cross_sell"]),
    description="Вероятность хотя бы одного из двух событий: повторный вход "
                "клиента или кросс-продажа.",
)

# Доли новых и повторных — вычисляются ПОСЛЕ clients_total (уровень 2),
# поэтому формально находятся между уровнями 2 и 3. Помещаем на уровень 2,
# чтобы corrgreкт обхода снизу вверх их посчитал в нужный момент.
SHARE_NEW_CLIENTS = NodeSpec(
    name="share_new_clients",
    display_name="Доля новых клиентов",
    node_type="computed",
    level=2,
    formula_inputs=["new_clients", "clients_total"],
    formula=lambda v: v["new_clients"] / v["clients_total"] if v["clients_total"] > 0 else 0,
)

SHARE_RETURNING_CLIENTS = NodeSpec(
    name="share_returning_clients",
    display_name="Доля повторных клиентов",
    node_type="computed",
    level=2,
    formula_inputs=["returning_clients", "clients_total"],
    formula=lambda v: v["returning_clients"] / v["clients_total"] if v["clients_total"] > 0 else 0,
)

# === Уровень 4: операционные драйверы ===
ACQUIRED_VIA_COMMUNICATIONS = NodeSpec(
    name="acquired_via_communications",
    display_name="Привлечённые коммуникациями",
    node_type="computed",
    level=4,
    formula_inputs=["communications_reach", "communications_ctr"],
    formula=lambda v: v["communications_reach"] * v["communications_ctr"],
)

ACQUIRED_VIA_BANK = NodeSpec(
    name="acquired_via_bank",
    display_name="Привлечённые из базы банка",
    node_type="computed",
    level=4,
    formula_inputs=["bank_clients", "penetration_rate"],
    formula=lambda v: v["bank_clients"] * v["penetration_rate"],
)

RESURRECTED_CLIENTS = NodeSpec(
    name="resurrected_clients",
    display_name="Возвращённые клиенты",
    node_type="computed",
    level=4,
    formula_inputs=["previously_active_clients", "resurrection_rate"],
    formula=lambda v: v["previously_active_clients"] * v["resurrection_rate"],
)

RENEWED_CLIENTS = NodeSpec(
    name="renewed_clients",
    display_name="Продлённые клиенты",
    node_type="computed",
    level=4,
    formula_inputs=["active_policies", "renewal_rate"],
    formula=lambda v: v["active_policies"] * v["renewal_rate"],
)

# === Уровень 4: прогнозируемые узлы ===
CTR_VITRINE = NodeSpec(
    name="ctr_vitrine",
    display_name="CTR витрины",
    node_type="forecasted",
    level=4,
    model_name="logreg_ctr_vitrine",
    description="Доля показов витрины, приведших к клику. Шаг 1 воронки.",
)
CR_CALC = NodeSpec(
    name="cr_calc",
    display_name="CR расчёта",
    node_type="forecasted",
    level=4,
    model_name="logreg_cr_calc",
    description="Доля кликов, приведших к запуску расчёта. Шаг 2 воронки.",
)
CR_FORM = NodeSpec(
    name="cr_form",
    display_name="CR оформления",
    node_type="forecasted",
    level=4,
    model_name="logreg_cr_form",
    description="Доля расчётов, приведших к началу оформления. Шаг 3 воронки.",
)
CR_PAYMENT = NodeSpec(
    name="cr_payment",
    display_name="CR оплаты",
    node_type="forecasted",
    level=4,
    model_name="logreg_cr_payment",
    description="Доля оформлений, приведших к успешной оплате. Шаг 4 воронки.",
)
COEF_REPEAT_ENTRY = NodeSpec(
    name="coef_repeat_entry",
    display_name="Коэффициент повторного входа",
    node_type="forecasted",
    level=4,
    model_name="logreg_repeat_entry",
    description="Вероятность повторной покупки клиентом, ранее купившим "
                "страховой продукт. Прогнозируется отдельной LogReg-моделью.",
)
PENETRATION_RATE = NodeSpec(
    name="penetration_rate",
    display_name="Коэффициент проникновения",
    node_type="forecasted",
    level=4,
    model_name="xgb_penetration",
    description="Доля клиентов банка, совершивших страховую покупку. "
                "XGBoost + SHAP.",
)
RENEWAL_RATE = NodeSpec(
    name="renewal_rate",
    display_name="Коэффициент продления",
    node_type="forecasted",
    level=4,
    model_name="cox_renewal",
    description="Вероятность продления полиса. Cox Proportional Hazards.",
)

# === Уровень 5: наблюдаемые входные данные ===
COMMUNICATIONS_REACH = NodeSpec(
    name="communications_reach",
    display_name="Охват коммуникаций",
    node_type="observed", level=5,
)
COMMUNICATIONS_CTR = NodeSpec(
    name="communications_ctr",
    display_name="CTR коммуникаций",
    node_type="observed", level=5,
)
BANK_CLIENTS = NodeSpec(
    name="bank_clients",
    display_name="Клиенты банка",
    node_type="observed", level=5,
)
PREVIOUSLY_ACTIVE_CLIENTS = NodeSpec(
    name="previously_active_clients",
    display_name="Ранее активные клиенты",
    node_type="observed", level=5,
)
ACTIVE_POLICIES = NodeSpec(
    name="active_policies",
    display_name="Активные полисы",
    node_type="observed", level=5,
)
RESURRECTION_RATE = NodeSpec(
    name="resurrection_rate",
    display_name="Коэффициент возврата",
    node_type="observed", level=5,
)
COEF_CROSS_SELL = NodeSpec(
    name="coef_cross_sell",
    display_name="Коэффициент кросс-продаж",
    node_type="observed", level=5,
    description="В дереве v8 это наблюдаемая величина, не прогнозируется. "
                "Берётся как историческое среднее по портфелю.",
)


# === Сборка дерева ===
ALL_NODES: list[NodeSpec] = [
    # Уровень 1
    NSM,
    # Уровень 2
    CLIENTS_TOTAL, OVERALL_CONVERSION, LTV,
    # Уровень 3
    NEW_CLIENTS, RETURNING_CLIENTS,
    CR_FIRST_PURCHASE, CR_REPEAT_PURCHASE,
    SHARE_NEW_CLIENTS, SHARE_RETURNING_CLIENTS,
    # Уровень 4 — расчётные
    ACQUIRED_VIA_COMMUNICATIONS, ACQUIRED_VIA_BANK,
    RESURRECTED_CLIENTS, RENEWED_CLIENTS,
    # Уровень 4 — прогнозируемые
    CTR_VITRINE, CR_CALC, CR_FORM, CR_PAYMENT,
    COEF_REPEAT_ENTRY, PENETRATION_RATE, RENEWAL_RATE,
    # Уровень 5 — наблюдаемые
    COMMUNICATIONS_REACH, COMMUNICATIONS_CTR, BANK_CLIENTS,
    PREVIOUSLY_ACTIVE_CLIENTS, ACTIVE_POLICIES,
    RESURRECTION_RATE, COEF_CROSS_SELL,
]

NODES_BY_NAME: dict[str, NodeSpec] = {n.name: n for n in ALL_NODES}


def get_node(name: str) -> NodeSpec:
    """Достаёт узел по имени или кидает понятную ошибку."""
    if name not in NODES_BY_NAME:
        raise KeyError(
            f"Узел '{name}' не найден в дереве. Доступные: "
            f"{sorted(NODES_BY_NAME.keys())}"
        )
    return NODES_BY_NAME[name]


def nodes_by_type(node_type: str) -> list[NodeSpec]:
    """Возвращает все узлы заданного типа."""
    return [n for n in ALL_NODES if n.node_type == node_type]


def tree_summary() -> dict:
    """Сводка структуры дерева — для отчёта и проверки."""
    return {
        "total_nodes": len(ALL_NODES),
        "by_type": {
            "observed":   len(nodes_by_type("observed")),
            "forecasted": len(nodes_by_type("forecasted")),
            "computed":   len(nodes_by_type("computed")),
        },
        "by_level": {
            level: sum(1 for n in ALL_NODES if n.level == level)
            for level in range(1, 6)
        },
        "forecasted_models": [n.model_name for n in nodes_by_type("forecasted")],
    }


if __name__ == "__main__":
    import json
    summary = tree_summary()
    print("=== Структура дерева ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Все узлы по уровням ===")
    for level in range(1, 6):
        print(f"\nУровень {level}:")
        for node in [n for n in ALL_NODES if n.level == level]:
            type_marker = {"observed": "obs", "forecasted": "FCT", "computed": "calc"}[node.node_type]
            print(f"  [{type_marker:>4s}] {node.name:<30s} — {node.display_name}")
