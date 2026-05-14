"""
Движок генерации рекомендаций.

Принцип работы:
    1. Сравнивает текущие/прогнозные значения узлов дерева с baseline.
    2. Для узлов с отклонением > порог — срабатывает правило из библиотеки.
    3. Каждое правило формулирует конкретное управленческое действие
       в терминах бизнес-операций, не математических.

Архитектурная заметка:
    База правил — словарь "имя_узла → правило". Это правильно для прозрачности
    в управленческом инструменте: каждая рекомендация имеет один явный источник,
    можно показать комиссии/менеджеру конкретное правило, отвечающее за вывод.

    Альтернатива — обучить ML-классификатор «отклонение → рекомендация» — была
    бы менее интерпретируема и излишне сложна для текущей задачи (управленческие
    действия в страховой воронке хорошо известны заранее).

Формат вывода:
    Каждая рекомендация — словарь с полями:
        node           — имя узла
        node_display   — читаемое название узла
        direction      — "снизился" / "вырос"
        delta_pct      — отклонение от baseline в %
        priority       — high / medium / low (по силе отклонения)
        action         — управленческая рекомендация
        rationale      — обоснование (почему это поможет)
        expected_uplift — ожидаемый прирост метрики при выполнении (если оценим)
"""
from __future__ import annotations
import json
from typing import Optional

from config import MODELS_ARTIFACTS_DIR
from tree.metric_tree import get_node


# === Библиотека правил ===
# Каждое правило содержит:
#   - threshold_pct: минимальное отклонение для срабатывания (в %)
#   - action: что делать
#   - rationale: почему это сработает
#   - expected_uplift_pct: ожидаемый прирост узла после действия (нижняя оценка)
#
# Действия и оценки uplift подобраны по бенчмаркам отрасли
# (UX-аудит формы → +5-15% к CR; ремаркетинг → +10-20% к повторным;
# упрощение оплаты → +3-8% к CR оплаты).
RECOMMENDATION_RULES = {
    "ctr_vitrine": {
        "threshold_pct": 5,
        "action_down": (
            "Пересмотреть размещение страховых офферов на витрине: "
            "повысить позицию (top-1 / top-2), включить промо-баннер, "
            "обновить креативы под высокосезонные продукты."
        ),
        "rationale_down": (
            "CTR витрины — ранжированный сигнал: позиция и визуальная "
            "выраженность увеличивают вероятность клика на 30-50% по "
            "данным A/B-тестов в банковских приложениях."
        ),
        "expected_uplift_pct": 8,
    },
    "cr_calc": {
        "threshold_pct": 5,
        "action_down": (
            "Упростить шаг расчёта стоимости: предзаполнить параметры "
            "из профиля клиента, сократить число обязательных полей, "
            "добавить дефолтные значения для типовых ситуаций."
        ),
        "rationale_down": (
            "Снижение когнитивной нагрузки на этапе расчёта повышает "
            "конверсию на 5-12% по результатам UX-исследований "
            "финансовых сервисов."
        ),
        "expected_uplift_pct": 7,
    },
    "cr_form": {
        "threshold_pct": 5,
        "action_down": (
            "Провести UX-аудит формы оформления: сократить количество "
            "полей до минимально необходимого, добавить прогресс-бар, "
            "разбить форму на этапы (пошаговое заполнение)."
        ),
        "rationale_down": (
            "Длинные формы — главная причина отказов на этапе оформления. "
            "Сокращение количества полей с 12 до 6-8 типично даёт "
            "+10-15% к CR оформления."
        ),
        "expected_uplift_pct": 10,
    },
    "cr_payment": {
        "threshold_pct": 5,
        "action_down": (
            "Проверить конфигурацию платёжного шлюза: время отклика, "
            "процент технических ошибок. Добавить альтернативные методы "
            "оплаты (СБП, Apple Pay, Google Pay)."
        ),
        "rationale_down": (
            "Технические сбои на оплате и недостаточное число способов "
            "оплаты — частая причина падения CR оплаты на 3-8 п.п. "
            "Подключение СБП в банковских приложениях даёт +5-10% "
            "к завершённым покупкам."
        ),
        "expected_uplift_pct": 6,
    },
    "renewal_rate": {
        "threshold_pct": 3,
        "action_down": (
            "Активировать кампанию напоминания о продлении: настроить "
            "коммуникации за 30/14/3 дня до окончания полиса, через "
            "пуш + email + SMS. Для премиальных сегментов — звонок "
            "от менеджера."
        ),
        "rationale_down": (
            "По результатам обученной Cox PH-модели, hazard ratio контактов "
            "в окне продления составляет 1.45 за каждый дополнительный "
            "контакт. Регулярное напоминание увеличивает вероятность "
            "продления на 8-15%."
        ),
        "expected_uplift_pct": 10,
    },
    "penetration_rate": {
        "threshold_pct": 5,
        "action_down": (
            "Расширить таргетинг на сегменты с высоким SHAP-значением: "
            "клиенты со стажем 24+ месяца, активные в мобильном "
            "приложении, посещавшие раздел страхования за последние "
            "90 дней. Запустить персонализированные офферы для этих "
            "сегментов."
        ),
        "rationale_down": (
            "Согласно SHAP-анализу обученной XGBoost-модели, главные "
            "драйверы проникновения — стаж клиента, цифровая активность "
            "и интерес к разделу страхования. Точечный таргетинг этих "
            "сегментов повышает конверсию проникновения в 2-3 раза "
            "относительно широкого охвата."
        ),
        "expected_uplift_pct": 15,
    },
    "coef_repeat_entry": {
        "threshold_pct": 5,
        "action_down": (
            "Настроить триггерную коммуникацию для ранее активных "
            "клиентов: при появлении в приложении показывать релевантный "
            "оффер, отправлять напоминания о возможности кросс-продажи."
        ),
        "rationale_down": (
            "Реактивация бывших покупателей — самая дешёвая форма "
            "привлечения: они уже доверяют бренду. Триггерные сценарии "
            "дают повторный вход на 12-20% выше базового уровня."
        ),
        "expected_uplift_pct": 12,
    },
}


def generate_recommendations(baseline_values: dict[str, float],
                              current_values: dict[str, float],
                              custom_thresholds: Optional[dict[str, float]] = None
                              ) -> list[dict]:
    """
    Генерирует список рекомендаций на основе сравнения baseline и текущих
    значений по узлам, у которых есть правила.

    Параметр custom_thresholds позволяет переопределить порог срабатывания
    для конкретного узла — например, для приоритетных метрик ставить
    более чувствительные пороги.
    """
    recs = []
    custom_thresholds = custom_thresholds or {}

    for node_name, rule in RECOMMENDATION_RULES.items():
        if node_name not in baseline_values or node_name not in current_values:
            continue

        baseline = baseline_values[node_name]
        current = current_values[node_name]

        if baseline == 0:
            continue

        delta_pct = (current - baseline) / baseline * 100
        threshold = custom_thresholds.get(node_name, rule["threshold_pct"])

        # Срабатывание только при снижении ниже порога — рекомендации направлены
        # на исправление проседания. При росте отдельная логика не нужна
        # (это уже хорошо).
        if delta_pct >= -threshold:
            continue

        # Приоритет по силе отклонения
        abs_delta = abs(delta_pct)
        if abs_delta >= 15:
            priority = "high"
        elif abs_delta >= 8:
            priority = "medium"
        else:
            priority = "low"

        node_spec = get_node(node_name)
        recs.append({
            "node":             node_name,
            "node_display":     node_spec.display_name,
            "direction":        "снизился",
            "current_value":    round(current, 4),
            "baseline_value":   round(baseline, 4),
            "delta_pct":        round(delta_pct, 2),
            "priority":         priority,
            "action":           rule["action_down"],
            "rationale":        rule["rationale_down"],
            "expected_uplift_pct": rule["expected_uplift_pct"],
        })

    # Сортируем по приоритету и силе отклонения (худшие — сверху)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: (priority_order[r["priority"]], r["delta_pct"]))

    return recs


def format_recommendations_text(recs: list[dict]) -> str:
    """Форматирует список рекомендаций как читаемый текст для менеджера."""
    if not recs:
        return "Все ключевые метрики в норме. Рекомендаций не сгенерировано."

    lines = [f"Сгенерировано рекомендаций: {len(recs)}\n"]
    priority_labels = {
        "high":   "[ВЫСОКИЙ ПРИОРИТЕТ]",
        "medium": "[СРЕДНИЙ ПРИОРИТЕТ]",
        "low":    "[НИЗКИЙ ПРИОРИТЕТ]",
    }
    for i, rec in enumerate(recs, 1):
        lines.append(f"--- Рекомендация {i} ---")
        lines.append(f"{priority_labels[rec['priority']]} {rec['node_display']}")
        lines.append(f"Отклонение: {rec['delta_pct']:+.1f}% от baseline "
                     f"(текущее = {rec['current_value']}, baseline = {rec['baseline_value']})")
        lines.append(f"\nДействие: {rec['action']}")
        lines.append(f"\nОбоснование: {rec['rationale']}")
        lines.append(f"\nОжидаемый прирост узла после внедрения: "
                     f"+{rec['expected_uplift_pct']}%")
        lines.append("")
    return "\n".join(lines)


def demo() -> dict:
    """
    Демонстрационный сценарий: берём baseline из tree_baseline.json,
    моделируем "проседание" нескольких узлов, генерируем рекомендации.

    Это и есть управленческий сценарий применения инструмента:
        - менеджер видит, что в этом периоде CR оформления упал на 12%,
          коэффициент продления — на 5%, остальное в норме;
        - инструмент сразу возвращает 2 рекомендации с действиями.
    """
    baseline_path = MODELS_ARTIFACTS_DIR / "tree_baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Сначала запусти tree/aggregator.py — нужен файл {baseline_path}"
        )

    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    baseline_tree = data["tree_values"]

    # Имитируем проседание трёх метрик
    current_tree = dict(baseline_tree)
    current_tree["cr_form"] = baseline_tree["cr_form"] * 0.88           # −12%
    current_tree["renewal_rate"] = baseline_tree["renewal_rate"] * 0.95  # −5%
    current_tree["penetration_rate"] = baseline_tree["penetration_rate"] * 0.93  # −7%

    print("=== Изменения наблюдаемые в периоде ===")
    for node in ["cr_form", "renewal_rate", "penetration_rate"]:
        b, c = baseline_tree[node], current_tree[node]
        print(f"  {node:<25s}: {b:.4f} → {c:.4f}  ({(c-b)/b*100:+.1f}%)")

    recs = generate_recommendations(baseline_tree, current_tree)

    print("\n" + "=" * 60)
    print("=== РЕКОМЕНДАЦИИ ===")
    print("=" * 60)
    print(format_recommendations_text(recs))

    # Сохраняем
    out_path = MODELS_ARTIFACTS_DIR / "recommendations_demo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False)
    print(f"Демо-рекомендации сохранены: {out_path}")

    return recs


if __name__ == "__main__":
    demo()
