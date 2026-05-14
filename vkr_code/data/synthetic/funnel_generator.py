"""
Генератор сессий и событий воронки для обучения LogReg.

Версия 2: усилена, чтобы воспроизводить более реалистичный набор сессионных
признаков и более сильные сигналы. Базовая версия давала AUC 0.55 на шагах
2-4 — слишком слабо, потому что в синтетике было 2-3 значимых признака на шаг.
Реальные A/B-логи онлайн-страхования содержат десятки признаков с коэффициентами
0.5-1.0 после стандартизации; здесь воспроизводится правдоподобный их набор.

Логика генерации построена так, чтобы вероятность перехода на следующий шаг
ЗАВИСЕЛА от признаков клиента и сессии — иначе модель не сможет их выучить и
AUC получится 0.5 (бессмысленный результат). Это ключевое свойство хорошей
синтетики: внутри неё должна быть выучиваемая структура.

Структура события:
  impression -> click -> calc_complete -> form_submit -> payment_success
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from config import (
    N_CLIENTS, RANDOM_SEED, SYNTHETIC_DIR, CUTOFF_DATE,
    WINDOWS_DAYS, FUNNEL_BASE_RATES, SESSIONS_PER_CLIENT_MEAN,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Сигмоида с защитой от переполнения. Используем для перевода линейной
    комбинации признаков в вероятность Bernoulli."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _logit(p: float) -> float:
    """Обратная сигмоиде. Через неё задаётся свободный член линейной комбинации,
    чтобы базовая вероятность шага была равна заданной FUNNEL_BASE_RATES[...]."""
    return np.log(p / (1 - p))


def _generate_session_features(n: int, rng: np.random.Generator) -> dict:
    """
    Генерирует сессионные признаки, которые НЕ зависят от клиента,
    но влияют на конверсию.

    Эти поля моделируют то, что в реальных данных лежит в логах продукта:
    A/B-варианты, размер скидки, число полей формы, время на странице.
    """
    # === A/B-вариант оффера ===
    # 3 варианта: control, v_discount (со скидкой), v_simplified (упрощённая форма).
    offer_version = rng.choice(
        ["control", "v_discount", "v_simplified"],
        size=n, p=[0.50, 0.25, 0.25],
    )

    # === Размер скидки ===
    # Только для оффера v_discount, остальные = 0.
    discount_pct = np.where(
        offer_version == "v_discount",
        rng.choice([5, 10, 15, 20], size=n, p=[0.40, 0.30, 0.20, 0.10]),
        0,
    )

    # === Число обязательных полей в форме ===
    # Зависит от типа оффера: упрощённая форма имеет меньше полей.
    n_form_fields = np.where(
        offer_version == "v_simplified",
        rng.integers(5, 9, size=n),       # 5-8 полей
        rng.integers(10, 15, size=n),     # 10-14 полей
    )

    # === Время на странице расчёта (секунды) ===
    # Лог-нормальное распределение, медиана около 90 секунд.
    time_on_calc_sec = rng.lognormal(mean=4.5, sigma=0.7, size=n).clip(5, 600).astype(int)

    # === Стоимость продукта в показанной офертe ===
    # Лог-нормальная вокруг 5000 руб. Дороже — ниже конверсия.
    offer_price_rub = rng.lognormal(mean=8.5, sigma=0.6, size=n).clip(500, 50000).astype(int)

    # === Количество просмотренных карточек продуктов до выбора ===
    # Чем больше клиент сравнивает, тем менее вероятна покупка (паралич выбора).
    products_viewed = rng.poisson(lam=2.0, size=n).clip(1, 10)

    # === Был ли промо-баннер на витрине ===
    has_promo_banner = rng.choice([0, 1], size=n, p=[0.65, 0.35])

    return {
        "offer_version": offer_version,
        "discount_pct": discount_pct,
        "n_form_fields": n_form_fields,
        "time_on_calc_sec": time_on_calc_sec,
        "offer_price_rub": offer_price_rub,
        "products_viewed": products_viewed,
        "has_promo_banner": has_promo_banner,
    }


def generate_funnel_events(clients_df: pd.DataFrame,
                            seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Генерирует таблицу событий воронки на основе клиентского профиля.

    На каждого клиента — переменное число сессий (Пуассон).
    В каждой сессии — последовательность из максимум 5 событий.
    Сессия обрывается на том шаге, где событие не произошло.
    """
    rng = np.random.default_rng(seed + 1)

    cutoff = datetime.fromisoformat(CUTOFF_DATE)
    window_start = cutoff - timedelta(days=WINDOWS_DAYS["logreg_funnel"])

    # === Сколько сессий у каждого клиента ===
    n_sessions_per_client = rng.poisson(lam=SESSIONS_PER_CLIENT_MEAN, size=len(clients_df))

    client_idx_per_session = np.repeat(np.arange(len(clients_df)), n_sessions_per_client)
    n_total_sessions = len(client_idx_per_session)

    if n_total_sessions == 0:
        return pd.DataFrame()

    # Достаём атрибуты клиентов в "длинном" виде по индексу сессии.
    sessions = pd.DataFrame({
        "client_id": clients_df["client_id"].values[client_idx_per_session],
        "age_years": clients_df["age_years"].values[client_idx_per_session],
        "bank_segment": clients_df["bank_segment"].values[client_idx_per_session],
        "products_count_bank": clients_df["products_count_bank"].values[client_idx_per_session],
        "has_credit_card": clients_df["has_credit_card"].values[client_idx_per_session],
        "has_deposit": clients_df["has_deposit"].values[client_idx_per_session],
        "app_sessions_30d": clients_df["app_sessions_30d"].values[client_idx_per_session],
        "insurance_section_visits_90d": clients_df["insurance_section_visits_90d"].values[client_idx_per_session],
        "ever_bought_insurance": clients_df["ever_bought_insurance"].values[client_idx_per_session],
    })

    sessions["session_id"] = [f"S{i:09d}" for i in range(n_total_sessions)]

    # === Время сессии ===
    timestamps = window_start + pd.to_timedelta(
        rng.integers(0, WINDOWS_DAYS["logreg_funnel"] * 86400, size=n_total_sessions),
        unit="s",
    )
    sessions["session_timestamp"] = timestamps
    sessions["day_of_week"] = sessions["session_timestamp"].dt.dayofweek
    sessions["hour_of_day"] = sessions["session_timestamp"].dt.hour
    # Выходной vs будний — отдельный признак, потому что выходные сильно меняют
    # паттерн поведения (свободное время → более вдумчивые покупки).
    sessions["is_weekend"] = (sessions["day_of_week"] >= 5).astype(int)

    # Тип устройства: 65% мобильные, 35% веб.
    sessions["device_type"] = rng.choice(["mobile", "web"], size=n_total_sessions, p=[0.65, 0.35])

    # Позиция витрины (1 — топ, 5 — дно). Влияет на CTR.
    sessions["shelf_position"] = rng.integers(1, 6, size=n_total_sessions)

    # === Сессионные признаки ===
    sf = _generate_session_features(n_total_sessions, rng)
    for col, arr in sf.items():
        sessions[col] = arr

    # === Стандартизованные версии признаков для линейных предикторов ===
    # Делается один раз тут, чтобы не повторять по 4 раза ниже.
    age_z = (sessions["age_years"].values - 38) / 12
    apps_z = (sessions["app_sessions_30d"].values - 12) / 8
    visits_z = (sessions["insurance_section_visits_90d"].values - 1) / 2
    discount_z = sf["discount_pct"] / 10              # шкала 0-2 (макс скидка 20%)
    fields_z = (sf["n_form_fields"] - 10) / 3
    time_calc_z = (np.log(sf["time_on_calc_sec"]) - 4.5) / 0.7
    price_z = (np.log(sf["offer_price_rub"]) - 8.5) / 0.6
    products_viewed_z = (sf["products_viewed"] - 2) / 1.5

    is_mobile = (sessions["device_type"] == "mobile").astype(int).values
    is_premium = sessions["bank_segment"].isin(["premium", "private"]).astype(int).values

    # === Шаг 1: impression -> click ===
    # Главные драйверы: позиция витрины, история страхования, промо-баннер,
    # активность в разделе. Позиция витрины — самый сильный сигнал.
    eta_ctr = (
        _logit(FUNNEL_BASE_RATES["ctr_vitrine"])
        - 0.55 * (sessions["shelf_position"].values - 1)
        + 0.60 * sessions["ever_bought_insurance"].values
        + 0.45 * sf["has_promo_banner"]
        + 0.35 * visits_z
        + 0.25 * apps_z
        + 0.20 * sessions["is_weekend"].values
        + 0.30 * (sf["offer_version"] == "v_discount").astype(float)
    )
    p_ctr = _sigmoid(eta_ctr)
    has_click = (rng.random(n_total_sessions) < p_ctr).astype(int)

    # === Шаг 2: click -> calc_complete ===
    # Главные драйверы: устройство, размер скидки, цена оффера, число полей.
    eta_calc = (
        _logit(FUNNEL_BASE_RATES["cr_calc"])
        + 0.50 * is_mobile
        + 0.45 * discount_z
        - 0.40 * price_z
        - 0.30 * age_z
        + 0.35 * (sf["offer_version"] == "v_simplified").astype(float)
        + 0.25 * sessions["has_credit_card"].values
    )
    p_calc = _sigmoid(eta_calc)
    has_calc = ((rng.random(n_total_sessions) < p_calc) & (has_click == 1)).astype(int)

    # === Шаг 3: calc_complete -> form_submit ===
    # На оформлении главное — UX формы (число полей) и платёжная готовность
    # (карта/депозит).
    eta_form = (
        _logit(FUNNEL_BASE_RATES["cr_form"])
        - 0.55 * fields_z                                # форма — главный убийца CR
        + 0.60 * sessions["has_credit_card"].values
        + 0.40 * is_premium
        + 0.35 * (sf["offer_version"] == "v_simplified").astype(float)
        + 0.25 * sessions["has_deposit"].values
        - 0.30 * age_z
        - 0.25 * products_viewed_z                       # больше сравнений → меньше покупок
        + 0.20 * time_calc_z                             # дольше изучал → серьёзнее настроен
    )
    p_form = _sigmoid(eta_form)
    has_form = ((rng.random(n_total_sessions) < p_form) & (has_calc == 1)).astype(int)

    # === Шаг 4: form_submit -> payment_success ===
    # Платёжный шаг: тут отваливаются те, у кого технические проблемы с оплатой
    # или сомнения в последний момент. Цена оффера снова важна.
    eta_pay = (
        _logit(FUNNEL_BASE_RATES["cr_payment"])
        + 0.55 * is_mobile
        + 0.45 * sessions["has_credit_card"].values
        - 0.40 * price_z
        - 0.35 * age_z
        + 0.30 * sessions["ever_bought_insurance"].values
        + 0.25 * discount_z
    )
    p_pay = _sigmoid(eta_pay)
    has_payment = ((rng.random(n_total_sessions) < p_pay) & (has_form == 1)).astype(int)

    # === Складываем все таргеты в одну таблицу ===
    sessions["had_impression"] = 1
    sessions["had_click"] = has_click
    sessions["had_calc"] = has_calc
    sessions["had_form"] = has_form
    sessions["had_payment"] = has_payment

    return sessions


if __name__ == "__main__":
    clients_path = SYNTHETIC_DIR / "clients.parquet"
    if not clients_path.exists():
        raise FileNotFoundError(
            f"Сначала запусти clients_generator.py — нужен файл {clients_path}"
        )

    clients_df = pd.read_parquet(clients_path)
    print(f"Загружено клиентов: {len(clients_df):,}")

    sessions = generate_funnel_events(clients_df)
    out_path = SYNTHETIC_DIR / "funnel_sessions.parquet"
    sessions.to_parquet(out_path, index=False)

    print(f"Сгенерировано сессий: {len(sessions):,}")
    print(f"Сохранено: {out_path}")
    print(f"Размер на диске: {out_path.stat().st_size / 1024**2:.1f} MB")

    # === Сводная статистика по воронке ===
    n = len(sessions)
    n_imp = int(sessions["had_impression"].sum())
    n_clk = int(sessions["had_click"].sum())
    n_clc = int(sessions["had_calc"].sum())
    n_frm = int(sessions["had_form"].sum())
    n_pay = int(sessions["had_payment"].sum())

    print("\n=== Воронка (абсолют / конверсия от предыдущего) ===")
    print(f"Impressions:  {n_imp:>8,}  ({n_imp/n_imp*100:.1f}%)")
    print(f"Clicks:       {n_clk:>8,}  ({n_clk/n_imp*100:.1f}%  от impression)")
    print(f"Calc:         {n_clc:>8,}  ({n_clc/n_clk*100:.1f}%  от click)")
    print(f"Form:         {n_frm:>8,}  ({n_frm/n_clc*100:.1f}%  от calc)")
    print(f"Payment:      {n_pay:>8,}  ({n_pay/n_frm*100:.1f}%  от form)")
    print(f"\nE2E конверсия: {n_pay/n_imp*100:.2f}% (impression -> payment)")
