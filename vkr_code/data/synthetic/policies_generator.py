"""
Генератор полисов с историей продлений для Cox PH.

Назначение: построить таблицу полисов с длительностью наблюдения и фактом
продления (для непродлённых — цензурирование). Это базовая структура данных
для survival analysis.

Схема выходной таблицы (policies.parquet):
    policy_id            — идентификатор полиса
    client_id            — связь с клиентом
    product_code         — тип продукта (NSZH/ISZH/PROPERTY)
    policy_start_date    — дата оформления
    policy_end_date      — плановая дата окончания (start + 365 дней)
    premium_amount       — премия по полису
    duration_days        — для Cox: дни до события или цензурирования
    event_observed       — 1 если продление произошло, 0 если цензурировано
    contacts_30d_before_end — число коммуникаций за 30 дней до окончания
    contacts_60d_before_end — то же за 60 дней
    contacts_90d_before_end — то же за 90 дней

Источник данных: уже сгенерированный `purchases.parquet`. Берём оттуда первичные
покупки как полисы, и для каждого определяем — было ли последующее продление
(покупка того же продукта тем же клиентом в окне [end - 30, end + 60]).

Цензурирование (важно для Cox):
    - Если policy_end_date > cutoff: клиент ещё не дошёл до даты окончания,
      продление не наблюдалось → event_observed=0,
      duration = (cutoff - policy_start_date)
    - Если в [end-30, end+60] не было новой покупки того же продукта:
      продления не было → event_observed=0,
      duration = (end+60 - policy_start_date) [конец окна продления]
    - Если в [end-30, end+60] была покупка того же продукта:
      продление состоялось → event_observed=1,
      duration = (renewal_date - policy_start_date)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import RANDOM_SEED, SYNTHETIC_DIR, CUTOFF_DATE


# === Срок действия полиса ===
# В розничном страховании стандартно 12 месяцев.
POLICY_DURATION_DAYS = 365

# === Окно продления ===
# За 30 дней до окончания страховая обычно начинает напоминать.
# 60 дней после окончания — грейс-период, в течение которого продление
# ещё засчитывается (см. Data_Specifications, методологический вопрос 1).
RENEWAL_WINDOW_BEFORE_END_DAYS = 30
RENEWAL_WINDOW_AFTER_END_DAYS = 60


def generate_policies(purchases_df: pd.DataFrame,
                       clients_df: pd.DataFrame,
                       seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Превращает таблицу покупок в таблицу полисов с метками продления.

    Логика:
        1. Берём только первичные покупки (is_renewal=False) — это начала полисов.
        2. Для каждой строим policy_end_date = start + POLICY_DURATION_DAYS.
        3. Ищем продление: повторная покупка того же продукта тем же клиентом
           в окне [end - 30 дней, end + 60 дней].
        4. Считаем duration и event_observed по правилам цензурирования.
        5. Прибавляем число контактов в окнах 30/60/90 дней до end.
    """
    rng = np.random.default_rng(seed + 4)
    cutoff = pd.to_datetime(CUTOFF_DATE)

    # === Шаг 1: первичные покупки = начала полисов ===
    purchases_df["purchase_date"] = pd.to_datetime(purchases_df["purchase_date"])
    primary = purchases_df[~purchases_df["is_renewal"]].copy()
    primary = primary.rename(columns={
        "purchase_date": "policy_start_date",
        "purchase_amount": "premium_amount",
    })
    primary["policy_end_date"] = primary["policy_start_date"] + pd.Timedelta(
        days=POLICY_DURATION_DAYS
    )

    # === Шаг 2: ищем продления ===
    # Продление = более поздняя покупка того же продукта тем же клиентом
    # в окне [end - 30, end + 60].
    # Используем merge_asof или явное соединение. Для простоты и читаемости
    # делаем явное self-join, потом фильтруем.
    all_purchases = purchases_df[["client_id", "purchase_date", "product_code"]].rename(
        columns={"purchase_date": "renewal_candidate_date"}
    )

    candidates = primary.merge(
        all_purchases,
        on=["client_id", "product_code"],
        how="left",
    )

    # Кандидат — это более поздняя покупка в окне продления
    in_renewal_window = (
        (candidates["renewal_candidate_date"] >= candidates["policy_end_date"]
         - pd.Timedelta(days=RENEWAL_WINDOW_BEFORE_END_DAYS)) &
        (candidates["renewal_candidate_date"] <= candidates["policy_end_date"]
         + pd.Timedelta(days=RENEWAL_WINDOW_AFTER_END_DAYS))
    )
    candidates_in_window = candidates[in_renewal_window].copy()

    # Берём первое продление в окне (минимальную дату). Если их несколько —
    # засчитывается ближайшее по времени продление.
    renewal_first = (
        candidates_in_window
        .sort_values(["policy_id", "renewal_candidate_date"])
        .drop_duplicates(subset="policy_id", keep="first")
        [["policy_id", "renewal_candidate_date"]]
        .rename(columns={"renewal_candidate_date": "renewal_date"})
    )

    policies = primary.merge(renewal_first, on="policy_id", how="left")

    # === Шаг 3: правила цензурирования и duration ===
    # Три случая:
    #   А. policy_end_date > cutoff → срок ещё не вышел, ждём продления →
    #      duration = cutoff - start, event=0
    #   Б. renewal_date присутствует → продление состоялось →
    #      duration = renewal_date - start, event=1
    #   В. policy_end_date <= cutoff и renewal_date отсутствует, но окно
    #      продления уже закрыто (end + 60 <= cutoff) → продления не будет →
    #      duration = (end + 60) - start = POLICY_DURATION_DAYS + 60, event=0
    #   Г. policy_end_date <= cutoff, окно продления ещё не закрыто
    #      (end < cutoff < end + 60) → пока цензурируем →
    #      duration = cutoff - start, event=0

    policy_end_after_cutoff = policies["policy_end_date"] > cutoff
    has_renewal = policies["renewal_date"].notna()
    renewal_window_closed = (
        policies["policy_end_date"]
        + pd.Timedelta(days=RENEWAL_WINDOW_AFTER_END_DAYS) <= cutoff
    )

    # event_observed
    policies["event_observed"] = has_renewal.astype(int)

    # duration_days по веткам выше
    duration = pd.Series(0, index=policies.index, dtype=float)
    # Б. Продление произошло
    duration[has_renewal] = (
        policies.loc[has_renewal, "renewal_date"]
        - policies.loc[has_renewal, "policy_start_date"]
    ).dt.days
    # В. Окно закрыто, продления нет — событие "непродление"
    mask_no_renewal_closed = (~has_renewal) & renewal_window_closed
    duration[mask_no_renewal_closed] = (
        policies.loc[mask_no_renewal_closed, "policy_end_date"]
        + pd.Timedelta(days=RENEWAL_WINDOW_AFTER_END_DAYS)
        - policies.loc[mask_no_renewal_closed, "policy_start_date"]
    ).dt.days
    # А и Г. Цензурирование на cutoff
    mask_censored = (~has_renewal) & (~renewal_window_closed)
    duration[mask_censored] = (
        cutoff - policies.loc[mask_censored, "policy_start_date"]
    ).dt.days

    policies["duration_days"] = duration.astype(int)

    # Полисы с duration <= 0 не должны попадать в обучение Cox — это
    # либо ошибка дат, либо очень свежие полисы. Отфильтровываем.
    policies = policies[policies["duration_days"] > 0].copy()

    # === Шаг 4: симулируем число контактов до окончания ===
    # В реальных данных эти числа берутся из CRM-таблицы communications.
    # Для синтетики моделируем: чем активнее клиент, тем больше контактов от банка
    # (триггерные кампании отзываются на цифровое поведение).
    # Контакты влияют на продление позитивно — это центральный сигнал для бизнеса.
    n_policies = len(policies)

    # Базовое число контактов в окне 90 дней — Пуассон со средним 2.5
    # (типично для retention-кампании страхового продукта).
    contacts_90d = rng.poisson(lam=2.5, size=n_policies)
    # 60 дней — поджатое подмножество 90-дневного окна
    contacts_60d = rng.binomial(contacts_90d, p=0.65)
    # 30 дней — поджатое подмножество 60-дневного окна
    contacts_30d = rng.binomial(contacts_60d, p=0.55)

    policies["contacts_30d_before_end"] = contacts_30d
    policies["contacts_60d_before_end"] = contacts_60d
    policies["contacts_90d_before_end"] = contacts_90d

    # === Шаг 5: добавляем клиентские признаки для Cox-модели ===
    # Берём демографию и сегмент. Цифровая активность не используется здесь —
    # на горизонте полиса (год) она менее предиктивна, чем для проникновения.
    client_features = clients_df[[
        "client_id", "age_years", "gender", "bank_segment",
        "tenure_bank_months", "products_count_bank",
    ]]
    policies = policies.merge(client_features, on="client_id", how="left")

    # === Шаг 6: усиленный сигнал контактов на продление ===
    # Реалистично: контакты помогают, но не на 100%. Добавим контактам
    # реальное влияние через ретроспективную модификацию event_observed.
    # Принцип: для части неактивных полисов (event=0) с высоким числом контактов
    # переключаем event=1, чтобы на тренировке был сигнал "контакты помогают".
    #
    # Это не "честная" симуляция, а калибровка — иначе на синтетике
    # коэффициент при контактах получается нулевым (контакты случайны).
    # Для целей синтетики приемлемо — мы хотим, чтобы Cox мог выучить
    # положительный hazard ratio при контактах.
    closed_policies = ~mask_censored.reindex(policies.index, fill_value=False)
    no_renewal_yet = (policies["event_observed"] == 0) & closed_policies

    # Вероятность ретро-добавить продление зависит от числа контактов:
    # 0 контактов — 0.05, 5+ — 0.45.
    p_retro_renewal = np.clip(
        0.05 + 0.08 * policies["contacts_60d_before_end"],
        0.05, 0.50,
    )
    flip_to_renewal = no_renewal_yet & (
        rng.random(len(policies)) < p_retro_renewal
    )
    policies.loc[flip_to_renewal, "event_observed"] = 1
    # duration таких полисов оставляем как есть — мы их моделируем как
    # продлённых "в последний момент" в окне продления.

    # === Финальный набор колонок ===
    final_cols = [
        "policy_id", "client_id", "product_code",
        "policy_start_date", "policy_end_date", "premium_amount",
        "duration_days", "event_observed",
        "contacts_30d_before_end", "contacts_60d_before_end",
        "contacts_90d_before_end",
        "age_years", "gender", "bank_segment",
        "tenure_bank_months", "products_count_bank",
    ]
    return policies[final_cols].reset_index(drop=True)


if __name__ == "__main__":
    purchases_path = SYNTHETIC_DIR / "purchases.parquet"
    clients_path = SYNTHETIC_DIR / "clients.parquet"
    if not purchases_path.exists():
        raise FileNotFoundError(f"Нужен файл {purchases_path}")
    if not clients_path.exists():
        raise FileNotFoundError(f"Нужен файл {clients_path}")

    purchases = pd.read_parquet(purchases_path)
    clients = pd.read_parquet(clients_path)
    print(f"Загружено покупок: {len(purchases):,}, клиентов: {len(clients):,}")

    policies = generate_policies(purchases, clients)
    out_path = SYNTHETIC_DIR / "policies.parquet"
    policies.to_parquet(out_path, index=False)

    print(f"\nСгенерировано полисов: {len(policies):,}")
    print(f"Сохранено: {out_path}")
    print(f"Размер: {out_path.stat().st_size / 1024**2:.1f} MB")

    print("\n=== Распределение событий ===")
    n_renewed = policies["event_observed"].sum()
    n_total = len(policies)
    print(f"Продлено (event=1):     {n_renewed:>6,} ({n_renewed/n_total*100:.1f}%)")
    print(f"Цензурировано (event=0): {n_total-n_renewed:>6,} ({(n_total-n_renewed)/n_total*100:.1f}%)")

    print("\n=== Распределение длительности (дни) ===")
    print(f"  Медиана: {policies['duration_days'].median():.0f}")
    print(f"  Среднее: {policies['duration_days'].mean():.0f}")
    print(f"  25/75%: {policies['duration_days'].quantile(0.25):.0f} / "
          f"{policies['duration_days'].quantile(0.75):.0f}")

    print("\n=== Связь контактов и продления ===")
    print(policies.groupby(
        pd.cut(policies["contacts_60d_before_end"], bins=[-1, 0, 1, 3, 100])
    , observed=True)["event_observed"].agg(["count", "mean"]).round(3))
