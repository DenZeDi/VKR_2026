"""
Генератор истории покупок страховых полисов.

Назначение: воспроизвести таблицу транзакций страхового канала, на которой
обучается LTV-модель (BG/NBD + Gamma-Gamma).

Схема выходной таблицы выбрана так, чтобы совпадать с тем, что будет
выгружено в реальном контуре банка (см. Data_Specifications.md, раздел 4).
Это значит: тот же код модели обучится и на синтетике здесь, и на реальной
выгрузке там, без правок.

Колонки выходной таблицы `purchases.parquet`:
    client_id        — идентификатор клиента (хэш)
    policy_id        — идентификатор полиса (хэш)
    purchase_date    — дата оформления (date)
    purchase_amount  — премия по полису (руб.)
    product_code     — код страхового продукта
    is_renewal       — продление существующего полиса (bool)

ВАЖНО про is_renewal: в LTV-модели используются ТОЛЬКО первичные покупки
(is_renewal == False). Продления моделируются отдельно через Cox PH —
смешение этих двух типов событий нарушает предпосылки BG/NBD
(нон-контрактуальные покупки). См. раздел 4 Data_Specifications.md,
вопрос 1, и обсуждение в Главе 3.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import (
    RANDOM_SEED, SYNTHETIC_DIR, CUTOFF_DATE, LTV_PARAMS,
)


# === Распределение продуктов в покупках ===
# Грубая модель: три категории продуктов, разные базовые чеки.
# Калибруется опционально из BI (топ-3 продукта по объёму).
PRODUCT_CATALOG = {
    "NSZH":      {"share": 0.45, "premium_multiplier": 1.6},   # накопительное страхование
    "ISZH":      {"share": 0.30, "premium_multiplier": 1.1},   # инвестиционное
    "PROPERTY":  {"share": 0.25, "premium_multiplier": 0.6},   # имущество
}


def _sample_purchase_count(n_clients: int, rng: np.random.Generator) -> np.ndarray:
    """
    Сэмплирует число покупок на клиента согласно LTV_PARAMS["purchase_count_distribution"].

    Логика: дискретное распределение для 1-5 покупок + геометрический хвост
    для 6+. Это типичная картина страхового канала: единицы клиентов с
    высокой частотой и большая масса однопокупочных.
    """
    dist = LTV_PARAMS["purchase_count_distribution"]
    discrete_counts = [k for k in dist if isinstance(k, int)]
    discrete_probs = [dist[k] for k in discrete_counts]
    p_six_plus = dist["6_plus"]

    # Сначала решаем "1-5 или 6+", потом для 1-5 — конкретное число
    is_six_plus = rng.random(n_clients) < p_six_plus
    counts = np.zeros(n_clients, dtype=int)

    # Для 1-5: нормализуем вероятности и сэмплируем
    probs_normalized = np.array(discrete_probs) / sum(discrete_probs)
    counts[~is_six_plus] = rng.choice(
        discrete_counts, size=(~is_six_plus).sum(), p=probs_normalized
    )

    # Для 6+: геометрическое распределение со средним 7
    counts[is_six_plus] = 6 + rng.geometric(p=0.5, size=is_six_plus.sum())

    return counts


def _sample_purchase_dates(n_purchases: int, T_days: int,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Сэмплирует даты покупок одного клиента.

    Первая покупка — равномерно в окне [0, T_days].
    Последующие — после первой, с экспоненциальным интервалом.
    Это аппроксимирует пуассоновский процесс покупок, на котором стоит BG/NBD.

    Возвращает массив дней от cutoff (отрицательные значения = в прошлом).
    """
    if n_purchases == 0:
        return np.array([])

    # Первая покупка: равномерно по всему окну T_days до cutoff.
    # Раньше требовалось min_observation_days — но это создавало "мёртвую зону"
    # последних 90 дней, в которой не было ни одной первой покупки. Это ломало
    # модель проникновения, у которой таргет — покупка в окне [T-90, T].
    # Сейчас разрешаем первую покупку в любой день; короткое окно после неё
    # естественным образом даёт frequency=0, что корректно обрабатывается BG/NBD.
    first_purchase_offset = rng.integers(1, T_days)

    if n_purchases == 1:
        return np.array([first_purchase_offset])

    # Последующие покупки: интервалы из экспоненциального распределения.
    # Средний интервал = время от первой покупки до cutoff, делённое на (n+1).
    # Это даёт расходящиеся интервалы и реалистичную recency.
    available_days = first_purchase_offset
    mean_interval = available_days / (n_purchases + 1)

    intervals = rng.exponential(scale=mean_interval, size=n_purchases - 1)

    # Кумулятивные даты от первой покупки.
    # clip чтобы не вылезти за cutoff (если интервалы слишком большие).
    subsequent_offsets = first_purchase_offset - np.cumsum(intervals)
    subsequent_offsets = np.clip(subsequent_offsets, 1, first_purchase_offset)

    all_offsets = np.concatenate([[first_purchase_offset], subsequent_offsets])
    return np.sort(all_offsets)[::-1]  # от первой к последней покупке (по убыванию offset)


def _sample_premium(n: int, segment: str, rng: np.random.Generator) -> np.ndarray:
    """
    Сэмплирует премию полиса в рублях.

    Лог-нормальное распределение с медианой LTV_PARAMS["premium_median_rub"]
    и параметром разброса. Сегмент клиента влияет на медиану через множитель
    (premium платит больше, mass — меньше).
    """
    median = LTV_PARAMS["premium_median_rub"]
    sigma = LTV_PARAMS["premium_lognorm_sigma"]

    segment_multiplier = {"mass": 0.85, "affluent": 1.20, "premium": 1.80, "private": 3.00}
    mult = segment_multiplier.get(segment, 1.0)

    # mu в лог-нормальном = log(медиана). Прибавляем log(mult) для сдвига.
    mu = np.log(median * mult)
    samples = rng.lognormal(mean=mu, sigma=sigma, size=n)

    return np.clip(samples, LTV_PARAMS["premium_min_rub"], LTV_PARAMS["premium_max_rub"])


def generate_purchases(clients_df: pd.DataFrame,
                        seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Главная функция: генерирует таблицу покупок для всех клиентов с историей.

    Шаги:
        1. Отобрать клиентов, у которых есть страховые покупки
           (share = LTV_PARAMS["share_with_purchases"]).
        2. Для каждого выбранного — сэмплировать число покупок.
        3. Для каждой покупки — дату, продукт, премию.
        4. Собрать в плоскую таблицу.

    Используется только client_id и bank_segment из clients_df. Остальные
    клиентские атрибуты понадобятся в других моделях, но не здесь — LTV
    работает на агрегатах транзакций (RFM-метрики).
    """
    rng = np.random.default_rng(seed + 2)
    cutoff = datetime.fromisoformat(CUTOFF_DATE)

    # === Отбор клиентов с историей страхования ===
    # Не делаем жёсткий бинарный отбор "купили / не купили". Вместо этого
    # каждый клиент имеет индивидуальную intensity (склонность к покупке),
    # зависящую от его атрибутов. Часть клиентов с высокой intensity сделает
    # одну или несколько покупок, часть — ни одной.
    #
    # Это критично: при жёстком отборе модель проникновения (XGBoost) находит
    # детерминированную связь "ever_bought_insurance → купит ещё", получая
    # AUC 0.92 за счёт data leakage. На реальных данных такой связи нет —
    # проникновение определяется поведенческими и транзакционными признаками,
    # а не флагом "был покупателем".
    n_clients_total = len(clients_df)

    # Базовая склонность к покупке. Зависит от сегмента, стажа, цифровой активности.
    # Эти зависимости — то, что должна выучить модель проникновения.
    segment_factor = clients_df["bank_segment"].map({
        "mass": 1.0, "affluent": 1.8, "premium": 2.5, "private": 3.0,
    }).values
    tenure_factor = np.clip(clients_df["tenure_bank_months"].values / 36, 0.5, 2.5)
    digital_factor = np.clip(clients_df["app_sessions_30d"].values / 12, 0.3, 2.0)
    visits_factor = 1 + 0.5 * np.clip(
        clients_df["insurance_section_visits_90d"].values, 0, 5
    )

    # Базовая интенсивность: примерно соответствует среднему числу покупок
    # за весь период наблюдения (36 месяцев).
    # Калибруем так, чтобы итоговая доля клиентов с покупкой была около
    # share_with_purchases (по умолчанию 18%).
    base_intensity = LTV_PARAMS["share_with_purchases"]
    intensity = base_intensity * segment_factor * tenure_factor * digital_factor * visits_factor

    # === Число покупок: Пуассон с заданной интенсивностью ===
    # Пуассон даёт реалистичное распределение: много нулей, длинный хвост.
    purchase_counts = rng.poisson(lam=intensity, size=n_clients_total)
    has_history = purchase_counts > 0
    clients_with_history = clients_df[has_history].reset_index(drop=True)
    purchase_counts_filtered = purchase_counts[has_history]
    n_buyers = len(clients_with_history)
    print(f"Клиентов с историей покупок: {n_buyers:,} из {n_clients_total:,}")
    n_total_purchases = int(purchase_counts_filtered.sum())
    print(f"Всего покупок: {n_total_purchases:,}")

    # === Развёрнем в "длинный" формат: одна строка = одна покупка ===
    # client_idx_per_purchase[i] = индекс клиента в clients_with_history,
    # совершившего i-ую покупку.
    client_idx_per_purchase = np.repeat(np.arange(n_buyers), purchase_counts_filtered)

    # === Даты покупок ===
    # Идём по клиентам, для каждого сэмплируем последовательность дат.
    # Цикл по клиентам, не по покупкам, — потому что даты внутри клиента связаны.
    all_offsets = []
    T_days = LTV_PARAMS["first_purchase_window_days"]
    for n_purch in purchase_counts_filtered:
        offsets = _sample_purchase_dates(int(n_purch), T_days, rng)
        all_offsets.append(offsets)
    purchase_offsets = np.concatenate(all_offsets)

    purchase_dates = pd.to_datetime([
        cutoff - timedelta(days=int(off)) for off in purchase_offsets
    ])

    # === Продукты и премии ===
    product_codes = list(PRODUCT_CATALOG.keys())
    product_shares = [PRODUCT_CATALOG[c]["share"] for c in product_codes]
    products = rng.choice(product_codes, size=n_total_purchases, p=product_shares)

    # Премия: сначала базовая по сегменту, потом множитель продукта.
    segments_per_purchase = clients_with_history["bank_segment"].values[client_idx_per_purchase]
    base_premiums = np.array([
        _sample_premium(1, seg, rng)[0] for seg in segments_per_purchase
    ])
    product_multipliers = np.array([
        PRODUCT_CATALOG[p]["premium_multiplier"] for p in products
    ])
    premiums = (base_premiums * product_multipliers).round(0)

    # === Метка "продление" ===
    # Простое правило: вторая и последующие покупки одного и того же продукта
    # у одного клиента считаются продлением. Первая покупка любого продукта —
    # is_renewal=False. Эта эвристика согласована с подходом из спецификации
    # (раздел 4): продления исключаются из LTV-модели, чтобы не пересекаться
    # с Cox PH.
    purchases_df = pd.DataFrame({
        "client_id": clients_with_history["client_id"].values[client_idx_per_purchase],
        "purchase_date": purchase_dates,
        "purchase_amount": premiums.astype(float),
        "product_code": products,
    })

    # Сортируем для корректного определения renewal
    purchases_df = purchases_df.sort_values(["client_id", "product_code", "purchase_date"]).reset_index(drop=True)
    purchases_df["is_renewal"] = purchases_df.duplicated(subset=["client_id", "product_code"], keep="first")

    # Хэш policy_id (простая нумерация)
    purchases_df["policy_id"] = [f"P{i:09d}" for i in range(len(purchases_df))]

    # Финальный порядок колонок — для совместимости с реальной выгрузкой
    purchases_df = purchases_df[[
        "client_id", "policy_id", "purchase_date",
        "purchase_amount", "product_code", "is_renewal",
    ]]

    return purchases_df


def print_summary(purchases_df: pd.DataFrame) -> None:
    """
    Печатает сводную статистику по сгенерированным покупкам.

    Эти числа полезно сравнивать с реальными агрегатами из BI:
    если медианы и квартили близки — синтетика откалибрована хорошо.
    """
    print("\n=== Сводка по покупкам ===")
    print(f"Всего покупок:        {len(purchases_df):,}")
    print(f"Уникальных клиентов:  {purchases_df['client_id'].nunique():,}")
    print(f"Из них первичные:     {(~purchases_df['is_renewal']).sum():,}")
    print(f"Продления:            {purchases_df['is_renewal'].sum():,}")

    print("\n=== Распределение числа покупок на клиента (только первичные) ===")
    primary = purchases_df[~purchases_df["is_renewal"]]
    counts_per_client = primary.groupby("client_id").size()
    for n in [1, 2, 3, 4, 5]:
        share = (counts_per_client == n).mean()
        print(f"  {n} покупок: {share*100:5.1f}%")
    share_six_plus = (counts_per_client >= 6).mean()
    print(f"  6+ покупок: {share_six_plus*100:5.1f}%")

    print("\n=== Распределение чека (премия полиса), руб. ===")
    premium = purchases_df["purchase_amount"]
    for q, label in [(0.25, "25%"), (0.50, "медиана"), (0.75, "75%"), (0.95, "95%")]:
        print(f"  {label:>8s}: {premium.quantile(q):>10,.0f}")

    print("\n=== Распределение по продуктам ===")
    print(purchases_df["product_code"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    clients_path = SYNTHETIC_DIR / "clients.parquet"
    if not clients_path.exists():
        raise FileNotFoundError(
            f"Сначала запусти clients_generator.py — нужен файл {clients_path}"
        )

    clients_df = pd.read_parquet(clients_path)
    print(f"Загружено клиентов: {len(clients_df):,}")

    purchases = generate_purchases(clients_df)

    out_path = SYNTHETIC_DIR / "purchases.parquet"
    purchases.to_parquet(out_path, index=False)
    print(f"\nСохранено: {out_path}")
    print(f"Размер на диске: {out_path.stat().st_size / 1024**2:.1f} MB")

    print_summary(purchases)
