"""
Генератор транзакционного и поведенческого агрегата клиентов.

Назначение: создать таблицу признаков, которые в реальном банке агрегируются
из транзакционной системы и логов мобильного приложения. Эти признаки —
главные предикторы покупки страхового продукта (модель проникновения).

Схема выходной таблицы (transactions_agg.parquet):
    client_id                    — связь с clients.parquet
    trans_count_3m, _12m         — число транзакций за окно
    total_spend_12m, avg_check_12m  — суммы и средний чек
    travel_spend_share_12m       — доля трат в MCC путешествий
    auto_spend_share_12m         — то же для автомобильных трат
    health_spend_share_12m       — то же для медицины
    app_sessions_90d             — сессий в приложении за 90 дней
    insurance_section_visits_365d — заходов в раздел страхования за год
    push_open_rate_90d           — доля открытых пушей

Все признаки строятся на горизонте до T - 90 дней (см. WINDOWS_DAYS), чтобы
не пересекаться с окном таргета модели проникновения.

Зависимости заложены реалистичные: активные клиенты (много транзакций) с большей
вероятностью совершают страховую покупку; высокая доля MCC путешествий тянет
к покупке ВЗР; высокая доля автомобильных трат — к КАСКО/ОСАГО; присутствие
истории страхования — самый сильный сигнал.
"""
import numpy as np
import pandas as pd

from config import RANDOM_SEED, SYNTHETIC_DIR


def generate_transactions_agg(clients_df: pd.DataFrame,
                                seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Генерирует агрегированный транзакционный/цифровой профиль на основе
    клиентского профиля. Один клиент → одна строка.

    Логика построения признаков:
      - Активность (число транзакций) масштабируется от сегмента и возраста.
      - Средний чек коррелирует с сегментом (premium > affluent > mass).
      - Доли MCC-категорий смещаются по полу и возрасту (медицина у старших,
        путешествия у молодых mass+, авто у мужчин 30-50).
      - Цифровая активность переносится из clients_df, расширяется длинным окном.
    """
    rng = np.random.default_rng(seed + 3)
    n = len(clients_df)

    # === Базовая активность через сегмент ===
    # Сегмент задаёт уровень транзакционной активности. Это приближение —
    # в реальности есть mass-клиенты с гипер-активностью, но средние такие.
    segment_to_base_trans = {
        "mass": 35, "affluent": 70, "premium": 120, "private": 180,
    }
    base_trans = clients_df["bank_segment"].map(segment_to_base_trans).values

    # Возраст: молодые активнее (карта в каждый чих), старшие реже но крупнее.
    age_factor = np.where(
        clients_df["age_years"] < 35, 1.3,
        np.where(clients_df["age_years"] < 55, 1.0, 0.7),
    )

    # === Транзакции за 12 месяцев ===
    # Пуассон со средним base_trans * age_factor.
    trans_count_12m = rng.poisson(lam=base_trans * age_factor).clip(0, 2000)

    # Транзакции за 3 месяца — примерно четверть от 12м с дисперсией.
    # Используем биномиальное: каждая транзакция за 12м с вероятностью ~0.25 попала в последние 3м.
    trans_count_3m = rng.binomial(n=trans_count_12m, p=0.27)

    # === Средний чек ===
    # Лог-нормальное распределение с медианой по сегменту.
    segment_to_check_median = {
        "mass": 800, "affluent": 1800, "premium": 4500, "private": 12000,
    }
    check_medians = clients_df["bank_segment"].map(segment_to_check_median).values
    avg_check_12m = rng.lognormal(
        mean=np.log(check_medians),
        sigma=0.5,
        size=n,
    ).clip(100, 200_000)

    # Общая сумма трат — произведение числа транзакций и среднего чека.
    # Округляем до сотен рублей — типичная гранулярность в DWH.
    total_spend_12m = (trans_count_12m * avg_check_12m / 100).round() * 100

    # === MCC-доли: путешествия, авто, медицина ===
    # Эти три категории — главные триггеры страховых продуктов.
    # Базовая доля + смещение по демографии.

    # Путешествия: молодые mass+ ездят больше, старшие реже.
    travel_base = 0.05 + 0.03 * (clients_df["bank_segment"].isin(["affluent", "premium", "private"])).astype(int)
    travel_age_shift = np.where(
        clients_df["age_years"] < 40, 0.02,
        np.where(clients_df["age_years"] < 60, 0, -0.02),
    )
    # Beta-распределение даёт долю в [0,1] с заданным средним.
    # alpha=2, beta вычисляется из заданного среднего.
    travel_mean = np.clip(travel_base + travel_age_shift, 0.01, 0.30)
    travel_spend_share_12m = rng.beta(
        a=travel_mean * 10,
        b=(1 - travel_mean) * 10,
        size=n,
    )

    # Автомобиль: чаще у мужчин 25-55.
    auto_base = np.where(
        (clients_df["gender"] == "M") & (clients_df["age_years"].between(25, 55)),
        0.08, 0.03,
    )
    auto_spend_share_12m = rng.beta(
        a=auto_base * 10,
        b=(1 - auto_base) * 10,
        size=n,
    )

    # Медицина: чаще у женщин и старших.
    health_base = np.where(clients_df["age_years"] > 45, 0.06, 0.03)
    health_base = np.where(clients_df["gender"] == "F", health_base + 0.02, health_base)
    health_spend_share_12m = rng.beta(
        a=health_base * 10,
        b=(1 - health_base) * 10,
        size=n,
    )

    # === Цифровое поведение ===
    # Перенос app_sessions_30d на 90д через мультипликатор + шум.
    # В реальности это были бы независимые агрегаты по разным окнам;
    # тут аппроксимируем.
    app_sessions_90d = (clients_df["app_sessions_30d"].values * 2.7
                        + rng.normal(0, 5, n)).clip(0, 300).astype(int)

    # Заходы в страховой раздел за год — расширение 90д.
    insurance_section_visits_365d = (
        clients_df["insurance_section_visits_90d"].values * 3.5
        + rng.poisson(0.5, n)
    ).clip(0, 100).astype(int)

    # Доля открытых пушей: бимодальная — у активных высокая, у пассивных низкая.
    is_engaged = clients_df["app_sessions_30d"].values > 8
    push_open_rate_90d = np.where(
        is_engaged,
        rng.beta(a=4, b=2, size=n),     # медиана ~0.65
        rng.beta(a=1, b=4, size=n),     # медиана ~0.20
    )

    # === Дни с последней транзакции ===
    # Активные — единицы дней, неактивные — десятки/сотни.
    days_since_last_trans = np.where(
        trans_count_3m > 0,
        rng.exponential(scale=3, size=n).clip(0, 90),
        rng.exponential(scale=60, size=n).clip(30, 365),
    ).astype(int)

    df = pd.DataFrame({
        "client_id": clients_df["client_id"].values,
        "trans_count_3m":               trans_count_3m,
        "trans_count_12m":              trans_count_12m,
        "total_spend_12m":              total_spend_12m,
        "avg_check_12m":                avg_check_12m.round(2),
        "travel_spend_share_12m":       travel_spend_share_12m.round(4),
        "auto_spend_share_12m":         auto_spend_share_12m.round(4),
        "health_spend_share_12m":       health_spend_share_12m.round(4),
        "app_sessions_90d":             app_sessions_90d,
        "insurance_section_visits_365d": insurance_section_visits_365d,
        "push_open_rate_90d":           push_open_rate_90d.round(4),
        "days_since_last_trans":        days_since_last_trans,
    })

    return df


if __name__ == "__main__":
    clients_path = SYNTHETIC_DIR / "clients.parquet"
    if not clients_path.exists():
        raise FileNotFoundError(
            f"Сначала запусти clients_generator.py — нужен файл {clients_path}"
        )

    clients_df = pd.read_parquet(clients_path)
    print(f"Загружено клиентов: {len(clients_df):,}")

    transactions_agg = generate_transactions_agg(clients_df)
    out_path = SYNTHETIC_DIR / "transactions_agg.parquet"
    transactions_agg.to_parquet(out_path, index=False)

    print(f"\nСохранено: {out_path}")
    print(f"Размер: {out_path.stat().st_size / 1024**2:.1f} MB")

    print("\n=== Распределение трансакционной активности по сегменту ===")
    merged = transactions_agg.merge(
        clients_df[["client_id", "bank_segment"]], on="client_id"
    )
    print(merged.groupby("bank_segment")[
        ["trans_count_12m", "avg_check_12m", "total_spend_12m"]
    ].median().round(0).to_string())

    print("\n=== Доли MCC по полу и возрастной группе ===")
    merged["age_group"] = pd.cut(
        clients_df["age_years"],
        bins=[0, 35, 55, 100],
        labels=["18-34", "35-54", "55+"],
    )
    print(merged.groupby(["gender", "age_group"], observed=True)[
        ["travel_spend_share_12m", "auto_spend_share_12m", "health_spend_share_12m"]
    ].mean().round(3).to_string())
