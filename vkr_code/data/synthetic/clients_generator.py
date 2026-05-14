"""
Генератор клиентского профиля.

Это базовая таблица, на которую опираются ВСЕ четыре модели. Один client_id
живёт во всех выгрузках с одним и тем же набором атрибутов. Это критично:
если в воронке у клиента возраст 35, а в LTV он же 50 — пайплайн ломается.

Распределения подобраны так, чтобы воспроизвести типичный профиль клиентской
базы российского розничного банка по публичным агрегатам (отчёты ЦБ РФ).
В реальной выгрузке эти распределения возьмутся из живых данных.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from config import N_CLIENTS, RANDOM_SEED, SYNTHETIC_DIR


def generate_clients(n_clients: int = N_CLIENTS,
                     seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Генерирует клиентский профиль из n_clients записей.

    Демография — три блока: возраст, пол, регион (упрощённо, через сегмент рынка).
    Банковский профиль — сегмент, стаж, набор продуктов.
    Цифровое поведение — активность в приложении.

    Все распределения параметризованы один раз на основе наблюдаемой статистики
    розничного банковского сектора. Никакой случайной "красоты" — каждое
    допущение выбрано осознанно и задокументировано в коде.
    """
    rng = np.random.default_rng(seed)

    # client_id — последовательный, без претензии на правдоподобие хэша.
    # На реальных данных это будет хэш от настоящего id.
    client_ids = [f"C{i:08d}" for i in range(n_clients)]

    # === Демография ===
    # Возраст: сдвинутое нормальное распределение с центром 38, обрезка до [18, 75].
    # Центр выше медианного по РФ, потому что банковская активность смещена к 30-50 годам.
    age = rng.normal(loc=38, scale=12, size=n_clients).clip(18, 75).astype(int)

    # Пол: 52/48 в пользу женщин — реалистично для розничной базы российских банков.
    gender = rng.choice(["F", "M"], size=n_clients, p=[0.52, 0.48])

    # Регион: через групповое деление "Москва / СПб / города-миллионники / прочие".
    # Это упрощение — в реальных данных будут все 85 субъектов.
    region = rng.choice(
        ["MSK", "SPB", "MILLION", "OTHER"],
        size=n_clients,
        p=[0.18, 0.07, 0.25, 0.50],
    )

    # === Банковский профиль ===
    # Сегмент: стандартная пирамида банковской базы — большинство в mass.
    bank_segment = rng.choice(
        ["mass", "affluent", "premium", "private"],
        size=n_clients,
        p=[0.78, 0.15, 0.06, 0.01],
    )

    # Стаж в банке: экспоненциальное распределение со средним 4 года.
    # Сильный хвост старых клиентов — характерно для крупных банков.
    tenure_bank_months = rng.exponential(scale=48, size=n_clients).clip(1, 240).astype(int)

    # Продукты банка: число активных продуктов от 1 до 6, со смещением к 2-3.
    # Геометрическое распределение хорошо ловит "большинство держит мало, единицы много".
    products_count_bank = rng.geometric(p=0.45, size=n_clients).clip(1, 6)

    # Карты и кредитные продукты — Bernoulli с зависимостью от сегмента.
    # Premium-клиенты с большей вероятностью имеют все продукты.
    segment_uplift = pd.Series(bank_segment).map(
        {"mass": 0.0, "affluent": 0.15, "premium": 0.25, "private": 0.30}
    ).values

    has_debit_card = (rng.random(n_clients) < (0.92 + segment_uplift * 0.05)).astype(int)
    has_credit_card = (rng.random(n_clients) < (0.45 + segment_uplift)).astype(int)
    has_deposit = (rng.random(n_clients) < (0.30 + segment_uplift * 1.2)).astype(int)
    has_mortgage = (rng.random(n_clients) < (0.08 + segment_uplift * 0.5)).astype(int)

    # === Цифровое поведение ===
    # Сессий в приложении за 30 дней: пуассон со средним, зависящим от возраста.
    # Молодые активнее в мобильном банке.
    age_factor = np.where(age < 35, 1.5, np.where(age < 50, 1.0, 0.6))
    app_sessions_30d = rng.poisson(lam=12 * age_factor).clip(0, 100)

    # Заходы в раздел страхования за 90 дней — намного реже, чем общая активность.
    # Большинство клиентов туда не заходят вообще.
    insurance_section_visits_90d = rng.poisson(lam=0.8 * age_factor).clip(0, 50)

    # Дней с последнего входа: для активных клиентов малое число, для отвалившихся большое.
    # Смесь двух экспонент.
    is_active = rng.random(n_clients) < 0.85
    last_login_days_ago = np.where(
        is_active,
        rng.exponential(scale=5, size=n_clients).clip(0, 30),
        rng.exponential(scale=60, size=n_clients).clip(30, 365),
    ).astype(int)

    # === История страховых покупок ===
    # Доля клиентов с любой историей страхования. Зависит от стажа и сегмента.
    # Это будет таргет для модели проникновения.
    p_ever_bought = 0.05 + 0.0015 * tenure_bank_months / 12 + 0.05 * segment_uplift
    p_ever_bought = np.clip(p_ever_bought, 0, 0.4)
    ever_bought_insurance = (rng.random(n_clients) < p_ever_bought).astype(int)

    df = pd.DataFrame({
        "client_id": client_ids,
        "age_years": age,
        "gender": gender,
        "region_code": region,
        "bank_segment": bank_segment,
        "tenure_bank_months": tenure_bank_months,
        "products_count_bank": products_count_bank,
        "has_debit_card": has_debit_card,
        "has_credit_card": has_credit_card,
        "has_deposit": has_deposit,
        "has_mortgage": has_mortgage,
        "app_sessions_30d": app_sessions_30d,
        "insurance_section_visits_90d": insurance_section_visits_90d,
        "last_login_days_ago": last_login_days_ago,
        "ever_bought_insurance": ever_bought_insurance,
    })

    return df


if __name__ == "__main__":
    # Запуск как скрипта: генерируем и сохраняем клиентский профиль.
    df = generate_clients()
    out_path = SYNTHETIC_DIR / "clients.parquet"
    df.to_parquet(out_path, index=False)

    # Краткая сводка для проверки, что распределения вышли разумными.
    print(f"Сгенерировано клиентов: {len(df):,}")
    print(f"Сохранено: {out_path}")
    print(f"\nРаспределение по сегменту:\n{df['bank_segment'].value_counts(normalize=True).round(3)}")
    print(f"\nДоля купивших страхование: {df['ever_bought_insurance'].mean():.3f}")
    print(f"Средний возраст: {df['age_years'].mean():.1f}")
    print(f"Размер на диске: {out_path.stat().st_size / 1024**2:.1f} MB")
