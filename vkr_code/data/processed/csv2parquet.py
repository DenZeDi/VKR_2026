"""Утилита для безопасной конвертации CSV в Parquet.

Пример:
    python data/processed/csv2parquet.py --input data/processed/purchases.csv --output data/processed/purchases.parquet --sep ; --encoding cp1251
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def to_numeric_money(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sep", default=";")
    parser.add_argument("--encoding", default="cp1251")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding=args.encoding, sep=args.sep)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    for col in ["purchase_date", "policy_start_date", "policy_end_date", "birth_date", "client_start_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    for col in ["purchase_amount", "purchase_amount_model", "txn_amount_30d", "txn_amount_90d", "avg_txn_amount_90d"]:
        if col in df.columns:
            df[col] = to_numeric_money(df[col])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"Saved {out}: {df.shape}")
    print(df.dtypes)
    print(df.head())


if __name__ == "__main__":
    main()
