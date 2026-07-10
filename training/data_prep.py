"""Data preparation: generate synthetic fraud data, split, save reference CSV."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REFERENCE_DIR = DATA_DIR / "reference"
RAW_DIR = DATA_DIR / "raw"


def generate_fraud_data(n_samples: int = 10000, fraud_ratio: float = 0.03) -> pd.DataFrame:
    """Generate synthetic fraud-like data with controlled imbalance."""
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_clusters_per_class=2,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.02,
        random_state=42,
    )

    columns = [
        "transaction_amount",
        "transaction_hour",
        "transaction_day",
        "merchant_category",
        "distance_from_home",
        "distance_from_last_txn",
        "ratio_to_median_amount",
        "is_international",
        "num_txns_last_1h",
        "num_txns_last_24h",
        "avg_amount_last_7d",
        "account_age_days",
        "num_chargebacks",
        "card_present",
        "pin_used",
    ]

    df = pd.DataFrame(X, columns=columns)
    df["is_fraud"] = y

    # Scale transaction_amount to be positive and realistic
    df["transaction_amount"] = np.abs(df["transaction_amount"]) * 500 + 1
    df["transaction_hour"] = df["transaction_hour"].clip(0, 23).astype(int)
    df["transaction_day"] = df["transaction_day"].clip(0, 6).astype(int)
    df["account_age_days"] = np.abs(df["account_age_days"]).astype(int) + 30
    df["num_chargebacks"] = np.abs(df["num_chargebacks"]).clip(0, 10).astype(int)
    df["card_present"] = (df["card_present"] > 0).astype(int)
    df["pin_used"] = (df["pin_used"] > 0).astype(int)
    df["is_international"] = (df["is_international"] > 0).astype(int)

    return df


def split_and_save(df: pd.DataFrame) -> None:
    """Split into train / val / reference / future and save to disk."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    train_val, future = train_test_split(df, test_size=0.15, random_state=42, stratify=df["is_fraud"])
    train, val = train_test_split(train_val, test_size=0.2, random_state=42, stratify=train_val["is_fraud"])

    # Reference = a clean slice of train (baseline for drift comparison)
    reference, _ = train_test_split(train, test_size=0.7, random_state=42, stratify=train["is_fraud"])

    train.to_csv(RAW_DIR / "train.csv", index=False)
    val.to_csv(RAW_DIR / "val.csv", index=False)
    future.to_csv(RAW_DIR / "future.csv", index=False)
    reference.to_csv(REFERENCE_DIR / "reference_data.csv", index=False)

    print(f"Train:       {len(train):>5} rows  (fraud: {train['is_fraud'].sum()})")
    print(f"Val:         {len(val):>5} rows  (fraud: {val['is_fraud'].sum()})")
    print(f"Reference:   {len(reference):>5} rows  (fraud: {reference['is_fraud'].sum()})")
    print(f"Future:      {len(future):>5} rows  (fraud: {future['is_fraud'].sum()})")
    print(f"\nSaved to: {DATA_DIR}")


if __name__ == "__main__":
    df = generate_fraud_data()
    split_and_save(df)
