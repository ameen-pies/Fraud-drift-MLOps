"""Simulate production traffic with shifted distributions."""

import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
API_URL = "http://localhost:8000/predict"


def load_future_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "raw" / "future.csv")


def apply_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Shift feature distributions to simulate real drift."""
    drifted = df.copy()

    # Shift transaction_amount by +2 std (higher spends)
    std_amt = drifted["transaction_amount"].std()
    drifted["transaction_amount"] = drifted["transaction_amount"] + 2 * std_amt

    # Shift distance_from_home (card used further from home)
    drifted["distance_from_home"] = drifted["distance_from_home"] + 3

    # Shift ratio_to_median_amount (spending above normal)
    drifted["ratio_to_median_amount"] = drifted["ratio_to_median_amount"] * 1.5

    # Shift distance_from_last_txn (unusual txn patterns)
    drifted["distance_from_last_txn"] = drifted["distance_from_last_txn"] + 2

    # Shift num_txns_last_24h (more frequent transactions)
    drifted["num_txns_last_24h"] = drifted["num_txns_last_24h"] + 3

    # Shift avg_amount_last_7d (higher average spending)
    drifted["avg_amount_last_7d"] = drifted["avg_amount_last_7d"] * 1.4

    # Shift merchant_category (different merchant patterns)
    drifted["merchant_category"] = drifted["merchant_category"] + 2

    # Shift num_txns_last_1h (burst of transactions)
    drifted["num_txns_last_1h"] = drifted["num_txns_last_1h"] + 2

    return drifted


def send_requests(df: pd.DataFrame, delay: float = 0.05) -> list[dict]:
    """Send each row as a POST request, return logged responses."""
    feature_cols = [c for c in df.columns if c != "is_fraud"]
    results = []

    for i, row in df.iterrows():
        payload = {col: float(row[col]) for col in feature_cols}
        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            result = resp.json()
            results.append(result)
        except requests.ConnectionError:
            print(f"Request {i} failed: API not reachable")
            continue
        time.sleep(delay)

    return results


def run(n_samples: int = 200, drift: bool = True) -> None:
    print(f"Loading future data...")
    df = load_future_data()

    if drift:
        print("Applying drift perturbations...")
        df = apply_drift(df)

    # Sample n rows
    df = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)
    print(f"Sending {len(df)} requests to API...")

    results = send_requests(df)

    # Save traffic log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "traffic_log.jsonl"
    with open(log_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Logged {len(results)} responses to {log_path}")

    # Also save the reference snapshot for drift comparison
    feature_cols = [c for c in df.columns if c != "is_fraud"]
    reference = pd.read_csv(DATA_DIR / "reference" / "reference_data.csv")
    reference[feature_cols].to_csv(DATA_DIR / "reference" / "reference_snapshot.csv", index=False)
    df[feature_cols].to_csv(DATA_DIR / "reference" / "live_snapshot.csv", index=False)
    print("Saved reference and live snapshots for drift comparison")


if __name__ == "__main__":
    drift_flag = "--no-drift" not in sys.argv
    run(drift=drift_flag)
