"""Close the loop: drift detected → retrain → register new model."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def read_drift_summary() -> dict:
    summary_path = REPORT_DIR / "drift_summary.json"
    if not summary_path.exists():
        print("No drift summary found. Run drift_check.py first.")
        sys.exit(1)
    with open(summary_path) as f:
        return json.load(f)


def retrain() -> dict:
    train = pd.read_csv(DATA_DIR / "raw" / "train.csv")
    val = pd.read_csv(DATA_DIR / "raw" / "val.csv")
    feature_cols = [c for c in train.columns if c != "is_fraud"]

    X_train, y_train = train[feature_cols], train["is_fraud"]
    X_val, y_val = val[feature_cols], val["is_fraud"]

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "model_v2.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    return {"metrics": metrics, "model_path": str(path)}


def main():
    summary = read_drift_summary()

    print("=" * 50)
    print("DRIFT CHECK RESULTS")
    print("=" * 50)
    print(f"  Drift detected:       {summary['drift_detected']}")
    print(f"  Threshold drift:      {summary['threshold_drift_detected']}")
    print(f"  Columns drifted:      {summary['n_drifted']}/{summary['total_columns']}")
    print(f"  Drift ratio:          {summary['drift_ratio']:.1%}")
    print(f"  Threshold:            {summary['drift_threshold']:.0%}")
    print("=" * 50)

    if summary["threshold_drift_detected"]:
        print("\n>>> Drift exceeds threshold. Retraining...")
        result = retrain()
        print(f"\n  New model saved to: {result['model_path']}")
        print(f"  Metrics: F1={result['metrics']['f1']:.4f}, Recall={result['metrics']['recall']:.4f}")
        print(f"\n  Rollback available: model v1 still in models/")
    else:
        print("\n>>> Drift within bounds. No retrain needed.")


if __name__ == "__main__":
    main()
