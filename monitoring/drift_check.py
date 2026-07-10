"""Evidently AI drift report + threshold logic."""

import json
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"

DRIFT_THRESHOLD = 0.3  # drift if >30% of columns drift


def run_drift_check() -> dict:
    reference_path = DATA_DIR / "reference" / "reference_snapshot.csv"
    live_path = DATA_DIR / "reference" / "live_snapshot.csv"

    if not reference_path.exists() or not live_path.exists():
        raise FileNotFoundError("Run simulate_traffic.py first to generate snapshots")

    reference = pd.read_csv(reference_path)
    live = pd.read_csv(live_path)

    column_mapping = ColumnMapping(
        numerical_features=[c for c in reference.columns],
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=live, column_mapping=column_mapping)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORT_DIR / "drift_report.html"
    report.save_html(str(html_path))
    print(f"HTML report saved to {html_path}")

    result = report.as_dict()

    # Parse DatasetDriftMetric
    dataset_drift = False
    n_drifted = 0
    total = 0
    for m in result["metrics"]:
        if m["metric"] == "DatasetDriftMetric":
            dataset_drift = m["result"]["dataset_drift"]
            n_drifted = m["result"]["number_of_drifted_columns"]
            total = m["result"]["number_of_columns"]

    # Parse DataDriftTable for column-level details
    drifted_columns = []
    for m in result["metrics"]:
        if m["metric"] == "DataDriftTable":
            for col_name, col_info in m["result"]["drift_by_columns"].items():
                if col_info.get("drift_detected"):
                    drifted_columns.append({
                        "column": col_name,
                        "test": col_info["stattest_name"],
                        "score": col_info["drift_score"],
                    })

    drift_ratio = n_drifted / total if total > 0 else 0
    threshold_drift = drift_ratio > DRIFT_THRESHOLD

    summary = {
        "drift_detected": dataset_drift,
        "threshold_drift_detected": threshold_drift,
        "drifted_columns": drifted_columns,
        "n_drifted": n_drifted,
        "total_columns": total,
        "drift_ratio": round(drift_ratio, 4),
        "drift_threshold": DRIFT_THRESHOLD,
    }

    json_path = REPORT_DIR / "drift_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to {json_path}")

    print(f"\n{'='*50}")
    print(f"Drift detected (Evidently):  {dataset_drift}")
    print(f"Threshold drift (>30%):      {threshold_drift}")
    print(f"Columns drifted:             {n_drifted}/{total} ({drift_ratio:.1%})")
    if drifted_columns:
        print(f"\nDrifted columns:")
        for col in drifted_columns:
            print(f"  - {col['column']}: {col['test']} (score: {col['score']:.4f})")
    print(f"{'='*50}")

    return summary


if __name__ == "__main__":
    run_drift_check()
