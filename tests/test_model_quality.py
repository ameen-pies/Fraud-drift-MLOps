"""Assert model quality metrics meet threshold."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import pytest
from sklearn.metrics import f1_score, recall_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model_v1.pkl"

F1_THRESHOLD = 0.15
RECALL_THRESHOLD = 0.05


@pytest.fixture(scope="module")
def model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def val_data():
    val = pd.read_csv(DATA_DIR / "raw" / "val.csv")
    feature_cols = [c for c in val.columns if c != "is_fraud"]
    return val[feature_cols], val["is_fraud"]


def test_model_exists():
    assert MODEL_PATH.exists(), "model_v1.pkl not found — run training first"


def test_f1_above_threshold(model, val_data):
    X_val, y_val = val_data
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    print(f"\nF1 score: {f1:.4f} (threshold: {F1_THRESHOLD})")
    assert f1 >= F1_THRESHOLD, f"F1 {f1:.4f} below threshold {F1_THRESHOLD}"


def test_recall_above_threshold(model, val_data):
    X_val, y_val = val_data
    y_pred = model.predict(X_val)
    rec = recall_score(y_val, y_pred)
    print(f"\nRecall: {rec:.4f} (threshold: {RECALL_THRESHOLD})")
    assert rec >= RECALL_THRESHOLD, f"Recall {rec:.4f} below threshold {RECALL_THRESHOLD}"
