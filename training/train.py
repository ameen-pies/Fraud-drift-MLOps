"""Train RandomForestClassifier, evaluate, save model."""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train = pd.read_csv(DATA_DIR / "raw" / "train.csv")
    val = pd.read_csv(DATA_DIR / "raw" / "val.csv")

    feature_cols = [c for c in train.columns if c != "is_fraud"]

    X_train, y_train = train[feature_cols], train["is_fraud"]
    X_val, y_val = val[feature_cols], val["is_fraud"]

    return X_train, X_val, y_train, y_val


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: RandomForestClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
    }

    print("\n=== Classification Report ===")
    print(classification_report(y_val, y_pred, target_names=["legit", "fraud"]))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_val, y_pred))

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}")

    return metrics


def save_model(model: RandomForestClassifier) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / "model_v1.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {path}")


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()
    model = train_model(X_train, y_train)
    metrics = evaluate(model, X_val, y_val)
    save_model(model)
