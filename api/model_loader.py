"""Load trained model at startup."""

import pickle
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_model = None


def load_model():
    global _model
    if _model is None:
        # Load latest model version
        model_files = sorted(MODEL_DIR.glob("model_v*.pkl"))
        if not model_files:
            raise FileNotFoundError("No model found in models/")
        latest = model_files[-1]
        with open(latest, "rb") as f:
            _model = pickle.load(f)
        print(f"Loaded model from {latest.name}")
    return _model


def get_model():
    return _model
