"""Load trained model at startup."""

import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model_v1.pkl"

_model = None


def load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def get_model():
    return _model
