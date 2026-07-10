"""Smoke tests for FastAPI endpoints."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

from model_loader import load_model
from main import app

load_model()
client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_prediction():
    payload = {
        "transaction_amount": 150.50,
        "transaction_hour": 14,
        "transaction_day": 2,
        "merchant_category": 5.0,
        "distance_from_home": 10.0,
        "distance_from_last_txn": 5.0,
        "ratio_to_median_amount": 1.2,
        "is_international": 0,
        "num_txns_last_1h": 1.0,
        "num_txns_last_24h": 3.0,
        "avg_amount_last_7d": 120.0,
        "account_age_days": 365,
        "num_chargebacks": 0,
        "card_present": 1,
        "pin_used": 1,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["fraud_probability"] <= 1


def test_predict_rejects_bad_payload():
    resp = client.post("/predict", json={"transaction_amount": -1})
    assert resp.status_code == 422
