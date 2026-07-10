"""FastAPI app with /health and /predict endpoints."""

import json
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure api/ is on path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import load_model, get_model

app = FastAPI(title="Fraud Detection API")
logger = logging.getLogger("uvicorn")

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "predictions.jsonl"


class PredictRequest(BaseModel):
    transaction_amount: float = Field(..., gt=0)
    transaction_hour: int = Field(..., ge=0, le=23)
    transaction_day: int = Field(..., ge=0, le=6)
    merchant_category: float
    distance_from_home: float
    distance_from_last_txn: float
    ratio_to_median_amount: float
    is_international: int = Field(..., ge=0, le=1)
    num_txns_last_1h: float
    num_txns_last_24h: float
    avg_amount_last_7d: float
    account_age_days: int = Field(..., gt=0)
    num_chargebacks: int = Field(..., ge=0)
    card_present: int = Field(..., ge=0, le=1)
    pin_used: int = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="0 = legit, 1 = fraud")
    fraud_probability: float = Field(..., ge=0, le=1)


class HealthResponse(BaseModel):
    status: str = "ok"


@app.on_event("startup")
def startup():
    load_model()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Model loaded and log directory ready")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = req.model_dump()
    X = [list(features.values())]

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": features,
        "prediction": prediction,
        "fraud_probability": round(probability, 4),
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return PredictResponse(prediction=prediction, fraud_probability=round(probability, 4))
