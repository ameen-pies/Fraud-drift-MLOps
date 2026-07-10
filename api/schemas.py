"""Pydantic models for /predict request/response."""

from pydantic import BaseModel, Field


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
