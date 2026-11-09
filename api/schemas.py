"""
Request / response contract for POST /predict - the single source of truth for the
API, dashboard and SMS service. Human-readable version: docs/API_CONTRACT.md.

Quarter keys: LF = left front, RF = right front, LR = left rear, RR = right rear.
Sensor values may be null (sensor fault); the model falls back to the cow's own
history or herd medians, exactly as ml/features.py does during training.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Breed = Literal["exotic", "local"]
Grazing = Literal["zero", "tethered", "open"]
RiskLevel = Literal["low", "medium", "high"]

# Physically possible ranges; anything outside -> 422
EC_RANGE = dict(ge=0, le=20)          # mS/cm
PH_RANGE = dict(ge=5, le=9)
TEMP_RANGE = dict(ge=30, le=45)       # deg C

REQUEST_EXAMPLE = {
    "cow_id": "COW-012",
    "farm_id": "FARM-01",
    "timestamp": "2026-11-08T07:30:00",
    "ec_ms_cm": 6.8,
    "ec_quarters": {"LF": 5.4, "RF": 5.5, "LR": 7.9, "RR": 5.6},
    "milk_temp_c": 38.4,
    "ph": 6.95,
    "milk_yield_l": 6.2,
    "cmt_score": None,
    "breed": "exotic",
    "parity": 3,
    "days_in_milk": 120,
    "grazing_system": "zero",
}

RESPONSE_EXAMPLE = {
    "cow_id": "COW-012",
    "risk_score": 0.82,
    "risk_level": "high",
    "top_reasons": [
        {"feature": "ec_quarter_diff", "effect": "+0.21",
         "text": "Left rear quarter conductivity much higher than others"},
        {"feature": "ec_vs_baseline", "effect": "+0.15",
         "text": "Conductivity 22% above this cow's 7-day average"},
        {"feature": "ph", "effect": "+0.08", "text": "Milk pH higher than normal"},
    ],
    "advice": "Do a CMT test on this cow today and call your vet.",
    "model_version": "rf-smote-v1",
    "alert": False,
}


class ECQuarters(BaseModel):
    """Electrical conductivity of the milk from each quarter (mS/cm). Any quarter may be null."""
    LF: float | None = Field(None, **EC_RANGE, description="Left front quarter EC, mS/cm")
    RF: float | None = Field(None, **EC_RANGE, description="Right front quarter EC, mS/cm")
    LR: float | None = Field(None, **EC_RANGE, description="Left rear quarter EC, mS/cm")
    RR: float | None = Field(None, **EC_RANGE, description="Right rear quarter EC, mS/cm")


class PredictRequest(BaseModel):
    cow_id: str = Field(..., min_length=1, max_length=64, description="Unique cow identifier")
    farm_id: str | None = Field(None, max_length=64, description="Farm identifier")
    timestamp: datetime | None = Field(
        None, description="Milking time (ISO 8601, farm local time). If omitted, server time is used.")
    ec_ms_cm: float | None = Field(
        None, **EC_RANGE, description="Whole-milk EC, mS/cm. Used only when all quarter readings are missing.")
    ec_quarters: ECQuarters | None = Field(
        None, description="Per-quarter EC, mS/cm. The main signal: one infected quarter stands out.")
    milk_temp_c: float | None = Field(None, **TEMP_RANGE, description="Milk temperature at milking, deg C")
    ph: float | None = Field(None, **PH_RANGE, description="Milk pH")
    milk_yield_l: float | None = Field(None, ge=0, le=60, description="Litres at this milking")
    cmt_score: int | None = Field(
        None, ge=0, le=4,
        description="California Mastitis Test, 0-4 (= field scale 1-5 minus 1). Optional; stored as a "
                    "future training label (positive = 2 or more). Not used for the prediction.")
    breed: Breed = Field(..., description="exotic (Friesian / crosses) or local (Ankole / Zebu)")
    parity: int = Field(..., ge=1, le=15, description="Number of calvings")
    days_in_milk: int | None = Field(
        None, ge=0, le=700,
        description="Days since the last calving. May be omitted for a cow whose calving date is on record "
                    "(it is then calculated); required for a cow the system has never seen.")
    grazing_system: Grazing = Field(..., description="zero-grazing, tethered or open grazing")

    model_config = {"json_schema_extra": {"examples": [REQUEST_EXAMPLE]}}


class Reason(BaseModel):
    feature: str = Field(..., description="Model feature that drives this reason, e.g. 'ec_quarter_diff'")
    effect: str = Field(..., pattern=r"^[+-]\d+\.\d{2}$",
                        description="Signed contribution to the risk score (SHAP), 2 decimals, e.g. '+0.21'")
    text: str = Field(..., description="Plain-English explanation for the farmer")


class PredictResponse(BaseModel):
    cow_id: str
    risk_score: float = Field(..., ge=0, le=1, description="Model score 0-1 (higher = more likely mastitis)")
    risk_level: RiskLevel = Field(..., description="Band for THIS reading: low / medium / high")
    top_reasons: list[Reason] = Field(..., min_length=1, max_length=3, description="1 to 3 main reasons")
    advice: str = Field(..., description="What the farmer should do, by risk level")
    model_version: str = Field(..., description="From the saved model metadata, e.g. 'rf-smote-v1'")
    alert: bool = Field(..., description="True only when this AND the cow's previous reading are both "
                                         "'high'. Only alert=true should trigger an SMS.")

    model_config = {"json_schema_extra": {"examples": [RESPONSE_EXAMPLE]}}


class ErrorResponse(BaseModel):
    error: str = Field(..., examples=["validation_error"])
    message: str = Field(..., examples=["breed: Input should be 'exotic' or 'local'"])
    detail: list[dict] = Field(default_factory=list, description="Per-field errors (FastAPI format)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    trained_at: str
    threshold: float
    risk_bands: dict
    alert_rule: str
    features: list[str]
    positive_definition: str
    data: str
    test_metrics: dict | None
    cv_results: list[dict] | None
