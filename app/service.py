"""
The reading pipeline (app/ layer): everything that happens when a milking reading arrives.

handle_reading() is used by POST /predict (api/routes/predict.py), the seed script and the
/demo replay, so they behave identically:
  1. find/create the farm and cow; work out days in milk (sent, or from calving_date)
  2. load the cow's recent readings -> features -> model score + SHAP (app/predictor.py)
  3. alert (per reading, docs/API_CONTRACT.md): this AND the previous reading are high
  4. farmer-friendly reasons + advice (app/reasons.py)
  5. save the reading (re-sending the same cow + timestamp updates it)
  6. alert episode + SMS (app/alerts.py, app/notify.py)
  7. clean-baseline flag: a high reading, or one taken while an episode is active, is stored
     with baseline_exclude=True so it never becomes part of the cow's "normal"
Also here: the data behind GET /history and GET /model-info.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.schemas import PredictRequest, PredictResponse
from app.alerts import active_alert, is_alert, update_episode
from app.config import ALERT_MAX_GAP, HISTORY_DAYS, QUARTER_KEYS
from app.models import Cow, Farm, Reading
from app.predictor import Predictor, get_predictor, risk_bands
from app.reasons import advice, top_reasons
from ml.features import suspect_quarter

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "reports" / "metrics.json"
METADATA_PATH = ROOT / "models" / "metadata.json"
TEXT_COLUMNS = ("cow_id", "timestamp", "breed", "grazing_system", "baseline_exclude")  # not cast to float
SECONDS_PER_HOUR = 3600


class InvalidReadingError(Exception):
    """A request that is valid JSON but can't be processed (-> 422 with a clear message)."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field, self.message = field, message


class ModelNotLoadedError(Exception):
    """No trained model in models/ (-> 503)."""


# --------------------------------------------------------------------------- steps
def get_or_create_cow(session: Session, request: PredictRequest) -> Cow:
    """The cow (and farm) for this reading, created on first sight and updated from the request."""
    farm = None
    if request.farm_id:
        farm = session.get(Farm, request.farm_id) or Farm(id=request.farm_id, name=request.farm_id)
        session.add(farm)
    cow = session.execute(select(Cow).where(Cow.tag == request.cow_id)).scalar_one_or_none()
    if cow is None:
        cow = Cow(tag=request.cow_id, breed=request.breed, parity=request.parity, grazing_system=request.grazing_system)
        session.add(cow)
    # The latest request is the most up-to-date description of the cow
    cow.breed, cow.parity, cow.grazing_system = request.breed, request.parity, request.grazing_system
    if farm is not None:
        cow.farm = farm
    return cow


def resolve_days_in_milk(cow: Cow, request: PredictRequest, timestamp: datetime) -> int:
    """Days in milk as sent (and remember the calving date), or calculated from the calving date."""
    if request.days_in_milk is not None:
        cow.calving_date = (timestamp - timedelta(days=request.days_in_milk)).date()  # remember for next time
        return request.days_in_milk
    days = cow.days_in_milk(timestamp)
    if days is None or days < 0:
        raise InvalidReadingError("days_in_milk", f"required: cow {request.cow_id} has no calving date on record")
    return days


def reading_to_row(reading: Reading, cow: Cow) -> dict:
    """DB reading -> one row with the training-CSV column names used by ml/features.py."""
    return {
        "cow_id": cow.tag,
        "timestamp": reading.timestamp,
        "breed": cow.breed,
        "parity": cow.parity,
        "days_in_milk": reading.days_in_milk,
        "grazing_system": cow.grazing_system,
        "ec_ms_cm": reading.ec_ms_cm,
        "ph": reading.ph,
        "milk_temp_c": reading.milk_temp_c,
        "milk_yield_l": reading.milk_yield_l,
        "baseline_exclude": bool(reading.baseline_exclude),
        **{f"ec_{key.lower()}": getattr(reading, f"ec_{key}") for key in QUARTER_KEYS},
    }


def new_reading_row(cow: Cow, request: PredictRequest, timestamp: datetime, days_in_milk: int) -> dict:
    """The incoming reading as a row with the training-CSV column names."""
    quarters = request.ec_quarters.model_dump() if request.ec_quarters else {}
    return {
        "cow_id": cow.tag,
        "timestamp": timestamp,
        "breed": cow.breed,
        "parity": cow.parity,
        "days_in_milk": days_in_milk,
        "grazing_system": cow.grazing_system,
        "ec_ms_cm": request.ec_ms_cm,
        "baseline_exclude": False,  # its own flag never affects its own features
        "ph": request.ph,
        "milk_temp_c": request.milk_temp_c,
        "milk_yield_l": request.milk_yield_l,
        **{f"ec_{key.lower()}": quarters.get(key) for key in QUARTER_KEYS},
    }


def load_history(session: Session, cow: Cow, timestamp: datetime) -> list[Reading]:
    """The cow's readings from the last HISTORY_DAYS days, strictly before this milking."""
    return (
        session.execute(
            select(Reading)
            .where(
                Reading.cow_id == cow.id,
                Reading.timestamp >= timestamp - timedelta(days=HISTORY_DAYS),
                Reading.timestamp < timestamp,
            )
            .order_by(Reading.timestamp)
        )
        .scalars()
        .all()
    )


def model_frame(history: list[Reading], cow: Cow, new_row: dict) -> pd.DataFrame:
    """History + new reading as one frame for ml/features.py (missing values as NaN)."""
    frame = pd.DataFrame([reading_to_row(reading, cow) for reading in history] + [new_row])
    numeric = [col for col in frame.columns if col not in TEXT_COLUMNS]
    frame[numeric] = frame[numeric].astype(float)  # None -> NaN
    return frame


def save_reading(
    session: Session, cow: Cow, request: PredictRequest, timestamp: datetime, days_in_milk: int, result: dict
) -> Reading:
    """Insert the reading, or update it if this cow + timestamp was already sent."""
    reading = session.execute(
        select(Reading).where(Reading.cow_id == cow.id, Reading.timestamp == timestamp)
    ).scalar_one_or_none()
    if reading is None:
        reading = Reading(cow=cow, timestamp=timestamp)
        session.add(reading)
    quarters = request.ec_quarters.model_dump() if request.ec_quarters else {}
    reading.ec_ms_cm, reading.ph = request.ec_ms_cm, request.ph
    reading.milk_temp_c, reading.milk_yield_l = request.milk_temp_c, request.milk_yield_l
    reading.days_in_milk = days_in_milk
    for key in QUARTER_KEYS:
        setattr(reading, f"ec_{key}", quarters.get(key))
    if request.cmt_score is not None:  # never wipe a label entered via /cmt
        reading.cmt_score = request.cmt_score
    reading.risk_score, reading.risk_level, reading.alert = result["score"], result["level"], result["alert"]
    reading.top_reasons, reading.model_version = result["reasons"], result["version"]
    session.flush()  # reading.id
    return reading


def _require_predictor() -> Predictor:
    predictor = get_predictor()
    if predictor is None:
        raise ModelNotLoadedError("Model not loaded - run `python -m ml.train` first")
    return predictor


# --------------------------------------------------------------------------- main entry point
def handle_reading(session: Session, request: PredictRequest) -> tuple[PredictResponse, Reading]:
    """Score, explain, store and alert on one milking reading (see the module docstring)."""
    predictor = _require_predictor()
    timestamp = (request.timestamp or datetime.now()).replace(tzinfo=None, microsecond=0)

    cow = get_or_create_cow(session, request)
    days_in_milk = resolve_days_in_milk(cow, request, timestamp)
    session.flush()  # cow.id

    history = load_history(session, cow, timestamp)
    new_row = new_reading_row(cow, request, timestamp, days_in_milk)
    score, level, shap_row, feature_row = predictor.score(model_frame(history, cow, new_row))

    alert = is_alert(level, timestamp, history[-1] if history else None)
    quarter = suspect_quarter(new_row)  # 'lr' etc. or None
    reasons = top_reasons(shap_row, predictor.features, feature_row, quarter, level, has_history=bool(history))

    result = {"score": score, "level": level, "alert": alert, "reasons": reasons, "version": predictor.version}
    reading = save_reading(session, cow, request, timestamp, days_in_milk, result)
    update_episode(session, cow, reading, quarter.upper() if quarter else None)
    session.flush()
    reading.baseline_exclude = level == "high" or active_alert(session, cow.id) is not None
    session.commit()

    response = PredictResponse(
        cow_id=request.cow_id,
        risk_score=round(score, 2),
        risk_level=level,
        top_reasons=reasons,
        advice=advice(level, quarter if level != "low" else None),
        model_version=predictor.version,
        alert=alert,
    )
    return response, reading


# --------------------------------------------------------------------------- read-only API data
def reading_json(reading: Reading, cow: Cow) -> dict:
    """A stored reading as returned by GET /history/{cow_id}."""
    return {
        "cow_id": cow.tag,
        "timestamp": reading.timestamp.isoformat(),
        "ec_ms_cm": reading.ec_ms_cm,
        "ec_quarters": {key: getattr(reading, f"ec_{key}") for key in QUARTER_KEYS},
        "milk_temp_c": reading.milk_temp_c,
        "ph": reading.ph,
        "milk_yield_l": reading.milk_yield_l,
        "days_in_milk": reading.days_in_milk,
        "cmt_score": reading.cmt_score,
        "risk_score": reading.risk_score,
        "risk_level": reading.risk_level,
        "alert": reading.alert,
        "top_reasons": reading.top_reasons,
        "model_version": reading.model_version,
    }


def cow_history(session: Session, cow_tag: str, limit: int) -> list[dict]:
    """Stored readings for a cow tag, newest first ([] for an unknown cow)."""
    cow = session.execute(select(Cow).where(Cow.tag == cow_tag)).scalar_one_or_none()
    if cow is None:
        return []
    readings = (
        session.execute(select(Reading).where(Reading.cow_id == cow.id).order_by(Reading.timestamp.desc()).limit(limit))
        .scalars()
        .all()
    )
    return [reading_json(reading, cow) for reading in readings]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def model_info() -> dict:
    """Model name, version, threshold, risk bands, alert rule and metrics (GET /model-info)."""
    bundle = _require_predictor().bundle
    metrics = _read_json(METRICS_PATH)
    metadata = _read_json(METADATA_PATH)
    return {
        "model_name": bundle["model_name"],
        "version": bundle["version"],
        "trained_at": bundle["trained_at"],
        "threshold": bundle["threshold"],
        "risk_bands": risk_bands(bundle["threshold"]),
        "alert_rule": f"alert=true when this and the previous reading of the cow are both 'high' "
        f"(at most {int(ALERT_MAX_GAP.total_seconds() // SECONDS_PER_HOUR)} h apart)",
        "features": bundle["features"],
        "positive_definition": bundle["positive_definition"],
        "data": bundle["data"],
        "test_metrics": metrics.get("test_metrics"),
        "cv_results": metadata.get("cv_results"),
    }


def health() -> dict:
    """Liveness + whether a model is loaded (GET /health)."""
    predictor = get_predictor()
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "model_version": predictor.version if predictor else None,
    }
