"""
handle_reading(): everything that happens when a milking reading arrives.
Used by POST /predict, scripts/seed_db.py and the /demo replay, so they behave identically.

  1. find/create the farm and cow; work out days in milk (sent, or from calving_date)
  2. load the cow's last 8 days of readings -> features -> model score + SHAP
  3. alert (per reading, as in docs/API_CONTRACT.md): this AND the previous reading are high
  4. save the reading (re-sending the same cow + timestamp updates it)
  5. alert EPISODES:
       alert=true and no active episode  -> new alerts row + SMS
       alert=true and an active episode  -> attach the reading to it (no SMS)
       4 consecutive non-high readings   -> episode auto-resolves ("resolved_auto")
"""
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.reasons import advice, top_reasons
from api.schemas import PredictRequest, PredictResponse
from app.models import ACTIVE_ALERT_STATUSES, Alert, Cow, Farm, Reading
from app.notify import notify_new_alert
from app.predictor import get_predictor
from ml.features import suspect_quarter

HISTORY_DAYS = 8                       # enough for the 7-day baseline
ALERT_MAX_GAP = timedelta(hours=48)    # 2 highs further apart than this are not "consecutive"
AUTO_RESOLVE_AFTER = 4                 # consecutive non-high readings that close an episode
QUARTERS = ["LF", "RF", "LR", "RR"]


class InputProblem(Exception):
    """A request that is valid JSON but can't be processed (-> 422 with a clear message)."""
    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field, self.message = field, message


class ModelNotLoaded(Exception):
    pass


# ------------------------------------------------------------------ helpers
def get_or_create_cow(session: Session, req: PredictRequest) -> Cow:
    farm = None
    if req.farm_id:
        farm = session.get(Farm, req.farm_id) or Farm(id=req.farm_id, name=req.farm_id)
        session.add(farm)
    cow = session.execute(select(Cow).where(Cow.tag == req.cow_id)).scalar_one_or_none()
    if cow is None:
        cow = Cow(tag=req.cow_id, breed=req.breed, parity=req.parity, grazing_system=req.grazing_system)
        session.add(cow)
    # The latest request is the most up-to-date description of the cow
    cow.breed, cow.parity, cow.grazing_system = req.breed, req.parity, req.grazing_system
    if farm is not None:
        cow.farm = farm
    return cow


def resolve_days_in_milk(cow: Cow, req: PredictRequest, ts: datetime) -> int:
    if req.days_in_milk is not None:
        cow.calving_date = (ts - timedelta(days=req.days_in_milk)).date()   # remember for next time
        return req.days_in_milk
    dim = cow.days_in_milk(ts)
    if dim is None or dim < 0:
        raise InputProblem("days_in_milk", f"required: cow {req.cow_id} has no calving date on record")
    return dim


def reading_to_row(r: Reading, cow: Cow) -> dict:
    """DB reading -> one row with the training-CSV column names used by ml/features.py."""
    return {"cow_id": cow.tag, "timestamp": r.timestamp, "breed": cow.breed, "parity": cow.parity,
            "days_in_milk": r.days_in_milk, "grazing_system": cow.grazing_system,
            "ec_ms_cm": r.ec_ms_cm, "ph": r.ph, "milk_temp_c": r.milk_temp_c, "milk_yield_l": r.milk_yield_l,
            **{f"ec_{q.lower()}": getattr(r, f"ec_{q}") for q in QUARTERS}}


def is_alert(level: str, ts: datetime, previous: Reading | None) -> bool:
    """Contract rule: this AND the cow's previous reading are both 'high', at most 48 h apart."""
    return (level == "high" and previous is not None and previous.risk_level == "high"
            and ts - previous.timestamp <= ALERT_MAX_GAP)


def active_alert(session: Session, cow_id: int) -> Alert | None:
    q = (select(Alert).where(Alert.cow_id == cow_id, Alert.status.in_(ACTIVE_ALERT_STATUSES))
         .order_by(Alert.id.desc()))
    return session.execute(q.limit(1)).scalar_one_or_none()


def update_episode(session: Session, cow: Cow, reading: Reading, quarter: str | None) -> Alert | None:
    """Create / extend / auto-resolve the cow's alert episode. Returns a NEW alert, if one was made."""
    current = active_alert(session, cow.id)
    if reading.alert:
        if current is None:                                    # new episode -> new row + SMS
            new = Alert(cow=cow, reading_id=reading.id, last_reading_id=reading.id, reading_count=1,
                        created_at=reading.timestamp, status="open")
            session.add(new)
            session.flush()                                    # get new.id for the SMS log
            notify_new_alert(session, new, quarter, when=reading.timestamp)
            return new
        if current.last_reading_id != reading.id:              # ongoing episode -> attach
            current.last_reading_id = reading.id
            current.reading_count += 1
        return None

    # Not an alert reading: close the episode after 4 consecutive non-high readings
    if current is not None and reading.risk_level != "high":
        since = session.get(Reading, current.last_reading_id).timestamp
        recent = session.execute(
            select(Reading.risk_level).where(Reading.cow_id == cow.id, Reading.timestamp > since)
            .order_by(Reading.timestamp.desc()).limit(AUTO_RESOLVE_AFTER)).scalars().all()
        if len(recent) == AUTO_RESOLVE_AFTER and all(level != "high" for level in recent):
            current.status = "resolved_auto"
            current.closed_at = reading.timestamp
            current.note = f"Auto-resolved after {AUTO_RESOLVE_AFTER} consecutive non-high readings"
    return None


# ------------------------------------------------------------------ main entry point
def handle_reading(session: Session, req: PredictRequest) -> tuple[PredictResponse, Reading]:
    predictor = get_predictor()
    if predictor is None:
        raise ModelNotLoaded("Model not loaded - run `python -m ml.train` first")
    ts = (req.timestamp or datetime.now()).replace(tzinfo=None, microsecond=0)

    # 1) Farm, cow, days in milk
    cow = get_or_create_cow(session, req)
    dim = resolve_days_in_milk(cow, req, ts)
    session.flush()                                            # cow.id

    # 2) History (strictly before this milking) + the new reading -> model
    history = session.execute(
        select(Reading).where(Reading.cow_id == cow.id, Reading.timestamp >= ts - timedelta(days=HISTORY_DAYS),
                              Reading.timestamp < ts).order_by(Reading.timestamp)).scalars().all()
    quarters = req.ec_quarters.model_dump() if req.ec_quarters else {}
    new = {"cow_id": cow.tag, "timestamp": ts, "breed": cow.breed, "parity": cow.parity,
           "days_in_milk": dim, "grazing_system": cow.grazing_system, "ec_ms_cm": req.ec_ms_cm,
           "ph": req.ph, "milk_temp_c": req.milk_temp_c, "milk_yield_l": req.milk_yield_l,
           **{f"ec_{q.lower()}": quarters.get(q) for q in QUARTERS}}
    frame = pd.DataFrame([reading_to_row(r, cow) for r in history] + [new])
    num = [c for c in frame.columns if c not in ("cow_id", "timestamp", "breed", "grazing_system")]
    frame[num] = frame[num].astype(float)                     # None -> NaN
    score, level, shap_row, feats = predictor.score(frame)

    # 3) Per-reading alert + farmer-facing explanation
    alert = is_alert(level, ts, history[-1] if history else None)
    quarter = suspect_quarter(new)                            # 'lr' etc. or None
    reasons = top_reasons(shap_row, predictor.bundle["features"], feats, quarter, level,
                          has_history=bool(history))

    # 4) Save (update if this cow + timestamp was already sent)
    reading = session.execute(
        select(Reading).where(Reading.cow_id == cow.id, Reading.timestamp == ts)).scalar_one_or_none()
    if reading is None:
        reading = Reading(cow=cow, timestamp=ts)
        session.add(reading)
    reading.ec_ms_cm, reading.ph = req.ec_ms_cm, req.ph
    reading.milk_temp_c, reading.milk_yield_l, reading.days_in_milk = req.milk_temp_c, req.milk_yield_l, dim
    for q in QUARTERS:
        setattr(reading, f"ec_{q}", quarters.get(q))
    if req.cmt_score is not None:                             # never wipe a label entered via /cmt
        reading.cmt_score = req.cmt_score
    reading.risk_score, reading.risk_level, reading.alert = score, level, alert
    reading.top_reasons, reading.model_version = reasons, predictor.version
    session.flush()                                           # reading.id

    # 5) Episodes + SMS
    update_episode(session, cow, reading, quarter.upper() if quarter else None)
    session.commit()

    response = PredictResponse(
        cow_id=req.cow_id, risk_score=round(score, 2), risk_level=level, top_reasons=reasons,
        advice=advice(level, quarter if level != "low" else None),
        model_version=predictor.version, alert=alert)
    return response, reading
