"""
Live demo for the /demo page (app/ layer): replays 3 simulated cows through the SAME code as
POST /predict (app.service.handle_reading) in a background thread, one milking every
`interval` seconds. The page polls status() and GET /api/cows to show progress.

The demo cows live on their own farm "DEMO" (tags "DEMO-..."), so the seeded history is never
touched. Their timestamps are shifted so the replay ends "now". Each start clears the previous run.
"""

import threading
import time
import traceback
from datetime import datetime

import pandas as pd
from sqlalchemy import delete, select

from api.schemas import PredictRequest
from app.db import SessionLocal
from app.models import Alert, Cow, Farm, Reading, SmsLog
from app.replay import CSV_PATH, held_out_farms, pick_cows, to_payload
from app.service import handle_reading

DEMO_FARM = "DEMO"
DEMO_FARM_DETAILS = {
    "name": "Demo farm",
    "district": "Live demo",
    "owner_name": "Demo farmer",
    "phone": "+256700000999",
}
DAYS_BEFORE_ONSET, DAYS_AFTER_ONSET = 4, 6
LOG_LENGTH = 30  # latest readings shown on the page
MAX_INTERVAL_S = 10.0
STATUS_KEYS = ["running", "step", "total", "error", "sick_cow", "log"]

_lock = threading.Lock()
STATE = {"running": False, "step": 0, "total": 0, "log": [], "error": None, "sick_cow": None}


def reset_demo_data() -> None:
    """Delete the previous demo run (demo cows, their readings, alerts and SMS log)."""
    with SessionLocal() as session:
        cow_ids = session.execute(select(Cow.id).where(Cow.farm_id == DEMO_FARM)).scalars().all()
        if cow_ids:
            session.execute(delete(SmsLog).where(SmsLog.cow_id.in_(cow_ids)))
            session.execute(delete(Alert).where(Alert.cow_id.in_(cow_ids)))
            session.execute(delete(Reading).where(Reading.cow_id.in_(cow_ids)))
            session.execute(delete(Cow).where(Cow.id.in_(cow_ids)))
        if session.get(Farm, DEMO_FARM) is None:
            session.add(Farm(id=DEMO_FARM, **DEMO_FARM_DETAILS))
        session.commit()


def demo_rows() -> tuple[pd.DataFrame, str]:
    """1 cow that develops mastitis + 2 healthy cows from the held-out test farms."""
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    sick, onset, healthy = pick_cows(df, held_out_farms())
    in_window = df["day"].between(onset - DAYS_BEFORE_ONSET, onset + DAYS_AFTER_ONSET)
    rows = df[df["cow_id"].isin([sick, *healthy]) & in_window].sort_values(["timestamp", "cow_id"]).copy()
    rows["timestamp"] += pd.Timestamp(datetime.now().replace(microsecond=0)) - rows["timestamp"].max()
    return rows, sick


def _replay_one(row: pd.Series) -> dict:
    """Send one milking through the prediction service; returns the log line for the page."""
    payload = to_payload(row) | {"cow_id": f"DEMO-{row['cow_id']}", "farm_id": DEMO_FARM}
    with SessionLocal() as session:
        response, _ = handle_reading(session, PredictRequest(**payload))
    return {
        "time": pd.Timestamp(row["timestamp"]).strftime("%d %b %H:%M"),
        "cow": response.cow_id,
        "risk_score": response.risk_score,
        "risk_level": response.risk_level,
        "alert": response.alert,
        "truth": "infected" if row["true_infected"] else "healthy",
        "reason": response.top_reasons[0].text if response.top_reasons else "",
    }


def _run(interval: float) -> None:
    try:
        reset_demo_data()
        rows, sick = demo_rows()
        STATE.update(total=len(rows), sick_cow=f"DEMO-{sick}")
        for step, (_, row) in enumerate(rows.iterrows(), start=1):
            line = _replay_one(row)
            STATE["step"] = step
            STATE["log"].insert(0, line)
            del STATE["log"][LOG_LENGTH:]
            time.sleep(interval)
    except Exception as error:  # shown on the demo page instead of dying silently
        STATE["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        STATE["running"] = False


def start(interval: float = 2.0) -> bool:
    """Start the replay in the background. Returns False if a demo is already running."""
    with _lock:
        if STATE["running"]:
            return False
        STATE.update(running=True, step=0, total=0, log=[], error=None, sick_cow=None)
    threading.Thread(target=_run, args=(max(0.0, min(interval, MAX_INTERVAL_S)),), daemon=True).start()
    return True


def status() -> dict:
    """Progress of the current/last demo run (GET /api/demo/status)."""
    return {key: STATE[key] for key in STATUS_KEYS}
