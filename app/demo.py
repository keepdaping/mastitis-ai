"""
Live demo for /demo: replays the simulator's cows through the SAME code as POST /predict
(app.service.handle_reading) in a background thread, one milking every `interval` seconds.

The demo cows live on their own farm "DEMO" (tags "DEMO-..."), so the seeded history is never
touched. Their timestamps are shifted so the replay ends "now". Each start clears the previous
demo run.
"""
import json
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, select

from api.schemas import PredictRequest
from app.db import SessionLocal
from app.models import Alert, Cow, Farm, Reading, SmsLog
from app.service import handle_reading
from simulator.send_readings import pick_cows, to_payload

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "synthetic_milkings.csv"
DEMO_FARM = "DEMO"

_lock = threading.Lock()
STATE = {"running": False, "step": 0, "total": 0, "log": [], "error": None, "sick_cow": None}


def reset_demo_data():
    """Delete the previous demo run (demo cows, their readings, alerts and SMS log)."""
    with SessionLocal() as s:
        cow_ids = s.execute(select(Cow.id).where(Cow.farm_id == DEMO_FARM)).scalars().all()
        if cow_ids:
            s.execute(delete(SmsLog).where(SmsLog.cow_id.in_(cow_ids)))
            s.execute(delete(Alert).where(Alert.cow_id.in_(cow_ids)))
            s.execute(delete(Reading).where(Reading.cow_id.in_(cow_ids)))
            s.execute(delete(Cow).where(Cow.id.in_(cow_ids)))
        if s.get(Farm, DEMO_FARM) is None:
            s.add(Farm(id=DEMO_FARM, name="Demo farm", district="Live demo",
                       owner_name="Demo farmer", phone="+256700000999"))
        s.commit()


def demo_rows(days_before: int = 4, days_after: int = 6) -> tuple[pd.DataFrame, str]:
    """1 cow that develops mastitis + 2 healthy cows from the held-out test farms."""
    df = pd.read_csv(CSV, parse_dates=["timestamp"])
    meta = ROOT / "models" / "metadata.json"
    farms = json.loads(meta.read_text())["test_farms"] if meta.exists() else []   # farms the model never saw
    sick, onset, healthy = pick_cows(df, farms)
    rows = df[df["cow_id"].isin([sick, *healthy]) & df["day"].between(onset - days_before, onset + days_after)]
    rows = rows.sort_values(["timestamp", "cow_id"]).copy()
    rows["timestamp"] += pd.Timestamp(datetime.now().replace(microsecond=0)) - rows["timestamp"].max()
    return rows, sick


def _run(interval: float):
    try:
        reset_demo_data()
        rows, sick = demo_rows()
        STATE.update(total=len(rows), sick_cow=f"DEMO-{sick}")
        for i, (_, r) in enumerate(rows.iterrows(), start=1):
            payload = to_payload(r) | {"cow_id": f"DEMO-{r['cow_id']}", "farm_id": DEMO_FARM}
            with SessionLocal() as s:
                resp, _ = handle_reading(s, PredictRequest(**payload))
            STATE["step"] = i
            STATE["log"].insert(0, {
                "time": pd.Timestamp(r["timestamp"]).strftime("%d %b %H:%M"), "cow": resp.cow_id,
                "risk_score": resp.risk_score, "risk_level": resp.risk_level, "alert": resp.alert,
                "truth": "infected" if r["true_infected"] else "healthy",
                "reason": resp.top_reasons[0].text if resp.top_reasons else ""})
            del STATE["log"][30:]
            time.sleep(interval)
    except Exception as e:                       # shown on the demo page instead of dying silently
        STATE["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        STATE["running"] = False


def start(interval: float = 2.0) -> bool:
    """Start the replay in the background. Returns False if a demo is already running."""
    with _lock:
        if STATE["running"]:
            return False
        STATE.update(running=True, step=0, total=0, log=[], error=None, sick_cow=None)
    threading.Thread(target=_run, args=(interval,), daemon=True).start()
    return True
