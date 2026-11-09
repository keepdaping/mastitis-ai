"""
Seeding the dashboard database with realistic (SIMULATED) history (app/ layer).

Takes 3 farms x 10 cows from data/synthetic_milkings.csv and replays their last 30 days of
milkings through the SAME code as POST /predict (app.service.handle_reading), so every
reading, risk score, alert episode and SMS-log entry is produced exactly as in live use.
Timestamps are shifted so the history ends today. The command line is scripts/seed_db.py.

Farmer behaviour is simulated too: every alert episode that starts more than 3 days before
today gets a fate, decided when it opens and applied 12-36 h later:
  ~60% treated (by the vet), ~15% marked false alarm, the rest left open.
A treated cow responds to treatment: 5-10 days after the treatment her remaining infected
readings are replaced by healthy readings (her own healthy levels + sensor noise).
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.schemas import PredictRequest
from app.alerts import close_alert
from app.db import SessionLocal, init_db
from app.models import Alert, Farm, Reading, SmsLog
from app.replay import CSV_PATH, to_payload
from app.service import handle_reading

QUARTERS = ["lf", "rf", "lr", "rr"]
P_TREATED, P_FALSE_ALARM = 0.60, 0.15  # the rest of the older episodes stay open
FATE_MIN_AGE = timedelta(days=3)  # episodes younger than this are left open
FARMER_DELAY_H = (12, 36)  # hours from alert to the farmer's action
RECOVERY_DAYS = (5, 10)  # days from treatment to healthy readings
NOTES = {
    "treated": "Treated by vet, intramammary antibiotic",
    "false_alarm": "Farmer: CMT negative at check, no treatment",
}
SENSOR_NOISE = {"ec": 0.20, "ph": 0.03, "milk_temp_c": 0.10}  # added to a cured cow's healthy levels
CURED_CMT_PROBS = [0.85, 0.13, 0.02]  # a cured cow tests (mostly) negative: score 0 / 1 / 2
COW_SAMPLE_SEED = 1
FATE_SEED = 7
PROGRESS_EVERY = 200

# Farm details are fictional (SIMULATED). Phone numbers are Africa's Talking sandbox-style numbers.
FARMS = {
    "F02": {"name": "Kayunga", "district": "Kayunga", "owner_name": "Nakato S. (fictional)", "phone": "+256700000002"},
    "F09": {
        "name": "Mukono Hill",
        "district": "Mukono",
        "owner_name": "Okello J. (fictional)",
        "phone": "+256700000009",
    },
    "F04": {
        "name": "Wakiso Valley",
        "district": "Wakiso",
        "owner_name": "Namubiru R. (fictional)",
        "phone": "+256700000004",
    },
}


def choose_cows(df: pd.DataFrame, farm: str, n: int, window_days: list[int], seed: int = COW_SAMPLE_SEED) -> list[str]:
    """n cows from a farm: about half with an infection inside the replay window, half healthy,
    so the dashboard shows both."""
    farm_df = df[(df["farm_id"] == farm) & df["day"].isin(window_days)]
    infected = farm_df.groupby("cow_id")["true_infected"].max()
    sick = infected[infected == 1].sample(frac=1, random_state=seed).index[: n // 2].tolist()
    healthy = infected[infected == 0].sample(frac=1, random_state=seed).index[: n - len(sick)].tolist()
    return sorted(sick + healthy)


def healthy_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Each cow's typical HEALTHY readings (median over milkings without any infected quarter)."""
    healthy = df[df["true_infected_quarters"].isna()]
    cols = [f"ec_{quarter}" for quarter in QUARTERS] + ["ph", "milk_temp_c"]
    return healthy.groupby("cow_id")[cols].median()


def recover_after_treatment(
    rows: pd.DataFrame,
    start_pos: int,
    cow: str,
    recovery_time: pd.Timestamp,
    levels: pd.DataFrame,
    rng: np.random.Generator,
) -> int:
    """Simulate a cure: from `recovery_time` until this infection would have ended anyway, replace the
    cow's readings with healthy ones (her own healthy level + sensor noise). Returns rows changed."""
    later = rows.iloc[start_pos:]
    later = later[later["cow_id"] == cow]
    infected = later["true_infected_quarters"].notna()
    # The current infection = the run of infected readings from now; it ends at the first healthy one
    run = later.loc[: infected.idxmin()] if not infected.all() else later
    target = run.index[(run["timestamp"] >= recovery_time) & run["true_infected_quarters"].notna()]
    for idx in target:
        for col in levels.columns:
            if pd.notna(rows.at[idx, col]):  # keep missing sensor values missing
                noise_sd = SENSOR_NOISE["ec"] if col.startswith("ec_") else SENSOR_NOISE[col]
                rows.at[idx, col] = round(float(levels.at[cow, col] + rng.normal(0, noise_sd)), 2)
        if pd.notna(rows.at[idx, "cmt_score"]):
            rows.at[idx, "cmt_score"] = int(rng.choice([0, 1, 2], p=CURED_CMT_PROBS))
        rows.at[idx, "true_infected"] = 0
        rows.at[idx, "true_infected_quarters"] = np.nan
    return len(target)


def _add_farms(farms: dict) -> None:
    with SessionLocal() as session:  # farms with owner + phone (needed for SMS)
        for farm_id, info in farms.items():
            session.merge(Farm(id=farm_id, **info))
        session.commit()


def replay_rows(df: pd.DataFrame, tags: list[str], window: list[int]) -> pd.DataFrame:
    """The chosen cows' milkings in time order, shifted (whole days) so the last day is today;
    milkings that would lie in the future are dropped."""
    rows = df[df["cow_id"].isin(tags) & df["day"].isin(window)].sort_values(["timestamp", "cow_id"]).copy()
    shift = pd.Timestamp(datetime.now().date()) - rows["timestamp"].max().normalize()
    rows["timestamp"] += shift
    rows = rows[rows["timestamp"] <= pd.Timestamp(datetime.now())].reset_index(drop=True)
    rows["true_infected_quarters"] = rows["true_infected_quarters"].astype(object)
    return rows


def _apply_due_actions(session: Session, pending: list, now: pd.Timestamp) -> None:
    """Carry out the farmer actions (treated / false alarm) that are due before this milking."""
    for action in sorted(item for item in pending if item[0] <= now):
        pending.remove(action)
        when, alert_id, status = action
        alert = session.get(Alert, alert_id)
        if alert is not None and alert.is_active:  # it may have auto-resolved meanwhile
            close_alert(session, alert, status, NOTES[status], when=when.to_pydatetime())
            session.commit()


class FarmerSimulation:
    """Decides, for each new alert episode, what the farmer will do about it - and when."""

    def __init__(self, rows: pd.DataFrame, levels: pd.DataFrame):
        self.rows, self.levels = rows, levels
        self.rng = np.random.default_rng(FATE_SEED)
        self.fate_before = (rows["timestamp"].max() - FATE_MIN_AGE).to_pydatetime()
        self.pending: list = []  # farmer actions: (time, alert_id, status)
        self.fates = {"treated": 0, "false_alarm": 0, "left_open": 0}
        self.made_healthy = 0

    def react(self, new_alert: Alert, row: pd.Series, position: int) -> None:
        """Draw the fate of a new episode (only episodes older than FATE_MIN_AGE get one)."""
        if new_alert.created_at >= self.fate_before:
            return
        draw = self.rng.random()
        if draw >= P_TREATED + P_FALSE_ALARM:
            self.fates["left_open"] += 1
            return
        status = "treated" if draw < P_TREATED else "false_alarm"
        when = row["timestamp"] + pd.Timedelta(hours=float(self.rng.uniform(*FARMER_DELAY_H)))
        self.pending.append((when, new_alert.id, status))
        self.fates[status] += 1
        if status == "treated":
            recovery = when + pd.Timedelta(days=float(self.rng.uniform(*RECOVERY_DAYS)))
            self.made_healthy += recover_after_treatment(
                self.rows, position + 1, row["cow_id"], recovery, self.levels, self.rng
            )


def _summary(farms: dict, tags: list[str], farmer: FarmerSimulation) -> dict:
    with SessionLocal() as session:
        return {
            "farms": len(farms),
            "cows": len(tags),
            "readings": session.query(Reading).count(),
            "alerts": session.query(Alert).count(),
            "sms_sent": session.query(SmsLog).filter_by(status="sent").count(),
            "fates_drawn": farmer.fates,
            "readings_made_healthy": farmer.made_healthy,
            "alert_status": dict(session.query(Alert.status, func.count(Alert.id)).group_by(Alert.status).all()),
        }


def seed(farms: dict = FARMS, cows_per_farm: int = 10, days: int = 30, quiet: bool = False) -> dict:
    """Add farms, cows and replayed readings to the database. Returns counts."""
    init_db()
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    last_day = int(df["day"].max())
    window = list(range(last_day - days + 1, last_day + 1))
    _add_farms(farms)
    tags = [cow for farm in farms for cow in choose_cows(df, farm, cows_per_farm, window)]
    rows = replay_rows(df, tags, window)
    farmer = FarmerSimulation(rows, healthy_levels(df))

    start = time.time()
    for position in range(len(rows)):
        row = rows.iloc[position]  # re-read every time: a treatment may have changed it
        with SessionLocal() as session:
            _apply_due_actions(session, farmer.pending, row["timestamp"])
            _, reading = handle_reading(session, PredictRequest(**to_payload(row)))
            new_alert = session.execute(select(Alert).where(Alert.reading_id == reading.id)).scalar_one_or_none()
            if new_alert is not None:
                farmer.react(new_alert, row, position)
        if not quiet and ((position + 1) % PROGRESS_EVERY == 0 or position + 1 == len(rows)):
            print(f"  {position + 1}/{len(rows)} readings ({time.time() - start:.0f} s)", flush=True)
    return _summary(farms, tags, farmer)
