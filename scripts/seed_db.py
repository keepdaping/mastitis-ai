"""
Seed the dashboard database with realistic (SIMULATED) history.

Takes 3 farms x 10 cows from data/synthetic_milkings.csv and replays their last 30 days of
milkings through the SAME code as POST /predict (app.service.handle_reading), so every
reading, risk score, alert episode and SMS-log entry is produced exactly as in live use.
Timestamps are shifted so the history ends today.

Farmer behaviour is simulated too: every alert episode that starts more than 3 days before
today gets a fate, decided when it opens and applied 12-36 h later:
  ~60% treated (by the vet), ~15% marked false alarm, the rest left open.
A treated cow responds to treatment: 5-10 days after the treatment her remaining infected
readings are replaced by healthy readings (her own healthy levels + sensor noise).

    python scripts/seed_db.py            # first run (refuses if the database already has data)
    python scripts/seed_db.py --reset    # wipe data/app.db and seed again

SMS during seeding always uses the console provider (nothing is really sent), unless you
pass --send-sms.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CSV = ROOT / "data" / "synthetic_milkings.csv"
QUARTERS = ["lf", "rf", "lr", "rr"]
P_TREATED, P_FALSE_ALARM = 0.60, 0.15     # the rest of the older episodes stay open
FATE_MIN_AGE = timedelta(days=3)          # episodes younger than this are left open
FARMER_DELAY_H = (12, 36)                 # hours from alert to the farmer's action
RECOVERY_DAYS = (5, 10)                   # days from treatment to healthy readings
NOTES = {"treated": "Treated by vet, intramammary antibiotic",
         "false_alarm": "Farmer: CMT negative at check, no treatment"}

# Farm details are fictional (SIMULATED). Phone numbers are Africa's Talking sandbox-style numbers.
FARMS = {
    "F02": {"name": "Kayunga", "district": "Kayunga", "owner_name": "Nakato S. (fictional)", "phone": "+256700000002"},
    "F09": {"name": "Mukono Hill", "district": "Mukono", "owner_name": "Okello J. (fictional)", "phone": "+256700000009"},
    "F04": {"name": "Wakiso Valley", "district": "Wakiso", "owner_name": "Namubiru R. (fictional)", "phone": "+256700000004"},
}


def choose_cows(df: pd.DataFrame, farm: str, n: int, window_days: list[int], seed: int = 1) -> list[str]:
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
    cols = [f"ec_{q}" for q in QUARTERS] + ["ph", "milk_temp_c"]
    return healthy.groupby("cow_id")[cols].median()


def recover_after_treatment(rows: pd.DataFrame, start_pos: int, cow: str, recovery_time: pd.Timestamp,
                            levels: pd.DataFrame, rng: np.random.Generator) -> int:
    """Simulate a cure: from `recovery_time` until this infection would have ended anyway, replace the
    cow's readings with healthy ones (her own healthy level + sensor noise). Returns rows changed."""
    later = rows.iloc[start_pos:]
    later = later[later["cow_id"] == cow]
    infected = later["true_infected_quarters"].notna()
    # The current infection = the run of infected readings from now; it ends at the first healthy one
    run = later.loc[:infected.idxmin()] if not infected.all() else later
    target = run.index[(run["timestamp"] >= recovery_time) & run["true_infected_quarters"].notna()]
    noise = {"ph": 0.03, "milk_temp_c": 0.10}
    for idx in target:
        for col in levels.columns:
            if pd.notna(rows.at[idx, col]):                  # keep missing sensor values missing
                sd = 0.20 if col.startswith("ec_") else noise[col]
                rows.at[idx, col] = round(float(levels.at[cow, col] + rng.normal(0, sd)), 2)
        if pd.notna(rows.at[idx, "cmt_score"]):              # a cured cow tests (mostly) negative
            rows.at[idx, "cmt_score"] = int(rng.choice([0, 1, 2], p=[0.85, 0.13, 0.02]))
        rows.at[idx, "true_infected"] = 0
        rows.at[idx, "true_infected_quarters"] = np.nan
    return len(target)


def seed(farms: dict = FARMS, cows_per_farm: int = 10, days: int = 30, quiet: bool = False) -> dict:
    """Add farms, cows and replayed readings to the database. Returns counts."""
    from api.schemas import PredictRequest
    from app.db import SessionLocal, init_db
    from sqlalchemy import func, select

    from app.models import Alert, Farm, Reading, SmsLog
    from app.service import close_alert, handle_reading
    from simulator.send_readings import to_payload

    init_db()
    df = pd.read_csv(CSV, parse_dates=["timestamp"])
    window = list(range(int(df["day"].max()) - days + 1, int(df["day"].max()) + 1))

    with SessionLocal() as s:                       # farms with owner + phone (needed for SMS)
        for farm_id, info in farms.items():
            s.merge(Farm(id=farm_id, **info))
        s.commit()

    tags = [c for f in farms for c in choose_cows(df, f, cows_per_farm, window)]
    rows = df[df["cow_id"].isin(tags) & df["day"].isin(window)].sort_values(["timestamp", "cow_id"]).copy()

    # Shift whole days so the last simulated day is today; drop milkings that would be in the future
    shift = pd.Timestamp(datetime.now().date()) - rows["timestamp"].max().normalize()
    rows["timestamp"] += shift
    rows = rows[rows["timestamp"] <= pd.Timestamp(datetime.now())].reset_index(drop=True)
    rows["true_infected_quarters"] = rows["true_infected_quarters"].astype(object)

    rng = np.random.default_rng(7)
    levels = healthy_levels(df)
    fate_before = (rows["timestamp"].max() - FATE_MIN_AGE).to_pydatetime()
    pending = []                      # farmer actions: (time, alert_id, status)
    fates = {"treated": 0, "false_alarm": 0, "left_open": 0}
    made_healthy = 0
    start = time.time()
    for pos in range(len(rows)):
        r = rows.iloc[pos]                                   # re-read: a treatment may have changed it
        with SessionLocal() as s:
            # 1) farmer actions that are due before this milking
            for action in sorted(a for a in pending if a[0] <= r["timestamp"]):
                pending.remove(action)
                alert = s.get(Alert, action[1])
                if alert is not None and alert.is_active:    # it may have auto-resolved meanwhile
                    close_alert(s, alert, action[2], NOTES[action[2]], when=action[0].to_pydatetime())
                    s.commit()
            # 2) the milking itself, through the same code as POST /predict
            _, reading = handle_reading(s, PredictRequest(**to_payload(r)))
            # 3) a new episode? decide now what the farmer will do about it
            new = s.execute(select(Alert).where(Alert.reading_id == reading.id)).scalar_one_or_none()
            if new is not None and new.created_at < fate_before:
                u = rng.random()
                if u < P_TREATED + P_FALSE_ALARM:
                    status = "treated" if u < P_TREATED else "false_alarm"
                    when = r["timestamp"] + pd.Timedelta(hours=float(rng.uniform(*FARMER_DELAY_H)))
                    pending.append((when, new.id, status))
                    fates[status] += 1
                    if status == "treated":
                        recovery = when + pd.Timedelta(days=float(rng.uniform(*RECOVERY_DAYS)))
                        made_healthy += recover_after_treatment(rows, pos + 1, r["cow_id"], recovery, levels, rng)
                else:
                    fates["left_open"] += 1
        if not quiet and ((pos + 1) % 200 == 0 or pos + 1 == len(rows)):
            print(f"  {pos + 1}/{len(rows)} readings ({time.time() - start:.0f} s)", flush=True)

    with SessionLocal() as s:
        counts = {"farms": len(farms), "cows": len(tags), "readings": s.query(Reading).count(),
                  "alerts": s.query(Alert).count(), "sms_sent": s.query(SmsLog).filter_by(status="sent").count(),
                  "fates_drawn": fates, "readings_made_healthy": made_healthy,
                  "alert_status": dict(s.query(Alert.status, func.count(Alert.id)).group_by(Alert.status).all())}
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset", action="store_true", help="delete ALL data in the database first")
    ap.add_argument("--cows-per-farm", type=int, default=10)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--send-sms", action="store_true", help="use SMS_PROVIDER from .env instead of console")
    args = ap.parse_args()
    if not args.send_sms:
        os.environ["SMS_PROVIDER"] = "console"

    from app.db import DB_PATH, SessionLocal, init_db, reset_db
    from app.models import Cow
    init_db()
    with SessionLocal() as s:
        has_data = s.query(Cow).count() > 0
    if has_data and not args.reset:
        sys.exit(f"{DB_PATH} already has data. Use --reset to wipe it and seed again.")
    if args.reset:
        reset_db()
        print(f"Database reset: {DB_PATH}")

    print(f"Seeding {len(FARMS)} farms x {args.cows_per_farm} cows, last {args.days} days (SIMULATED data)...")
    counts = seed(cows_per_farm=args.cows_per_farm, days=args.days)
    print("Done:", ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
