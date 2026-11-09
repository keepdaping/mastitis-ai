"""
Seed the dashboard database with realistic (SIMULATED) history.

Takes 3 farms x 10 cows from data/synthetic_milkings.csv and replays their last 30 days of
milkings through the SAME code as POST /predict (app.service.handle_reading), so every
reading, risk score, alert episode and SMS-log entry is produced exactly as in live use.
Timestamps are shifted so the history ends today.

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

import pandas as pd  # noqa: E402

CSV = ROOT / "data" / "synthetic_milkings.csv"

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


def seed(farms: dict = FARMS, cows_per_farm: int = 10, days: int = 30, quiet: bool = False) -> dict:
    """Add farms, cows and replayed readings to the database. Returns counts."""
    from api.schemas import PredictRequest
    from app.db import SessionLocal, init_db
    from app.models import Alert, Farm, Reading, SmsLog
    from app.service import handle_reading
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
    rows = rows[rows["timestamp"] <= pd.Timestamp(datetime.now())]

    start = time.time()
    for i, (_, r) in enumerate(rows.iterrows(), start=1):
        with SessionLocal() as s:
            handle_reading(s, PredictRequest(**to_payload(r)))
        if not quiet and (i % 200 == 0 or i == len(rows)):
            print(f"  {i}/{len(rows)} readings ({time.time() - start:.0f} s)", flush=True)

    with SessionLocal() as s:
        counts = {"farms": len(farms), "cows": len(tags), "readings": s.query(Reading).count(),
                  "alerts": s.query(Alert).count(), "sms_sent": s.query(SmsLog).filter_by(status="sent").count()}
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
