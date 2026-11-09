"""
Command line for seeding the dashboard database with SIMULATED history (the logic is app/seeding.py).

Replays 3 farms x 10 cows x 30 days from data/synthetic_milkings.csv through the same code as
POST /predict, including simulated farmer actions (treated / false alarm) and recovery.

    python scripts/seed_db.py            # first run (refuses if the database already has data)
    python scripts/seed_db.py --reset    # wipe data/app.db and seed again

SMS during seeding always uses the console provider (nothing is really sent), unless you
pass --send-sms.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import DB_PATH, SessionLocal, init_db, reset_db  # noqa: E402
from app.models import Cow  # noqa: E402
from app.seeding import FARMS, seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="delete ALL data in the database first")
    parser.add_argument("--cows-per-farm", type=int, default=10)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--send-sms", action="store_true", help="use SMS_PROVIDER from .env instead of console")
    return parser.parse_args()


def main() -> None:
    """Seed the database (see the module docstring)."""
    args = parse_args()
    if not args.send_sms:
        os.environ["SMS_PROVIDER"] = "console"  # read by app/notify.py at send time

    init_db()
    with SessionLocal() as session:
        has_data = session.query(Cow).count() > 0
    if has_data and not args.reset:
        sys.exit(f"{DB_PATH} already has data. Use --reset to wipe it and seed again.")
    if args.reset:
        reset_db()
        print(f"Database reset: {DB_PATH}")

    print(f"Seeding {len(FARMS)} farms x {args.cows_per_farm} cows, last {args.days} days (SIMULATED data)...")
    counts = seed(cows_per_farm=args.cows_per_farm, days=args.days)
    print("Done:", ", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
