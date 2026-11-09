"""
Fake in-line milk sensor for the terminal demo: an external client of the API.

Replays a few cows from the SIMULATED CSV to POST /predict, one milking at a time, and prints
the risk the API returns. By default it picks cows from the held-out TEST farms (never seen
by the model):
  * one cow that develops mastitis during the replay window, and
  * two cows that stay healthy.
The window starts ~8 days before the infection so the cow's baseline builds up.
Requests follow docs/API_CONTRACT.md exactly (built by app/replay.py). "SMS!" marks responses
with alert=true (2 consecutive high readings) - the only ones that would trigger an SMS.

Run (API must be running):
  python simulator/send_readings.py
  python simulator/send_readings.py --interval 0.2          # faster
  python simulator/send_readings.py --url http://192.168.1.10:8000
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.replay import CSV_PATH, held_out_farms, pick_cows, to_payload  # noqa: E402

COLOURS = {"low": "\033[32m", "medium": "\033[33m", "high": "\033[31m"}
RESET = "\033[0m"
BAR_WIDTH = 20
REQUEST_TIMEOUT_S = 10
HEALTH_TIMEOUT_S = 5
HTTP_OK = 200


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between readings")
    parser.add_argument("--days-before", type=int, default=8)
    parser.add_argument("--days-after", type=int, default=8)
    parser.add_argument("--all-farms", action="store_true", help="pick cows from any farm, not only test farms")
    return parser.parse_args()


def format_line(row: pd.Series, result: dict) -> str:
    """One coloured terminal line for an API response."""
    bar = "#" * round(result["risk_score"] * BAR_WIDTH)
    truth = "INFECTED" if row["true_infected"] else "healthy"
    reason = result["top_reasons"][0]["text"] if result["top_reasons"] else ""
    colour = COLOURS[result["risk_level"]]
    sms = f"{COLOURS['high']}SMS!{RESET}" if result["alert"] else "    "
    return (
        f"day {row['day']:2d} {row['milking']}  {row['cow_id']}  {colour}{result['risk_score']:.2f} "
        f"{result['risk_level']:6s}{RESET} |{bar:<{BAR_WIDTH}}| {sms} truth: {truth:8s}  {reason[:80]}"
    )


def main() -> None:
    """Replay the demo cows to the API and print each response."""
    args = parse_args()
    df = pd.read_csv(args.csv)
    sick_id, onset, healthy = pick_cows(df, [] if args.all_farms else held_out_farms())
    first, last = max(0, onset - args.days_before), onset + args.days_after
    rows = df[df["cow_id"].isin([sick_id, *healthy]) & df["day"].between(first, last)]
    rows = rows.sort_values(["timestamp", "cow_id"])
    print(f"Replaying {len(rows)} milkings, days {first}-{last}, to {args.url}/predict")
    print(f"  cow developing mastitis: {sick_id} (infection starts day {onset})")
    print(f"  healthy cows:            {', '.join(healthy)}\n")

    try:
        requests.get(f"{args.url}/health", timeout=HEALTH_TIMEOUT_S).raise_for_status()
    except requests.RequestException as error:
        raise SystemExit(f"API not reachable at {args.url} ({error}). Start it with: uvicorn api.main:app") from error

    for _, row in rows.iterrows():
        response = requests.post(f"{args.url}/predict", json=to_payload(row), timeout=REQUEST_TIMEOUT_S)
        if response.status_code != HTTP_OK:
            print(f"  ERROR {response.status_code}: {response.text[:200]}")
            continue
        print(format_line(row, response.json()))
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
