"""
Fake in-line milk sensor for the live demo.

Replays a few cows from the SIMULATED CSV to the API's POST /predict, one
milking at a time, and prints the risk the API returns. By default it picks
cows from the held-out TEST farms (never seen by the model):
  * one cow that develops mastitis during the replay window, and
  * two cows that stay healthy.
The window starts ~8 days before the infection so the cow's baseline builds up.
Requests follow docs/API_CONTRACT.md exactly. "SMS!" marks responses with alert=true
(2 consecutive high readings) - the only ones that would trigger an SMS.

Run (API must be running):
  python simulator/send_readings.py
  python simulator/send_readings.py --interval 0.2          # faster
  python simulator/send_readings.py --url http://192.168.1.10:8000
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
QUARTERS = ["lf", "rf", "lr", "rr"]
COLOURS = {"low": "\033[32m", "medium": "\033[33m", "high": "\033[31m"}
RESET = "\033[0m"


def pick_cows(df: pd.DataFrame, farms: list[str], n_healthy: int = 2, seed: int = 7):
    """Return (sick_cow_id, onset_day, [healthy_cow_ids])."""
    pool = df[df["farm_id"].isin(farms)] if farms else df
    onset = pool[pool["true_infected"] == 1].groupby("cow_id")["day"].min()
    onset = onset[(onset >= 15) & (onset <= 70)]          # room for a baseline before, and follow-up after
    # onset = FIRST infected day, so the cow is healthy before it (clean baseline)
    sick = onset.sample(1, random_state=seed)
    sick_id, onset_day = sick.index[0], int(sick.iloc[0])
    never = pool.groupby("cow_id")["true_infected"].max()
    healthy = never[never == 0].sample(n_healthy, random_state=seed).index.tolist()
    return sick_id, onset_day, healthy


def clean(v):
    """NaN -> None (JSON null), like a missing sensor value."""
    return None if (v is None or pd.isna(v)) else v


def to_payload(r: pd.Series) -> dict:
    """CSV row -> /predict request body, exactly as in docs/API_CONTRACT.md."""
    quarters = {q.upper(): clean(r[f"ec_{q}"]) for q in QUARTERS}       # lf -> LF, ...
    present = [v for v in quarters.values() if v is not None]
    return {
        "cow_id": r["cow_id"],
        "farm_id": r["farm_id"],
        "timestamp": pd.Timestamp(r["timestamp"]).isoformat(),
        # A real sensor measures whole-milk EC separately; here: mean of the quarters
        "ec_ms_cm": round(sum(present) / len(present), 2) if present else None,
        "ec_quarters": quarters,
        "milk_temp_c": clean(r["milk_temp_c"]),
        "ph": clean(r["ph"]),
        "milk_yield_l": clean(r["milk_yield_l"]),
        # CMT is recorded on ~1 in 7 milkings; sent so the API stores it as a future label
        "cmt_score": None if pd.isna(r["cmt_score"]) else int(r["cmt_score"]),
        "breed": r["breed"],
        "parity": int(r["parity"]),
        "days_in_milk": int(r["days_in_milk"]),
        "grazing_system": r["grazing_system"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--csv", default=str(ROOT / "data" / "synthetic_milkings.csv"))
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between readings")
    ap.add_argument("--days-before", type=int, default=8)
    ap.add_argument("--days-after", type=int, default=8)
    ap.add_argument("--all-farms", action="store_true", help="pick cows from any farm, not only test farms")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    meta_path = ROOT / "models" / "metadata.json"
    farms = [] if args.all_farms or not meta_path.exists() else json.loads(meta_path.read_text())["test_farms"]
    sick_id, onset, healthy = pick_cows(df, farms)
    first, last = max(0, onset - args.days_before), onset + args.days_after

    cows = [sick_id, *healthy]
    rows = df[df["cow_id"].isin(cows) & df["day"].between(first, last)].sort_values(["timestamp", "cow_id"])
    print(f"Replaying {len(rows)} milkings, days {first}-{last}, to {args.url}/predict")
    print(f"  cow developing mastitis: {sick_id} (infection starts day {onset})")
    print(f"  healthy cows:            {', '.join(healthy)}\n")

    try:
        requests.get(f"{args.url}/health", timeout=5).raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"API not reachable at {args.url} ({e}). Start it with: uvicorn api.main:app")

    for _, r in rows.iterrows():
        res = requests.post(f"{args.url}/predict", json=to_payload(r), timeout=10)
        if res.status_code != 200:
            print(f"  ERROR {res.status_code}: {res.text[:200]}")
            continue
        p = res.json()
        bar = "#" * round(p["risk_score"] * 20)
        truth = "INFECTED" if r["true_infected"] else "healthy"
        reason = p["top_reasons"][0]["text"] if p["top_reasons"] else ""
        colour = COLOURS[p["risk_level"]]
        sms = f"{COLOURS['high']}SMS!{RESET}" if p["alert"] else "    "
        print(f"day {r['day']:2d} {r['milking']}  {r['cow_id']}  {colour}{p['risk_score']:.2f} "
              f"{p['risk_level']:6s}{RESET} |{bar:<20}| {sms} truth: {truth:8s}  {reason[:80]}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
