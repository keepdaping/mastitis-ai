"""
Replaying the SIMULATED CSV (app/ layer): turns synthetic milkings into /predict requests.

Shared by the seed script (app/seeding.py), the dashboard live demo (app/demo.py) and the
terminal fake sensor (simulator/send_readings.py), so all three send exactly the same payloads.
"""

import json
from pathlib import Path

import pandas as pd

from app.config import QUARTER_KEYS

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "synthetic_milkings.csv"
METADATA_PATH = ROOT / "models" / "metadata.json"
ONSET_DAY_RANGE = (15, 70)  # demo cow: room for a baseline before the infection and follow-up after
DEMO_SEED = 7


def held_out_farms() -> list[str]:
    """The test farms the model never saw in training ([] if no model has been trained)."""
    return json.loads(METADATA_PATH.read_text())["test_farms"] if METADATA_PATH.exists() else []


def pick_cows(
    df: pd.DataFrame, farms: list[str], n_healthy: int = 2, seed: int = DEMO_SEED
) -> tuple[str, int, list[str]]:
    """Cows for a demo: (cow that develops mastitis, its first infected day, healthy cows)."""
    pool = df[df["farm_id"].isin(farms)] if farms else df
    onset = pool[pool["true_infected"] == 1].groupby("cow_id")["day"].min()
    # onset = FIRST infected day, so the cow is healthy before it (clean baseline)
    onset = onset[(onset >= ONSET_DAY_RANGE[0]) & (onset <= ONSET_DAY_RANGE[1])]
    sick = onset.sample(1, random_state=seed)
    sick_id, onset_day = sick.index[0], int(sick.iloc[0])
    ever_infected = pool.groupby("cow_id")["true_infected"].max()
    healthy = ever_infected[ever_infected == 0].sample(n_healthy, random_state=seed).index.tolist()
    return sick_id, onset_day, healthy


def _clean(value: object) -> object:
    """NaN -> None (JSON null), like a missing sensor value."""
    return None if (value is None or pd.isna(value)) else value


def to_payload(row: pd.Series) -> dict:
    """CSV row -> /predict request body, exactly as in docs/API_CONTRACT.md."""
    quarters = {key: _clean(row[f"ec_{key.lower()}"]) for key in QUARTER_KEYS}  # ec_lf -> LF, ...
    present = [value for value in quarters.values() if value is not None]
    return {
        "cow_id": row["cow_id"],
        "farm_id": row["farm_id"],
        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
        # A real sensor measures whole-milk EC separately; here: mean of the quarters
        "ec_ms_cm": round(sum(present) / len(present), 2) if present else None,
        "ec_quarters": quarters,
        "milk_temp_c": _clean(row["milk_temp_c"]),
        "ph": _clean(row["ph"]),
        "milk_yield_l": _clean(row["milk_yield_l"]),
        # CMT is recorded on ~1 in 7 milkings; sent so the API stores it as a future label
        "cmt_score": None if pd.isna(row["cmt_score"]) else int(row["cmt_score"]),
        "breed": row["breed"],
        "parity": int(row["parity"]),
        "days_in_milk": int(row["days_in_milk"]),
        "grazing_system": row["grazing_system"],
    }
