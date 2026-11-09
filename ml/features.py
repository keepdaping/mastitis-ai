"""
Feature engineering - the ONE place where raw milking readings become model inputs.

The same build_features() is used by:
  * ml/train.py   (on the whole synthetic dataset)
  * api/main.py   (on a cow's stored history + the new reading)
so training and serving can never drift apart.

Input columns (one row per milking):
  cow_id, timestamp, breed, parity, days_in_milk, grazing_system,
  ec_lf, ec_rf, ec_lr, ec_rr, ph, milk_temp_c, milk_yield_l
Optional: ec_ms_cm (whole-milk EC), used only when all four quarter readings are missing.
Missing values (NaN) are allowed anywhere in the sensor columns.
"""
import warnings

import numpy as np
import pandas as pd

QUARTERS = ["lf", "rf", "lr", "rr"]
EC_COLS = [f"ec_{q}" for q in QUARTERS]
BREEDS = ["exotic", "local"]
GRAZING = ["zero", "tethered", "open"]
BASELINE_WINDOW = "7D"          # rolling window for the cow's own "normal"
YIELD_WINDOW = "3D"

# Final feature list, in the exact order the model expects
FEATURES = [
    "ec_mean", "ec_max", "ec_quarter_diff", "ec_quarter_ratio",
    "ec_vs_baseline", "yield_vs_baseline", "temp_vs_baseline",
    "ph", "milk_temp_c", "milk_yield_l", "yield_change_3d",
    "parity", "days_in_milk",
    "breed_exotic", "breed_local",
    "grazing_zero", "grazing_tethered", "grazing_open",
]

# Fallback herd values, used when a cow has no history and no reference was given.
# (train.py computes real herd medians from the training farms and saves them with the model.)
DEFAULT_REFERENCE = {
    "baseline": {   # per breed: typical healthy ec_mean / temperature / yield per milking
        "exotic": {"ec_mean": 5.25, "milk_temp_c": 37.5, "milk_yield_l": 5.5},
        "local": {"ec_mean": 5.25, "milk_temp_c": 37.5, "milk_yield_l": 2.0},
    },
    "raw": {"ph": 6.7, "milk_temp_c": 37.5, "milk_yield_l": 4.5, "parity": 2, "days_in_milk": 150},
}


def _quarter_features(ec: pd.DataFrame) -> pd.DataFrame:
    """EC summary across the 4 quarters, robust to missing quarters.

    An infected quarter stands out from the other three, so we compare the
    highest quarter against the mean of the rest (difference and ratio).
    """
    vals = ec.to_numpy(dtype=float)
    n_ok = np.sum(~np.isnan(vals), axis=1)
    with warnings.catch_warnings():         # rows with all quarters missing -> NaN (handled later)
        warnings.simplefilter("ignore", RuntimeWarning)
        ec_mean = np.nanmean(vals, axis=1)
        ec_max = np.nanmax(vals, axis=1)
        others_mean = (np.nansum(vals, axis=1) - ec_max) / (n_ok - 1)
    enough = n_ok >= 2                    # need >= 2 quarters to compare
    diff = np.where(enough, ec_max - others_mean, 0.0)
    ratio = np.where(enough, ec_max / others_mean, 1.0)
    return pd.DataFrame({"ec_mean": ec_mean, "ec_max": ec_max,
                         "ec_quarter_diff": diff, "ec_quarter_ratio": ratio}, index=ec.index)


def _rolling_prev_mean(df: pd.DataFrame, col: str, window: str) -> pd.Series:
    """Per-cow mean of `col` over the previous `window`, EXCLUDING the current milking."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("cow_id", sort=False):
        s = g.set_index("timestamp")[col]
        out.loc[g.index] = s.rolling(window, closed="left", min_periods=1).mean().to_numpy()
    return out


def _pct_change(current: pd.Series, baseline: pd.Series) -> pd.Series:
    return 100.0 * (current - baseline) / baseline


def build_features(df: pd.DataFrame, reference: dict | None = None) -> pd.DataFrame:
    """Turn raw milking rows into model features (same row order and index as `df`).

    reference: herd medians from training (see compute_reference). Used to
               fill gaps and as the baseline for cows without history.
    """
    ref = reference or DEFAULT_REFERENCE
    df = df.reset_index(drop=True)          # simple 0..n-1 index; output keeps this order
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["cow_id", "timestamp"], kind="stable")

    # 1) Fill single-value gaps in ph/temp/yield with the cow's previous reading
    for col in ["ph", "milk_temp_c", "milk_yield_l"]:
        df[col] = df.groupby("cow_id", sort=False)[col].ffill()

    feats = _quarter_features(df[EC_COLS])
    # All 4 quarters missing but a whole-milk EC reading exists (optional column, API only)
    if "ec_ms_cm" in df:
        feats["ec_mean"] = feats["ec_mean"].fillna(pd.to_numeric(df["ec_ms_cm"], errors="coerce"))
    df["ec_mean"] = feats["ec_mean"]

    # 2) Baselines: the cow's own rolling 7-day mean (previous milkings only)
    for col in ["ec_mean", "milk_temp_c", "milk_yield_l"]:
        base = _rolling_prev_mean(df, col, BASELINE_WINDOW)
        # No history -> fall back to the herd median for that breed
        herd = df["breed"].map(lambda b: ref["baseline"].get(b, ref["baseline"]["exotic"])[col])
        df[f"_base_{col}"] = base.fillna(herd)

    feats["ec_vs_baseline"] = _pct_change(df["ec_mean"], df["_base_ec_mean"])
    feats["yield_vs_baseline"] = _pct_change(df["milk_yield_l"], df["_base_milk_yield_l"])
    feats["temp_vs_baseline"] = _pct_change(df["milk_temp_c"], df["_base_milk_temp_c"])

    # 3) Short-term yield trend: today vs the previous 3 days (0 if no history)
    prev3 = _rolling_prev_mean(df, "milk_yield_l", YIELD_WINDOW)
    feats["yield_change_3d"] = _pct_change(df["milk_yield_l"], prev3).fillna(0.0)

    # 4) Raw readings and cow attributes
    for col in ["ph", "milk_temp_c", "milk_yield_l", "parity", "days_in_milk"]:
        feats[col] = pd.to_numeric(df[col], errors="coerce")
    for b in BREEDS:
        feats[f"breed_{b}"] = (df["breed"] == b).astype(int)
    for g in GRAZING:
        feats[f"grazing_{g}"] = (df["grazing_system"] == g).astype(int)

    # 5) Anything still missing -> herd median (or a neutral value)
    for col, val in ref["raw"].items():
        feats[col] = feats[col].fillna(val)
    feats["ec_mean"] = feats["ec_mean"].fillna(df["_base_ec_mean"])
    feats["ec_max"] = feats["ec_max"].fillna(feats["ec_mean"])
    feats = feats.fillna({"ec_vs_baseline": 0.0, "yield_vs_baseline": 0.0, "temp_vs_baseline": 0.0})
    feats = feats.replace([np.inf, -np.inf], 0.0)

    return feats[FEATURES].sort_index()     # back to the caller's original row order


def compute_reference(df: pd.DataFrame) -> dict:
    """Herd medians from (training) data, saved with the model for use at serving time."""
    d = df.copy()
    d["ec_mean"] = d[EC_COLS].mean(axis=1)
    baseline = {}
    for b in BREEDS:
        sub = d[d["breed"] == b]
        baseline[b] = {c: float(sub[c].median()) for c in ["ec_mean", "milk_temp_c", "milk_yield_l"]}
    raw = {c: float(d[c].median()) for c in ["ph", "milk_temp_c", "milk_yield_l", "parity", "days_in_milk"]}
    return {"baseline": baseline, "raw": raw}


def suspect_quarter(row: dict) -> str | None:
    """Quarter with the highest EC (where an infection is most likely), or None."""
    vals = {q: row.get(f"ec_{q}") for q in QUARTERS}
    vals = {q: v for q, v in vals.items() if v is not None and not pd.isna(v)}
    return max(vals, key=vals.get) if vals else None
