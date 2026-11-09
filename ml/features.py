"""
Feature engineering - the ONE place where raw milking readings become model inputs.

The same build_features() is used by:
  * ml/train.py      (on the whole synthetic dataset)
  * app/predictor.py (on a cow's stored history + the new reading, for POST /predict)
so training and serving can never drift apart.

Input columns (one row per milking):
  cow_id, timestamp, breed, parity, days_in_milk, grazing_system,
  ec_lf, ec_rf, ec_lr, ec_rr, ph, milk_temp_c, milk_yield_l
Optional: ec_ms_cm (whole-milk EC), used only when all four quarter readings are missing.
Optional: baseline_exclude (bool) - readings that must NOT count towards the cow's "normal"
          (high-risk readings and readings during an open alert episode; see ml/episodes.py).
Missing values (NaN) are allowed anywhere in the sensor columns.

CLEAN BASELINE: ec/yield/temp "vs baseline" compare the current milking with the cow's own
normal = mean of her previous CLEAN readings (never the current one, never future ones):
  last 7 days if it has >= 3 clean readings, else last 21 days if >= 3, else the herd median.
This stops a long infection from slowly becoming the cow's "normal".
"""

import warnings

import numpy as np
import pandas as pd

QUARTERS = ["lf", "rf", "lr", "rr"]
EC_COLS = [f"ec_{quarter}" for quarter in QUARTERS]
BREEDS = ["exotic", "local"]
GRAZING = ["zero", "tethered", "open"]
BASELINE_WINDOW = "7D"  # rolling window for the cow's own "normal"
BASELINE_WINDOW_LONG = "21D"  # used when the last 7 days have too few clean readings
MIN_CLEAN_READINGS = 3  # fewer than this -> extend the window, then herd median
YIELD_WINDOW = "3D"
MIN_QUARTERS_TO_COMPARE = 2  # need at least 2 quarters to compare one against the others
FILLED_SENSORS = ["ph", "milk_temp_c", "milk_yield_l"]  # gaps filled from the cow's previous reading
BASELINE_SENSORS = ["ec_mean", "milk_temp_c", "milk_yield_l"]

# Final feature list, in the exact order the model expects
FEATURES = [
    "ec_mean", "ec_max", "ec_quarter_diff", "ec_quarter_ratio",
    "ec_vs_baseline", "yield_vs_baseline", "temp_vs_baseline",
    "ph", "milk_temp_c", "milk_yield_l", "yield_change_3d",
    "parity", "days_in_milk",
    "breed_exotic", "breed_local",
    "grazing_zero", "grazing_tethered", "grazing_open",
]  # fmt: skip

# Fallback herd values, used when a cow has no history and no reference was given.
# (train.py computes real herd medians from the training farms and saves them with the model.)
DEFAULT_REFERENCE = {
    "baseline": {  # per breed: typical healthy ec_mean / temperature / yield per milking
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
    values = ec.to_numpy(dtype=float)
    n_present = np.sum(~np.isnan(values), axis=1)
    with warnings.catch_warnings():  # rows with all quarters missing -> NaN (handled later)
        warnings.simplefilter("ignore", RuntimeWarning)
        ec_mean = np.nanmean(values, axis=1)
        ec_max = np.nanmax(values, axis=1)
        others_mean = (np.nansum(values, axis=1) - ec_max) / (n_present - 1)
    enough = n_present >= MIN_QUARTERS_TO_COMPARE
    diff = np.where(enough, ec_max - others_mean, 0.0)
    ratio = np.where(enough, ec_max / others_mean, 1.0)
    return pd.DataFrame(
        {"ec_mean": ec_mean, "ec_max": ec_max, "ec_quarter_diff": diff, "ec_quarter_ratio": ratio},
        index=ec.index,
    )


def _rolling_prev_mean(df: pd.DataFrame, col: str, window: str, min_periods: int = 1) -> pd.Series:
    """Per-cow mean of `col` over the previous `window`, EXCLUDING the current milking.
    NaN values are skipped and don't count towards min_periods."""
    out = pd.Series(np.nan, index=df.index)
    for _, group in df.groupby("cow_id", sort=False):
        series = group.set_index("timestamp")[col]
        out.loc[group.index] = series.rolling(window, closed="left", min_periods=min_periods).mean().to_numpy()
    return out


def _clean_baseline(df: pd.DataFrame, col: str) -> pd.Series:
    """The cow's normal for `col` from previous CLEAN readings: 7 days (>= 3 readings),
    else 21 days (>= 3 readings), else NaN (-> herd median)."""
    clean = df[col].where(~df["_exclude"])  # excluded readings become NaN = ignored
    tmp = df[["cow_id", "timestamp"]].assign(_v=clean)
    short = _rolling_prev_mean(tmp, "_v", BASELINE_WINDOW, MIN_CLEAN_READINGS)
    long = _rolling_prev_mean(tmp, "_v", BASELINE_WINDOW_LONG, MIN_CLEAN_READINGS)
    return short.fillna(long)


def _pct_change(current: pd.Series, baseline: pd.Series) -> pd.Series:
    return 100.0 * (current - baseline) / baseline


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with a 0..n-1 index, sorted by cow and time, single sensor gaps filled from the
    cow's previous reading, and the baseline-exclusion flag as a boolean column."""
    df = df.reset_index(drop=True)  # simple 0..n-1 index; the output keeps this order
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["cow_id", "timestamp"], kind="stable")
    for col in FILLED_SENSORS:
        df[col] = df.groupby("cow_id", sort=False)[col].ffill()
    if "baseline_exclude" in df:
        df["_exclude"] = df["baseline_exclude"].fillna(False).astype(bool)
    else:
        df["_exclude"] = False
    return df


def _add_baseline_features(df: pd.DataFrame, feats: pd.DataFrame, ref: dict) -> None:
    """ec / yield / temp vs the cow's clean baseline, and the 3-day yield trend (in place)."""
    for col in BASELINE_SENSORS:
        base = _clean_baseline(df, col)
        # Too little clean history -> fall back to the herd median for that breed
        herd = df["breed"].map(lambda breed, col=col: ref["baseline"].get(breed, ref["baseline"]["exotic"])[col])
        df[f"_base_{col}"] = base.fillna(herd)

    feats["ec_vs_baseline"] = _pct_change(df["ec_mean"], df["_base_ec_mean"])
    feats["yield_vs_baseline"] = _pct_change(df["milk_yield_l"], df["_base_milk_yield_l"])
    feats["temp_vs_baseline"] = _pct_change(df["milk_temp_c"], df["_base_milk_temp_c"])

    # Short-term yield trend: today vs the previous 3 days (0 if no history)
    prev3 = _rolling_prev_mean(df, "milk_yield_l", YIELD_WINDOW)
    feats["yield_change_3d"] = _pct_change(df["milk_yield_l"], prev3).fillna(0.0)


def _add_cow_features(df: pd.DataFrame, feats: pd.DataFrame) -> None:
    """Raw readings, parity, days in milk and one-hot breed / grazing system (in place)."""
    for col in ["ph", "milk_temp_c", "milk_yield_l", "parity", "days_in_milk"]:
        feats[col] = pd.to_numeric(df[col], errors="coerce")
    for breed in BREEDS:
        feats[f"breed_{breed}"] = (df["breed"] == breed).astype(int)
    for system in GRAZING:
        feats[f"grazing_{system}"] = (df["grazing_system"] == system).astype(int)


def _fill_remaining(df: pd.DataFrame, feats: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Anything still missing -> herd median (or a neutral value)."""
    for col, value in ref["raw"].items():
        feats[col] = feats[col].fillna(value)
    feats["ec_mean"] = feats["ec_mean"].fillna(df["_base_ec_mean"])
    feats["ec_max"] = feats["ec_max"].fillna(feats["ec_mean"])
    feats = feats.fillna({"ec_vs_baseline": 0.0, "yield_vs_baseline": 0.0, "temp_vs_baseline": 0.0})
    return feats.replace([np.inf, -np.inf], 0.0)


def build_features(df: pd.DataFrame, reference: dict | None = None) -> pd.DataFrame:
    """Turn raw milking rows into model features (same row order as `df`, columns = FEATURES).

    reference: herd medians from training (see compute_reference). Used to fill gaps and as
               the baseline for cows without enough clean history.
    """
    ref = reference or DEFAULT_REFERENCE
    df = _prepare(df)

    feats = _quarter_features(df[EC_COLS])
    # All 4 quarters missing but a whole-milk EC reading exists (optional column, API only)
    if "ec_ms_cm" in df:
        feats["ec_mean"] = feats["ec_mean"].fillna(pd.to_numeric(df["ec_ms_cm"], errors="coerce"))
    df["ec_mean"] = feats["ec_mean"]

    _add_baseline_features(df, feats, ref)
    _add_cow_features(df, feats)
    feats = _fill_remaining(df, feats, ref)
    return feats[FEATURES].sort_index()  # back to the caller's original row order


def compute_reference(df: pd.DataFrame) -> dict:
    """Herd medians from (training) data, saved with the model for use at serving time."""
    data = df.copy()
    data["ec_mean"] = data[EC_COLS].mean(axis=1)
    baseline = {}
    for breed in BREEDS:
        subset = data[data["breed"] == breed]
        baseline[breed] = {col: float(subset[col].median()) for col in BASELINE_SENSORS}
    raw = {col: float(data[col].median()) for col in ["ph", "milk_temp_c", "milk_yield_l", "parity", "days_in_milk"]}
    return {"baseline": baseline, "raw": raw}


def suspect_quarter(row: dict) -> str | None:
    """Quarter with the highest EC (where an infection is most likely), lowercase key, or None."""
    values = {quarter: row.get(f"ec_{quarter}") for quarter in QUARTERS}
    values = {quarter: value for quarter, value in values.items() if value is not None and not pd.isna(value)}
    return max(values, key=values.get) if values else None
