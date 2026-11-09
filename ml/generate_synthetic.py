"""
Synthetic dataset generator for subclinical mastitis (SCM) detection - the ml/ layer's data source.

*** ALL DATA PRODUCED HERE IS SIMULATED. ***
It is meant to look plausible for smallholder dairy farms in Uganda, so the ML pipeline
(ml/train.py) and the app (seed + live demo) can be built before real sensor data exists.

How it works (one cow at a time):
  1. Draw cow attributes (breed, parity, days in milk, farm -> grazing system).
  2. Walk through the 90 days. Each day a healthy cow has a small chance ("hazard") of
     starting an infection episode. Risk factors multiply it.
  3. An episode infects 1 quarter (sometimes 2) for 7-30 days. Its intensity ramps up over a
     few days, plateaus, then ramps down.
  4. For every milking (twice a day) compute per-quarter conductivity (EC), pH, milk temperature
     and yield from the healthy baseline + infection effect + sensor noise.
  5. A CMT score is recorded on ~1 in 7 milkings (like a real farm).
  6. ~3% of sensor values are blanked out to mimic missing readings.

Run:  python -m ml.generate_synthetic
Out:  data/synthetic_milkings.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# All parameters in one place (easy to defend / tweak in front of a panel)
# --------------------------------------------------------------------------
CONFIG = {
    "seed": 42,
    "n_farms": 10,
    "cows_per_farm": 30,  # 10 x 30 = 300 cows
    "n_days": 90,
    "milkings_per_day": 2,  # am + pm
    "milking_hours": (6, 18),  # am and pm milking times
    "start_date": "2026-01-05",
    # Cow attributes
    "p_exotic": 0.70,  # exotic (Friesian/crosses) 70%, local (Ankole/Zebu) 30%
    "parity_range": (1, 7),
    "parity_poisson_mean": 1.5,  # parity = 1 + Poisson: skewed towards younger cows
    "dim_start_range": (5, 220),  # days in milk at the start of the period
    "grazing_systems": ["zero", "tethered", "open"],
    "grazing_probs": [0.4, 0.35, 0.25],  # assigned per FARM (a farm uses one system)
    # Infection risk: target share of cows with >= 1 episode (Ugandan studies: 53-90%)
    "target_cow_prevalence": 0.60,
    "rr_exotic": 1.8,  # relative risk multipliers on the daily hazard
    "rr_parity_gt1": 1.5,
    "rr_zero_grazing": 1.6,
    "rr_early_lactation": 1.7,  # DIM < early_lactation_dim
    "early_lactation_dim": 60,
    "farm_hygiene_sd": 0.35,  # random farm effect (log scale) - some farms are worse
    "max_daily_hazard": 0.99,
    "p_two_quarters": 0.2,  # chance an episode infects 2 quarters
    "episode_days": (7, 30),
    "ramp_up_days": (2, 4),  # EC rises "over a few days"
    "ramp_down_days": (2, 3),
    "gap_after_episode_days": 3,  # short recovery gap before a possible new episode
    # Healthy milk (per quarter EC, whole-milk pH and temperature)
    "ec_healthy": (4.5, 6.0),  # mS/cm, cow baseline drawn in this range
    "ec_quarter_sd": 0.15,  # small natural differences between quarters
    "farm_sensor_offset_sd": 0.15,  # different sensor units per farm (calibration)
    "ph_healthy": (6.6, 6.8),
    "temp_healthy": (37.0, 38.0),  # deg C
    "pm_temp_offset": 0.1,  # evening milk is slightly warmer
    # Infected quarter effects (at full intensity)
    "ec_rise": (1.5, 3.5),
    "ph_infected": (6.9, 7.2),
    "temp_rise": (0.3, 0.8),
    "yield_drop": (0.05, 0.15),
    # Not every infection changes whole-milk pH / temperature clearly (one infected quarter is
    # diluted by three healthy ones). Share of episodes with the full pH/temp shift; the rest
    # get only 0-30% of it. Set to 1.0 for "every infection shifts pH and temp".
    "p_systemic_response": 0.7,
    "weak_systemic_response": (0, 0.3),
    "infected_intensity": 0.3,  # intensity from which a milking counts as "truly infected"
    # Yield per milking at lactation peak (litres)
    "peak_yield_exotic": (5.0, 8.0),
    "peak_yield_local": (1.5, 3.0),
    "pm_yield_factor": 0.85,  # evening milking is usually smaller
    "min_yield_l": 0.1,
    # Sensor noise
    "noise_ec": 0.20,
    "noise_ph": 0.03,
    "noise_temp": 0.10,
    "noise_yield_frac": 0.06,
    "p_ec_glitch": 0.003,  # occasional dirty-electrode spike
    "ec_glitch_rise": (1.0, 2.5),
    "p_missing": 0.03,  # ~3% missing sensor values
    # CMT (California Mastitis Test) recording
    "p_cmt_recorded": 1 / 7,
    "p_cmt_misread": 0.05,  # CMT is subjective: +-1 reading error
    "cmt_healthy_probs": [0.85, 0.13, 0.02],  # healthy cow: score 0 / 1 / 2
    "output": "data/synthetic_milkings.csv",
}

QUARTERS = ["lf", "rf", "lr", "rr"]  # left-front, right-front, left-rear, right-rear
N_QUARTERS = len(QUARTERS)

# Wood's lactation curve y = dim^b * exp(-c * dim); the peak is at dim = b / c (~day 50)
WOOD_B, WOOD_C = 0.2, 0.004
# CMT score from infection strength: round(offset + slope * strength + noise)
CMT_OFFSET, CMT_SLOPE, CMT_NOISE_SD, CMT_MAX = 0.8, 3.5, 0.5, 4
EC_RISE_MAX = 3.5  # strongest EC rise; scales the CMT strength
# Bisection that calibrates the base hazard
HAZARD_SEARCH_RANGE = (1e-6, 0.2)
HAZARD_SEARCH_STEPS = 60


def lactation_curve(dim: np.ndarray | int) -> np.ndarray:
    """Wood's lactation curve, scaled so the peak (~day 50) equals 1."""
    curve = np.power(np.maximum(dim, 1), WOOD_B) * np.exp(-WOOD_C * dim)
    peak = np.power(WOOD_B / WOOD_C, WOOD_B) * np.exp(-WOOD_B)
    return curve / peak


def make_cows(rng: np.random.Generator, cfg: dict) -> pd.DataFrame:
    """Draw farm- and cow-level attributes."""
    farm_grazing = rng.choice(cfg["grazing_systems"], size=cfg["n_farms"], p=cfg["grazing_probs"])
    farm_hygiene = rng.normal(0, cfg["farm_hygiene_sd"], cfg["n_farms"])
    farm_sensor_offset = rng.normal(0, cfg["farm_sensor_offset_sd"], cfg["n_farms"])

    rows = []
    for farm in range(cfg["n_farms"]):
        for cow in range(cfg["cows_per_farm"]):
            breed = "exotic" if rng.random() < cfg["p_exotic"] else "local"
            parity = int(min(cfg["parity_range"][1], 1 + rng.poisson(cfg["parity_poisson_mean"])))
            peak = cfg["peak_yield_exotic"] if breed == "exotic" else cfg["peak_yield_local"]
            rows.append(
                {
                    "cow_id": f"F{farm + 1:02d}-C{cow + 1:03d}",
                    "farm_id": f"F{farm + 1:02d}",
                    "farm_idx": farm,
                    "breed": breed,
                    "parity": parity,
                    "grazing_system": farm_grazing[farm],
                    "dim_start": int(rng.integers(*cfg["dim_start_range"])),
                    "farm_hygiene": farm_hygiene[farm],
                    "farm_sensor_offset": farm_sensor_offset[farm],
                    "ec_base": rng.uniform(*cfg["ec_healthy"]),
                    "ph_base": rng.uniform(*cfg["ph_healthy"]),
                    "temp_base": rng.uniform(*cfg["temp_healthy"]),
                    "peak_yield": rng.uniform(*peak),
                }
            )
    return pd.DataFrame(rows)


def daily_hazard_multiplier(cow: pd.Series, dims: np.ndarray, cfg: dict) -> np.ndarray:
    """Relative risk of starting an episode on each day (risk factors combined)."""
    multiplier = np.exp(cow["farm_hygiene"])
    if cow["breed"] == "exotic":
        multiplier *= cfg["rr_exotic"]
    if cow["parity"] > 1:
        multiplier *= cfg["rr_parity_gt1"]
    if cow["grazing_system"] == "zero":
        multiplier *= cfg["rr_zero_grazing"]
    return multiplier * np.where(dims < cfg["early_lactation_dim"], cfg["rr_early_lactation"], 1.0)


def calibrate_base_hazard(cows: pd.DataFrame, cfg: dict) -> float:
    """Find the base daily hazard so that ~target% of cows get >= 1 episode.

    P(cow never infected) = prod_d (1 - h_d). We bisect on the base hazard so the
    herd average of P(>=1 episode) matches the target prevalence.
    """
    multipliers = [
        daily_hazard_multiplier(cow, cow["dim_start"] + np.arange(cfg["n_days"]), cfg) for _, cow in cows.iterrows()
    ]
    low, high = HAZARD_SEARCH_RANGE
    for _ in range(HAZARD_SEARCH_STEPS):
        mid = (low + high) / 2
        p_any = np.mean([1 - np.prod(1 - np.minimum(mid * mult, cfg["max_daily_hazard"])) for mult in multipliers])
        low, high = (mid, high) if p_any < cfg["target_cow_prevalence"] else (low, mid)
    return (low + high) / 2


def _episode_shape(length: int, ramp_up: int, ramp_down: int) -> np.ndarray:
    """Intensity over an episode: ramp up -> plateau at 1 -> ramp down."""
    shape = np.ones(length)
    shape[:ramp_up] = np.linspace(1 / ramp_up, 1, ramp_up)
    shape[-ramp_down:] = np.minimum(shape[-ramp_down:], np.linspace(1, 1 / ramp_down, ramp_down))
    return shape


def _draw_episode(
    rng: np.random.Generator, cow: pd.Series, start: int, intensity: np.ndarray, ec_rise: np.ndarray, cfg: dict
) -> tuple[int, dict, list]:
    """Draw one episode starting on day `start`; writes into intensity / ec_rise.
    Returns (end day, cow-level severity, infected quarters). The order of random draws matters
    for reproducibility - do not reorder."""
    n_days = cfg["n_days"]
    length = int(rng.integers(cfg["episode_days"][0], cfg["episode_days"][1] + 1))
    n_infected = 2 if rng.random() < cfg["p_two_quarters"] else 1
    quarters = rng.choice(N_QUARTERS, size=n_infected, replace=False)
    ramp_up = int(rng.integers(cfg["ramp_up_days"][0], cfg["ramp_up_days"][1] + 1))
    ramp_down = int(rng.integers(cfg["ramp_down_days"][0], cfg["ramp_down_days"][1] + 1))
    shape = _episode_shape(length, ramp_up, ramp_down)
    end = min(n_days, start + length)
    for quarter in quarters:
        intensity[start:end, quarter] = np.maximum(intensity[start:end, quarter], shape[: end - start])
        ec_rise[quarter] = rng.uniform(*cfg["ec_rise"])
    # Full pH/temp response in most episodes, weak response in the rest
    systemic = 1.0 if rng.random() < cfg["p_systemic_response"] else rng.uniform(*cfg["weak_systemic_response"])
    ph_target = rng.uniform(*cfg["ph_infected"])
    severity = {
        "ph": cow["ph_base"] + systemic * (ph_target - cow["ph_base"]),
        "temp": systemic * rng.uniform(*cfg["temp_rise"]),
        "yield": rng.uniform(*cfg["yield_drop"]),
    }
    return end, severity, list(quarters)


def simulate_episodes(
    rng: np.random.Generator, cow: pd.Series, base_hazard: float, cfg: dict
) -> tuple[np.ndarray, np.ndarray, dict, list]:
    """Per-day, per-quarter infection intensity (n_days x 4), EC rise per quarter, cow-level
    severity of the last episode, and the list of episodes (start, end, quarters)."""
    n_days = cfg["n_days"]
    intensity = np.zeros((n_days, N_QUARTERS))
    ec_rise = np.zeros(N_QUARTERS)
    severity = {"ph": cfg["ph_healthy"][0], "temp": 0.0, "yield": 0.0}  # cow-level effects
    dims = cow["dim_start"] + np.arange(n_days)
    hazard = np.minimum(base_hazard * daily_hazard_multiplier(cow, dims, cfg), cfg["max_daily_hazard"])

    episodes = []
    day = 0
    while day < n_days:
        if rng.random() < hazard[day]:
            end, severity, quarters = _draw_episode(rng, cow, day, intensity, ec_rise, cfg)
            episodes.append((day, end, quarters))
            day = end + cfg["gap_after_episode_days"]
        else:
            day += 1
    return intensity, ec_rise, severity, episodes


def cmt_from_intensity(rng: np.random.Generator, cow_intensity: float, ec_rise_max: float, cfg: dict) -> int:
    """Map infection intensity to a CMT score 0-4 (0 = negative ... 4 = strong positive).

    Field scale is 1-5, so our 0-4 = field score - 1. Positive (SCM) = cmt >= 2.
    """
    if cow_intensity <= 0:
        score = rng.choice([0, 1, 2], p=cfg["cmt_healthy_probs"])  # healthy, occasional trace
    else:
        strength = cow_intensity * (ec_rise_max / EC_RISE_MAX)  # stronger EC rise -> stronger CMT
        score = int(np.clip(np.round(CMT_OFFSET + CMT_SLOPE * strength + rng.normal(0, CMT_NOISE_SD)), 0, CMT_MAX))
    if rng.random() < cfg["p_cmt_misread"]:  # human reading error
        score = int(np.clip(score + rng.choice([-1, 1]), 0, CMT_MAX))
    return score


def _sensor_values(
    rng: np.random.Generator,
    cow: pd.Series,
    day: int,
    is_pm: bool,
    infection: dict,
    quarter_offsets: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, float, float, float]:
    """Per-quarter EC, pH, temperature and yield for one milking (healthy level + infection + noise).
    The order of random draws matters for reproducibility - do not reorder."""
    quarter_intensity = infection["intensity"][day]
    cow_intensity = quarter_intensity.max()
    severity = infection["severity"]

    # Per-quarter EC: baseline + quarter offset + farm sensor offset + infection + noise
    ec = (
        cow["ec_base"]
        + quarter_offsets
        + cow["farm_sensor_offset"]
        + infection["ec_rise"] * quarter_intensity
        + rng.normal(0, cfg["noise_ec"], N_QUARTERS)
    )
    glitch = rng.random(N_QUARTERS) < cfg["p_ec_glitch"]
    ec[glitch] += rng.uniform(*cfg["ec_glitch_rise"], glitch.sum())

    ph = cow["ph_base"] + (severity["ph"] - cow["ph_base"]) * cow_intensity + rng.normal(0, cfg["noise_ph"])
    temp = (
        cow["temp_base"]
        + severity["temp"] * cow_intensity
        + (cfg["pm_temp_offset"] if is_pm else 0.0)
        + rng.normal(0, cfg["noise_temp"])
    )
    milk_yield = (
        cow["peak_yield"]
        * lactation_curve(cow["dim_start"] + day)
        * (cfg["pm_yield_factor"] if is_pm else 1.0)
        * (1 - severity["yield"] * cow_intensity)
        * (1 + rng.normal(0, cfg["noise_yield_frac"]))
    )
    return ec, ph, temp, milk_yield


def simulate_milking(
    rng: np.random.Generator,
    cow: pd.Series,
    day: int,
    is_pm: bool,
    infection: dict,
    quarter_offsets: np.ndarray,
    cfg: dict,
) -> dict:
    """One milking record (sensor values + CMT + ground truth). The order of random draws
    matters for reproducibility - do not reorder."""
    am_hour, pm_hour = cfg["milking_hours"]
    quarter_intensity = infection["intensity"][day]
    cow_intensity = quarter_intensity.max()
    ec, ph, temp, milk_yield = _sensor_values(rng, cow, day, is_pm, infection, quarter_offsets, cfg)

    cmt = np.nan
    if rng.random() < cfg["p_cmt_recorded"]:
        rise_max = infection["ec_rise"][quarter_intensity > 0].max() if cow_intensity > 0 else 0.0
        cmt = cmt_from_intensity(rng, cow_intensity, rise_max, cfg)

    record = {
        "cow_id": cow["cow_id"],
        "farm_id": cow["farm_id"],
        "timestamp": pd.Timestamp(cfg["start_date"]) + pd.Timedelta(days=day, hours=pm_hour if is_pm else am_hour),
        "day": day,
        "milking": "pm" if is_pm else "am",
        "breed": cow["breed"],
        "parity": cow["parity"],
        "days_in_milk": cow["dim_start"] + day,
        "grazing_system": cow["grazing_system"],
    }
    for index, quarter in enumerate(QUARTERS):
        record[f"ec_{quarter}"] = round(float(ec[index]), 2)
    record.update(
        {
            "ph": round(float(ph), 2),
            "milk_temp_c": round(float(temp), 2),
            "milk_yield_l": round(float(max(milk_yield, cfg["min_yield_l"])), 2),
            "cmt_score": cmt,
            # Ground truth (NOT a model input): useful for analysis and the demo
            "true_infected": int(cow_intensity >= cfg["infected_intensity"]),
            "true_infected_quarters": ",".join(QUARTERS[i] for i in np.where(quarter_intensity > 0)[0]),
        }
    )
    return record


def add_missing_values(df: pd.DataFrame, rng: np.random.Generator, cfg: dict) -> None:
    """Blank out ~p_missing of every sensor column (in place)."""
    sensor_cols = [f"ec_{quarter}" for quarter in QUARTERS] + ["ph", "milk_temp_c", "milk_yield_l"]
    for col in sensor_cols:
        df.loc[rng.random(len(df)) < cfg["p_missing"], col] = np.nan


def generate(cfg: dict = CONFIG) -> tuple[pd.DataFrame, int, float]:
    """Simulate the whole herd. Returns (milkings, number of cows with >= 1 episode, base hazard)."""
    rng = np.random.default_rng(cfg["seed"])
    cows = make_cows(rng, cfg)
    base_hazard = calibrate_base_hazard(cows, cfg)

    records = []
    n_cows_infected = 0
    for _, cow in cows.iterrows():
        intensity, ec_rise, severity, episodes = simulate_episodes(rng, cow, base_hazard, cfg)
        n_cows_infected += len(episodes) > 0
        quarter_offsets = rng.normal(0, cfg["ec_quarter_sd"], N_QUARTERS)
        infection = {"intensity": intensity, "ec_rise": ec_rise, "severity": severity}
        for day in range(cfg["n_days"]):
            for milking in range(cfg["milkings_per_day"]):
                records.append(simulate_milking(rng, cow, day, milking == 1, infection, quarter_offsets, cfg))

    df = pd.DataFrame(records)
    add_missing_values(df, rng, cfg)
    df["cmt_score"] = df["cmt_score"].astype("Int64")
    return df, n_cows_infected, base_hazard


def main() -> None:
    """Generate the dataset, write it to CONFIG['output'] and print a summary."""
    df, n_infected, base_hazard = generate(CONFIG)
    out = Path(CONFIG["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_cows = df["cow_id"].nunique()
    labelled = df["cmt_score"].notna()
    positive = (df.loc[labelled, "cmt_score"] >= 2).mean()
    print(f"SIMULATED data written to {out}")
    print(f"  rows: {len(df):,}  cows: {n_cows}  farms: {df['farm_id'].nunique()}")
    print(f"  base daily hazard: {base_hazard:.4f}")
    print(f"  cows with >=1 SCM episode: {n_infected}/{n_cows} ({n_infected / n_cows:.0%})")
    print(f"  CMT-labelled milkings: {labelled.sum():,} ({labelled.mean():.1%}), positive (cmt>=2): {positive:.1%}")
    print(f"  missing sensor values: {df[['ec_lf', 'ph', 'milk_yield_l']].isna().mean().round(3).to_dict()}")


if __name__ == "__main__":
    main()
