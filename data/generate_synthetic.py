"""
Synthetic dataset generator for subclinical mastitis (SCM) detection.

*** ALL DATA PRODUCED HERE IS SIMULATED. ***
It is meant to look plausible for smallholder dairy farms in Uganda, so the
ML pipeline can be built and demonstrated before real sensor data exists.

How it works (one cow at a time):
  1. Draw cow attributes (breed, parity, days in milk, farm -> grazing system).
  2. Walk through the 90 days. Each day a healthy cow has a small chance
     ("hazard") of starting an infection episode. Risk factors multiply it.
  3. An episode infects 1 quarter (sometimes 2) for 7-30 days. Its intensity
     ramps up over a few days, plateaus, then ramps down.
  4. For every milking (twice a day) we compute per-quarter electrical
     conductivity (EC), pH, milk temperature and yield from the healthy
     baseline + infection effect + sensor noise.
  5. A CMT score is recorded on ~1 in 7 milkings (like a real farm).
  6. ~3% of sensor values are blanked out to mimic missing readings.

Run:  python data/generate_synthetic.py
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
    "cows_per_farm": 30,            # 10 x 30 = 300 cows
    "n_days": 90,
    "milkings_per_day": 2,          # am + pm
    "start_date": "2026-01-05",

    # Cow attributes
    "p_exotic": 0.70,               # exotic (Friesian/crosses) 70%, local (Ankole/Zebu) 30%
    "parity_range": (1, 7),
    "dim_start_range": (5, 220),    # days in milk at the start of the period
    "grazing_systems": ["zero", "tethered", "open"],
    "grazing_probs": [0.4, 0.35, 0.25],   # assigned per FARM (a farm uses one system)

    # Infection risk: target share of cows with >= 1 episode (Ugandan studies: 53-90%)
    "target_cow_prevalence": 0.60,
    "rr_exotic": 1.8,               # relative risk multipliers on the daily hazard
    "rr_parity_gt1": 1.5,
    "rr_zero_grazing": 1.6,
    "rr_early_lactation": 1.7,      # DIM < early_lactation_dim
    "early_lactation_dim": 60,
    "farm_hygiene_sd": 0.35,        # random farm effect (log scale) - some farms are worse
    "p_two_quarters": 0.2,          # chance an episode infects 2 quarters
    "episode_days": (7, 30),
    "ramp_up_days": (2, 4),         # EC rises "over a few days"
    "ramp_down_days": (2, 3),

    # Healthy milk (per quarter EC, whole-milk pH and temperature)
    "ec_healthy": (4.5, 6.0),       # mS/cm, cow baseline drawn in this range
    "ec_quarter_sd": 0.15,          # small natural differences between quarters
    "farm_sensor_offset_sd": 0.15,  # different sensor units per farm (calibration)
    "ph_healthy": (6.6, 6.8),
    "temp_healthy": (37.0, 38.0),   # deg C

    # Infected quarter effects (at full intensity)
    "ec_rise": (1.5, 3.5),
    "ph_infected": (6.9, 7.2),
    "temp_rise": (0.3, 0.8),
    "yield_drop": (0.05, 0.15),
    # Not every infection changes whole-milk pH / temperature clearly (one infected quarter is
    # diluted by three healthy ones). Share of episodes with the full pH/temp shift; the rest
    # get only 0-30% of it. Set to 1.0 for "every infection shifts pH and temp".
    "p_systemic_response": 0.7,

    # Yield per milking at lactation peak (litres)
    "peak_yield_exotic": (5.0, 8.0),
    "peak_yield_local": (1.5, 3.0),
    "pm_yield_factor": 0.85,        # evening milking is usually smaller

    # Sensor noise
    "noise_ec": 0.20,
    "noise_ph": 0.03,
    "noise_temp": 0.10,
    "noise_yield_frac": 0.06,
    "p_ec_glitch": 0.003,           # occasional dirty-electrode spike
    "p_missing": 0.03,              # ~3% missing sensor values

    # CMT (California Mastitis Test) recording
    "p_cmt_recorded": 1 / 7,
    "p_cmt_misread": 0.05,          # CMT is subjective: +-1 reading error

    "output": "data/synthetic_milkings.csv",
}

QUARTERS = ["lf", "rf", "lr", "rr"]   # left-front, right-front, left-rear, right-rear


def lactation_curve(dim):
    """Wood's lactation curve, scaled so the peak (~day 50) equals 1."""
    b, c = 0.2, 0.004
    y = np.power(np.maximum(dim, 1), b) * np.exp(-c * dim)
    peak = np.power(b / c, b) * np.exp(-b)
    return y / peak


def make_cows(rng, cfg):
    """Draw farm- and cow-level attributes."""
    farm_grazing = rng.choice(cfg["grazing_systems"], size=cfg["n_farms"], p=cfg["grazing_probs"])
    farm_hygiene = rng.normal(0, cfg["farm_hygiene_sd"], cfg["n_farms"])
    farm_sensor_offset = rng.normal(0, cfg["farm_sensor_offset_sd"], cfg["n_farms"])

    rows = []
    for f in range(cfg["n_farms"]):
        for k in range(cfg["cows_per_farm"]):
            breed = "exotic" if rng.random() < cfg["p_exotic"] else "local"
            # Parity skewed towards younger cows (more 1st-3rd lactation animals)
            parity = int(min(cfg["parity_range"][1], 1 + rng.poisson(1.5)))
            peak = cfg["peak_yield_exotic"] if breed == "exotic" else cfg["peak_yield_local"]
            rows.append({
                "cow_id": f"F{f + 1:02d}-C{k + 1:03d}",
                "farm_id": f"F{f + 1:02d}",
                "farm_idx": f,
                "breed": breed,
                "parity": parity,
                "grazing_system": farm_grazing[f],
                "dim_start": int(rng.integers(*cfg["dim_start_range"])),
                "farm_hygiene": farm_hygiene[f],
                "farm_sensor_offset": farm_sensor_offset[f],
                "ec_base": rng.uniform(*cfg["ec_healthy"]),
                "ph_base": rng.uniform(*cfg["ph_healthy"]),
                "temp_base": rng.uniform(*cfg["temp_healthy"]),
                "peak_yield": rng.uniform(*peak),
            })
    return pd.DataFrame(rows)


def daily_hazard_multiplier(cow, dims, cfg):
    """Relative risk of starting an episode on each day (risk factors combined)."""
    m = np.exp(cow["farm_hygiene"])
    if cow["breed"] == "exotic":
        m *= cfg["rr_exotic"]
    if cow["parity"] > 1:
        m *= cfg["rr_parity_gt1"]
    if cow["grazing_system"] == "zero":
        m *= cfg["rr_zero_grazing"]
    return m * np.where(dims < cfg["early_lactation_dim"], cfg["rr_early_lactation"], 1.0)


def calibrate_base_hazard(cows, cfg):
    """Find the base daily hazard so that ~target% of cows get >= 1 episode.

    P(cow never infected) = prod_d (1 - h_d). We bisect on the base hazard so the
    herd average of P(>=1 episode) matches the target prevalence.
    """
    mults = [daily_hazard_multiplier(c, c["dim_start"] + np.arange(cfg["n_days"]), cfg)
             for _, c in cows.iterrows()]
    lo, hi = 1e-6, 0.2
    for _ in range(60):
        mid = (lo + hi) / 2
        p_any = np.mean([1 - np.prod(1 - np.minimum(mid * m, 0.99)) for m in mults])
        lo, hi = (mid, hi) if p_any < cfg["target_cow_prevalence"] else (lo, mid)
    return (lo + hi) / 2


def simulate_episodes(rng, cow, base_hazard, cfg):
    """Return per-day, per-quarter infection intensity (n_days x 4) and severity rise values."""
    n = cfg["n_days"]
    intensity = np.zeros((n, 4))
    ec_rise = np.zeros(4)
    sev = {"ph": cfg["ph_healthy"][0], "temp": 0.0, "yield": 0.0}   # cow-level effects
    dims = cow["dim_start"] + np.arange(n)
    hazard = np.minimum(base_hazard * daily_hazard_multiplier(cow, dims, cfg), 0.99)

    episodes = []
    d = 0
    while d < n:
        if rng.random() < hazard[d]:
            length = int(rng.integers(cfg["episode_days"][0], cfg["episode_days"][1] + 1))
            n_q = 2 if rng.random() < cfg["p_two_quarters"] else 1
            quarters = rng.choice(4, size=n_q, replace=False)
            up = int(rng.integers(cfg["ramp_up_days"][0], cfg["ramp_up_days"][1] + 1))
            down = int(rng.integers(cfg["ramp_down_days"][0], cfg["ramp_down_days"][1] + 1))
            # Intensity shape: ramp up -> plateau -> ramp down
            shape = np.ones(length)
            shape[:up] = np.linspace(1 / up, 1, up)
            shape[-down:] = np.minimum(shape[-down:], np.linspace(1, 1 / down, down))
            end = min(n, d + length)
            for q in quarters:
                intensity[d:end, q] = np.maximum(intensity[d:end, q], shape[: end - d])
                ec_rise[q] = rng.uniform(*cfg["ec_rise"])
            # Full pH/temp response in most episodes, weak response in the rest
            systemic = 1.0 if rng.random() < cfg["p_systemic_response"] else rng.uniform(0, 0.3)
            ph_target = rng.uniform(*cfg["ph_infected"])
            sev = {
                "ph": cow["ph_base"] + systemic * (ph_target - cow["ph_base"]),
                "temp": systemic * rng.uniform(*cfg["temp_rise"]),
                "yield": rng.uniform(*cfg["yield_drop"]),
            }
            episodes.append((d, end, list(quarters)))
            d = end + 3   # short recovery gap before a possible new episode
        else:
            d += 1
    return intensity, ec_rise, sev, episodes


def cmt_from_intensity(rng, cow_intensity, ec_rise_max, cfg):
    """Map infection intensity to a CMT score 0-4 (0 = negative ... 4 = strong positive).

    Field scale is 1-5, so our 0-4 = field score - 1. Positive (SCM) = cmt >= 2.
    """
    if cow_intensity <= 0:
        score = rng.choice([0, 1, 2], p=[0.85, 0.13, 0.02])   # healthy, occasional trace
    else:
        strength = cow_intensity * (ec_rise_max / 3.5)          # stronger EC rise -> stronger CMT
        score = int(np.clip(np.round(0.8 + 3.5 * strength + rng.normal(0, 0.5)), 0, 4))
    if rng.random() < cfg["p_cmt_misread"]:                     # human reading error
        score = int(np.clip(score + rng.choice([-1, 1]), 0, 4))
    return score


def generate(cfg=CONFIG):
    rng = np.random.default_rng(cfg["seed"])
    cows = make_cows(rng, cfg)
    base_hazard = calibrate_base_hazard(cows, cfg)
    start = pd.Timestamp(cfg["start_date"])
    n_days, mpd = cfg["n_days"], cfg["milkings_per_day"]

    records = []
    n_cows_infected = 0
    for _, cow in cows.iterrows():
        intensity, ec_rise, sev, episodes = simulate_episodes(rng, cow, base_hazard, cfg)
        n_cows_infected += len(episodes) > 0
        q_offsets = rng.normal(0, cfg["ec_quarter_sd"], 4)

        for d in range(n_days):
            dim = cow["dim_start"] + d
            for m in range(mpd):
                pm = m == 1
                ts = start + pd.Timedelta(days=d, hours=18 if pm else 6)
                q_int = intensity[d]
                cow_int = q_int.max()

                # Per-quarter EC: baseline + quarter offset + farm sensor offset + infection + noise
                ec = (cow["ec_base"] + q_offsets + cow["farm_sensor_offset"]
                      + ec_rise * q_int + rng.normal(0, cfg["noise_ec"], 4))
                glitch = rng.random(4) < cfg["p_ec_glitch"]
                ec[glitch] += rng.uniform(1.0, 2.5, glitch.sum())

                ph = cow["ph_base"] + (sev["ph"] - cow["ph_base"]) * cow_int + rng.normal(0, cfg["noise_ph"])
                temp = (cow["temp_base"] + sev["temp"] * cow_int + (0.1 if pm else 0.0)
                        + rng.normal(0, cfg["noise_temp"]))
                yld = (cow["peak_yield"] * lactation_curve(dim) * (cfg["pm_yield_factor"] if pm else 1.0)
                       * (1 - sev["yield"] * cow_int) * (1 + rng.normal(0, cfg["noise_yield_frac"])))

                cmt = np.nan
                if rng.random() < cfg["p_cmt_recorded"]:
                    rise_max = ec_rise[q_int > 0].max() if cow_int > 0 else 0.0
                    cmt = cmt_from_intensity(rng, cow_int, rise_max, cfg)

                rec = {
                    "cow_id": cow["cow_id"], "farm_id": cow["farm_id"],
                    "timestamp": ts, "day": d, "milking": "pm" if pm else "am",
                    "breed": cow["breed"], "parity": cow["parity"],
                    "days_in_milk": dim, "grazing_system": cow["grazing_system"],
                }
                for i, q in enumerate(QUARTERS):
                    rec[f"ec_{q}"] = round(float(ec[i]), 2)
                rec.update({
                    "ph": round(float(ph), 2),
                    "milk_temp_c": round(float(temp), 2),
                    "milk_yield_l": round(float(max(yld, 0.1)), 2),
                    "cmt_score": cmt,
                    # Ground truth (NOT a model input): useful for analysis and the demo
                    "true_infected": int(cow_int >= 0.3),
                    "true_infected_quarters": ",".join(QUARTERS[i] for i in np.where(q_int > 0)[0]),
                })
                records.append(rec)

    df = pd.DataFrame(records)

    # ~3% missing values on every sensor column
    sensor_cols = [f"ec_{q}" for q in QUARTERS] + ["ph", "milk_temp_c", "milk_yield_l"]
    for col in sensor_cols:
        df.loc[rng.random(len(df)) < cfg["p_missing"], col] = np.nan

    df["cmt_score"] = df["cmt_score"].astype("Int64")
    return df, n_cows_infected, base_hazard


def main():
    df, n_inf, base_hazard = generate(CONFIG)
    out = Path(CONFIG["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_cows = df["cow_id"].nunique()
    labelled = df["cmt_score"].notna()
    pos = (df.loc[labelled, "cmt_score"] >= 2).mean()
    print(f"SIMULATED data written to {out}")
    print(f"  rows: {len(df):,}  cows: {n_cows}  farms: {df['farm_id'].nunique()}")
    print(f"  base daily hazard: {base_hazard:.4f}")
    print(f"  cows with >=1 SCM episode: {n_inf}/{n_cows} ({n_inf / n_cows:.0%})")
    print(f"  CMT-labelled milkings: {labelled.sum():,} ({labelled.mean():.1%}), positive (cmt>=2): {pos:.1%}")
    print(f"  missing sensor values: {df[['ec_lf', 'ph', 'milk_yield_l']].isna().mean().round(3).to_dict()}")


if __name__ == "__main__":
    main()
