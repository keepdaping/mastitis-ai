"""Feature engineering must work for a brand-new cow (no history) and with missing values."""

import numpy as np
import pandas as pd

from ml.features import DEFAULT_REFERENCE, FEATURES, build_features


def one_reading(**overrides):
    row = {
        "cow_id": "NEW-COW",
        "timestamp": "2026-03-01 06:00",
        "breed": "local",
        "parity": 2,
        "days_in_milk": 30,
        "grazing_system": "open",
        "ec_lf": 5.0,
        "ec_rf": 5.1,
        "ec_lr": 5.0,
        "ec_rr": 5.2,
        "ph": 6.7,
        "milk_temp_c": 37.5,
        "milk_yield_l": 2.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_cow_with_no_history_uses_herd_fallback():
    X = build_features(one_reading())
    assert list(X.columns) == FEATURES
    assert len(X) == 1
    assert not X.isna().any().any()
    # Baseline = herd median for local cows -> EC vs baseline is a small % change
    herd_ec = DEFAULT_REFERENCE["baseline"]["local"]["ec_mean"]
    expected = 100 * (np.mean([5.0, 5.1, 5.0, 5.2]) - herd_ec) / herd_ec
    assert abs(X["ec_vs_baseline"].iloc[0] - expected) < 1e-9
    assert X["yield_change_3d"].iloc[0] == 0.0  # no 3-day history -> neutral
    assert X["breed_local"].iloc[0] == 1 and X["grazing_open"].iloc[0] == 1


def test_missing_values_are_filled():
    X = build_features(
        one_reading(
            ec_lf=np.nan, ec_rf=np.nan, ec_lr=np.nan, ec_rr=np.nan, ph=np.nan, milk_temp_c=np.nan, milk_yield_l=np.nan
        )
    )
    assert not X.isna().any().any()
    assert X["ec_quarter_ratio"].iloc[0] == 1.0  # cannot compare quarters -> neutral


def test_one_high_quarter_stands_out():
    X = build_features(one_reading(ec_rr=8.0))
    assert X["ec_quarter_diff"].iloc[0] > 2.5
    assert X["ec_quarter_ratio"].iloc[0] > 1.5


def test_baseline_uses_only_previous_readings_and_keeps_row_order():
    rows = pd.concat([one_reading(timestamp=f"2026-03-0{d} 06:00") for d in range(1, 6)], ignore_index=True)
    rows.loc[4, ["ec_lf", "ec_rf", "ec_lr", "ec_rr"]] = [6.0, 6.1, 6.0, 6.2]  # EC jumps on day 5
    shuffled = rows.iloc[[4, 0, 3, 1, 2]].reset_index(drop=True)
    X = build_features(shuffled)
    # Row 0 of the shuffled input is day 5: compared with days 1-4 (mean 5.075) -> ~ +20%
    assert abs(X["ec_vs_baseline"].iloc[0] - 100 * (6.075 - 5.075) / 5.075) < 1e-6


# ---------------------------------------------------------------- clean baseline
def cow_days(n, start_day=1, **overrides):
    """n twice-daily readings for one cow, starting on March `start_day`."""
    rows = []
    for i in range(n):
        ts = pd.Timestamp("2026-03-01 06:00") + pd.Timedelta(days=start_day - 1, hours=12 * i)
        rows.append(one_reading(timestamp=ts, **overrides))
    return pd.concat(rows, ignore_index=True)


def test_excluded_readings_do_not_become_the_baseline():
    df = cow_days(8)  # 4 days, EC 5.075 on average
    df.loc[4:6, ["ec_lf", "ec_rf", "ec_lr", "ec_rr"]] = [5.0, 5.1, 8.0, 5.2]  # infected readings
    df["baseline_exclude"] = [False] * 4 + [True] * 3 + [False]
    X = build_features(df)
    # Last reading is healthy again: its baseline is the 4 clean readings only -> ~0% change
    assert abs(X["ec_vs_baseline"].iloc[-1]) < 1e-9
    # Without the flags the infection would have raised the "normal" -> negative change
    assert build_features(df.drop(columns="baseline_exclude"))["ec_vs_baseline"].iloc[-1] < -5


def test_baseline_extends_to_21_days_then_herd_median():
    df = cow_days(20)  # 10 days of readings
    df["baseline_exclude"] = [False] * 3 + [True] * 17  # only the first 3 are clean
    X = build_features(df)
    # Last reading (day 10.5): < 3 clean readings in 7 days -> 21-day window (the 3 clean ones)
    assert abs(X["ec_vs_baseline"].iloc[-1]) < 1e-9
    df["baseline_exclude"] = [False] * 2 + [True] * 18  # only 2 clean readings at all
    herd = DEFAULT_REFERENCE["baseline"]["local"]["ec_mean"]
    expected = 100 * (5.075 - herd) / herd  # -> herd median
    assert abs(build_features(df)["ec_vs_baseline"].iloc[-1] - expected) < 1e-9


def test_first_readings_use_herd_median_until_three_clean():
    X = build_features(cow_days(4))
    herd = DEFAULT_REFERENCE["baseline"]["local"]["ec_mean"]
    expected = 100 * (5.075 - herd) / herd
    assert all(abs(v - expected) < 1e-9 for v in X["ec_vs_baseline"].iloc[:3])  # 0, 1, 2 previous
    assert abs(X["ec_vs_baseline"].iloc[3]) < 1e-9  # 3 previous -> own


def test_episode_flags_follow_the_live_rules():
    from ml.episodes import episode_exclusion_flags

    ts = pd.Series(pd.date_range("2026-03-01 06:00", periods=10, freq="12h"))
    high = [False, True, True, True, False, False, False, False, False, True]
    flags = episode_exclusion_flags(ts, high)
    # highs excluded; episode opens at reading 2, stays active for 3 normal readings, closes at the 4th
    assert list(flags) == [False, True, True, True, True, True, True, False, False, True]
