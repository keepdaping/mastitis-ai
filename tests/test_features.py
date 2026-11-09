"""Feature engineering must work for a brand-new cow (no history) and with missing values."""
import numpy as np
import pandas as pd

from ml.features import DEFAULT_REFERENCE, FEATURES, build_features


def one_reading(**overrides):
    row = {"cow_id": "NEW-COW", "timestamp": "2026-03-01 06:00", "breed": "local", "parity": 2,
           "days_in_milk": 30, "grazing_system": "open",
           "ec_lf": 5.0, "ec_rf": 5.1, "ec_lr": 5.0, "ec_rr": 5.2,
           "ph": 6.7, "milk_temp_c": 37.5, "milk_yield_l": 2.0}
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
    assert X["yield_change_3d"].iloc[0] == 0.0           # no 3-day history -> neutral
    assert X["breed_local"].iloc[0] == 1 and X["grazing_open"].iloc[0] == 1


def test_missing_values_are_filled():
    X = build_features(one_reading(ec_lf=np.nan, ec_rf=np.nan, ec_lr=np.nan, ec_rr=np.nan,
                                   ph=np.nan, milk_temp_c=np.nan, milk_yield_l=np.nan))
    assert not X.isna().any().any()
    assert X["ec_quarter_ratio"].iloc[0] == 1.0          # cannot compare quarters -> neutral


def test_one_high_quarter_stands_out():
    X = build_features(one_reading(ec_rr=8.0))
    assert X["ec_quarter_diff"].iloc[0] > 2.5
    assert X["ec_quarter_ratio"].iloc[0] > 1.5


def test_baseline_uses_only_previous_readings_and_keeps_row_order():
    rows = pd.concat([one_reading(timestamp=f"2026-03-0{d} 06:00") for d in range(1, 6)], ignore_index=True)
    rows.loc[4, ["ec_lf", "ec_rf", "ec_lr", "ec_rr"]] = [6.0, 6.1, 6.0, 6.2]   # EC jumps on day 5
    shuffled = rows.iloc[[4, 0, 3, 1, 2]].reset_index(drop=True)
    X = build_features(shuffled)
    # Row 0 of the shuffled input is day 5: compared with days 1-4 (mean 5.075) -> ~ +20%
    assert abs(X["ec_vs_baseline"].iloc[0] - 100 * (6.075 - 5.075) / 5.075) < 1e-6
