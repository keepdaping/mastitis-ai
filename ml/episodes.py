"""
Alert / episode rules shared by training (ml/train.py) and the live service (app/service.py).

Per reading (docs/API_CONTRACT.md):
  alert = this reading is high AND the cow's previous reading was high (at most 48 h earlier)
Episodes (dashboard):
  an alert reading opens an episode if none is active; it stays active until
  4 consecutive non-high readings (auto-resolve). In the live system a farmer can also close it.
Baseline exclusion (ml/features.py):
  a reading is excluded from the cow's "normal" if it is high, or if an episode is
  active after processing it.

episode_exclusion_flags() replays these rules over a cow's readings in time order. Training
uses it to create the same baseline_exclude flags the live service stores in the database.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

ALERT_MAX_GAP = timedelta(hours=48)   # 2 highs further apart than this are not "consecutive"
AUTO_RESOLVE_AFTER = 4                # consecutive non-high readings that close an episode


def episode_exclusion_flags(timestamps: pd.Series, high: pd.Series) -> np.ndarray:
    """For ONE cow's readings sorted by time: True where the reading must be excluded from the
    baseline. Uses only the current and earlier readings (no look-ahead)."""
    ts = pd.to_datetime(timestamps).to_numpy()
    hi = np.asarray(high, dtype=bool)
    flags = np.zeros(len(hi), dtype=bool)
    active, normal_run = False, 0
    for i in range(len(hi)):
        alert = hi[i] and i > 0 and hi[i - 1] and (ts[i] - ts[i - 1]) <= np.timedelta64(ALERT_MAX_GAP)
        if alert:
            active, normal_run = True, 0          # open (or continue) the episode
        elif active and not hi[i]:
            normal_run += 1
            if normal_run >= AUTO_RESOLVE_AFTER:  # auto-resolve
                active, normal_run = False, 0
        elif active and hi[i]:
            normal_run = 0                        # a lone high inside an episode resets the count
        flags[i] = hi[i] or active
    return flags


def flags_for_frame(df: pd.DataFrame, high: pd.Series) -> pd.Series:
    """episode_exclusion_flags() for every cow in a frame (cow_id, timestamp). Same index as df."""
    out = pd.Series(False, index=df.index)
    order = df.sort_values(["cow_id", "timestamp"]).index
    for _, g in df.loc[order].groupby("cow_id", sort=False):
        out.loc[g.index] = episode_exclusion_flags(g["timestamp"], high.loc[g.index])
    return out
