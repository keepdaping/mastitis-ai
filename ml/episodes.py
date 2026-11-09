"""
Alert / episode rules replayed over historical data - used by training (ml/train.py).

The live service (app/alerts.py) applies the same rules reading by reading, with the same
constants from app/config.py:
  alert    = this reading is high AND the cow's previous reading was high (at most 48 h earlier)
  episode  = opened by an alert reading if none is active; active until 4 consecutive non-high
             readings (auto-resolve). In the live system a farmer can also close it.
  baseline exclusion (ml/features.py): a reading is excluded from the cow's "normal" if it is
             high, or if an episode is active after processing it.

episode_exclusion_flags() replays these rules over one cow's readings in time order, so
training gets the same baseline_exclude flags the live service stores in the database.
"""

import numpy as np
import pandas as pd

from app.config import ALERT_MAX_GAP, AUTO_RESOLVE_AFTER


def episode_exclusion_flags(timestamps: pd.Series, high: pd.Series) -> np.ndarray:
    """For ONE cow's readings sorted by time: True where the reading must be excluded from the
    baseline. Uses only the current and earlier readings (no look-ahead)."""
    times = pd.to_datetime(timestamps).to_numpy()
    is_high = np.asarray(high, dtype=bool)
    flags = np.zeros(len(is_high), dtype=bool)
    active, normal_run = False, 0
    max_gap = np.timedelta64(ALERT_MAX_GAP)
    for i in range(len(is_high)):
        alert = is_high[i] and i > 0 and is_high[i - 1] and (times[i] - times[i - 1]) <= max_gap
        if alert:
            active, normal_run = True, 0  # open (or continue) the episode
        elif active and not is_high[i]:
            normal_run += 1
            if normal_run >= AUTO_RESOLVE_AFTER:  # auto-resolve
                active, normal_run = False, 0
        elif active and is_high[i]:
            normal_run = 0  # a lone high inside an episode resets the count
        flags[i] = is_high[i] or active
    return flags


def flags_for_frame(df: pd.DataFrame, high: pd.Series) -> pd.Series:
    """episode_exclusion_flags() for every cow in a frame (cow_id, timestamp). Same index as df."""
    out = pd.Series(False, index=df.index)
    order = df.sort_values(["cow_id", "timestamp"]).index
    for _, group in df.loc[order].groupby("cow_id", sort=False):
        out.loc[group.index] = episode_exclusion_flags(group["timestamp"], high.loc[group.index])
    return out
