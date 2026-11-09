"""
Turn SHAP numbers into short, farmer-friendly reasons and advice (app/ layer, used by app/service.py).

Related features are grouped (e.g. ec_quarter_diff / ec_quarter_ratio / ec_max all say
"one quarter is higher") so the farmer sees one reason per idea, not three copies.
Each reason reports:
  feature - the group member with the biggest contribution (a real model feature name)
  effect  - the group's combined SHAP contribution, signed, 2 decimals ("+0.21")
  text    - plain English, naming the quarter in words
"""

from collections.abc import Callable, Sequence

from app.config import MAX_REASONS, MIN_REASON_EFFECT, QUARTER_NAMES

GROUPS = {
    "quarter_ec": ["ec_quarter_diff", "ec_quarter_ratio", "ec_max"],
    "ec_level": ["ec_vs_baseline", "ec_mean"],
    "ph": ["ph"],
    "temperature": ["temp_vs_baseline", "milk_temp_c"],
    "yield": ["yield_vs_baseline", "yield_change_3d", "milk_yield_l"],
    "breed": ["breed_exotic", "breed_local"],
    "grazing": ["grazing_zero", "grazing_tethered", "grazing_open"],
    "parity": ["parity"],
    "lactation_stage": ["days_in_milk"],
}
# Fixed cow attributes are only ever given as reasons for HIGHER risk. With few farms the
# model can learn odd farm-level associations; we never tell a farmer e.g. that
# zero-grazing protects the cow.
RISK_ONLY = {"breed", "grazing", "parity", "lactation_stage"}

# Thresholds that decide the wording of a reason
PH_NORMAL_MAX = 6.8  # healthy milk pH is 6.6-6.8
EC_RISE_WORTH_PERCENT = 3  # "x% above average" only from this rise on
YIELD_DROP_WORTH_PERCENT = -3  # "yield x% below average" only from this drop on
EARLY_LACTATION_DAYS = 60
GRAZING_LABELS = {"zero": "Zero-grazing", "tethered": "Tethered", "open": "Open-grazing"}

ADVICE = {
    "low": "No sign of mastitis. Keep teats clean and dry, dip teats after milking, and do a CMT check every week.",
    "medium": ("Watch this cow. Do a CMT test at the next milking, milk her last, "
               "and check her readings again tomorrow."),
    "high": ("Likely mastitis{where}. Do a CMT test on this cow today and call your vet. Milk her last, "
             "and do not sell milk from a treated cow until the withdrawal period ends."),
}  # fmt: skip


def _quarter_key(quarter: str | None) -> str | None:
    """Accept 'lr' (ml/features) or 'LR' (API) and return the API key."""
    return quarter.upper() if quarter else None


# --------------------------------------------------------------------------- one sentence per factor
# Each function gets: up (pushes risk up?), feature values, quarter key, and the wording context.
def _quarter_ec_text(up: bool, values: dict, quarter: str | None, words: dict) -> str:
    if quarter is None:
        return "Quarter conductivity readings missing (check the sensor)"
    if up:
        return (
            f"{QUARTER_NAMES[quarter]} quarter conductivity much higher than others "
            f"(+{values['ec_quarter_diff']:.1f} mS/cm)"
        )
    return "Conductivity similar in all four quarters"


def _ec_level_text(up: bool, values: dict, quarter: str | None, words: dict) -> str:
    if up and values["ec_vs_baseline"] > EC_RISE_WORTH_PERCENT:
        return f"Conductivity {values['ec_vs_baseline']:.0f}% above {words['average']}"
    if up:
        return f"Conductivity on the high side ({values['ec_mean']:.1f} mS/cm)"
    return "Conductivity normal for this cow"


def _ph_text(up: bool, values: dict, quarter: str | None, words: dict) -> str:
    if up and values["ph"] > PH_NORMAL_MAX:
        return f"Milk pH higher than normal ({values['ph']:.2f}; normal is 6.6-6.8)"
    if up:
        return f"Milk pH at the upper end of normal ({values['ph']:.2f})"
    return f"Milk pH normal ({values['ph']:.2f})"


def _temperature_text(up: bool, values: dict, quarter: str | None, words: dict) -> str:
    if up and values["temp_vs_baseline"] > 0:
        return f"Milk warmer than {words['usual']} ({values['milk_temp_c']:.1f} C)"
    if up:
        return f"Milk temperature slightly unusual ({values['milk_temp_c']:.1f} C)"
    return f"Milk temperature normal ({values['milk_temp_c']:.1f} C)"


def _yield_text(up: bool, values: dict, quarter: str | None, words: dict) -> str:
    change = values["yield_vs_baseline"]
    if up and change < YIELD_DROP_WORTH_PERCENT:
        return f"Milk yield {abs(change):.0f}% below {words['average']}"
    if up:
        return f"Milk yield unusual ({change:+.0f}% vs {words['average']})"
    return "Milk yield steady"


def _breed_text(up: bool, values: dict, quarter: str | None, words: dict) -> str:
    return f"{'Exotic' if values['breed_exotic'] else 'Local'} breed cows here get udder infections more often"


def _grazing_text(up: bool, values: dict, quarter: str | None, words: dict) -> str:
    system = next(name for name in GRAZING_LABELS if values[f"grazing_{name}"])
    return f"{GRAZING_LABELS[system]} farms in this herd have more udder infections - keep the cow's area clean and dry"


def _parity_text(up: bool, values: dict, quarter: str | None, words: dict) -> str:
    return f"Cows with several calvings are more prone to udder infection (this cow: {int(values['parity'])})"


def _lactation_text(up: bool, values: dict, quarter: str | None, words: dict) -> str:
    days = int(values["days_in_milk"])
    if days < EARLY_LACTATION_DAYS:
        return f"Early lactation ({days} days in milk) is a high-risk time for udder infection"
    return f"Stage of lactation ({days} days in milk) adds some risk"


TEXT_BUILDERS: dict[str, Callable[[bool, dict, str | None, dict], str]] = {
    "quarter_ec": _quarter_ec_text,
    "ec_level": _ec_level_text,
    "ph": _ph_text,
    "temperature": _temperature_text,
    "yield": _yield_text,
    "breed": _breed_text,
    "grazing": _grazing_text,
    "parity": _parity_text,
    "lactation_stage": _lactation_text,
}


def reason_text(group: str, up: bool, values: dict, quarter: str | None, has_history: bool = True) -> str:
    """One short sentence for a factor group. has_history=False: the baseline is the herd median,
    so we must not say "this cow's average"."""
    words = {
        "average": "this cow's 7-day average" if has_history else "the herd average",
        "usual": "this cow's usual" if has_history else "usual",
    }
    builder = TEXT_BUILDERS.get(group)
    return builder(up, values, _quarter_key(quarter), words) if builder else group


# --------------------------------------------------------------------------- picking the reasons
def _group_candidates(
    contrib: dict, want_up: bool, values: dict, quarter: str | None, has_history: bool
) -> list[tuple[float, dict]]:
    """(size, reason) for every factor group that pushes in the wanted direction, biggest first."""
    picked = []
    for group, cols in GROUPS.items():
        total = float(sum(contrib[col] for col in cols))
        if total == 0 or (total > 0) != want_up:
            continue
        if group in RISK_ONLY and total < 0:
            continue
        # The member feature that contributes most in the group's direction
        main = max(cols, key=lambda col, total=total: contrib[col] if total > 0 else -contrib[col])
        picked.append(
            (
                abs(total),
                {
                    "feature": main,
                    "effect": f"{total:+.2f}",
                    "text": reason_text(group, total > 0, values, quarter, has_history),
                },
            )
        )
    picked.sort(key=lambda item: item[0], reverse=True)
    return picked


def top_reasons(
    shap_row: Sequence[float],
    feature_names: list[str],
    feature_values: dict,
    quarter: str | None,
    risk_level: str,
    n: int = MAX_REASONS,
    has_history: bool = True,
) -> list[dict]:
    """The n factors that best explain this prediction (always at least one).

    low risk  -> factors pulling risk DOWN (reassurance)
    otherwise -> factors pushing risk UP (what to check)
    Reasons with |effect| < MIN_REASON_EFFECT are left out unless nothing else is available.
    """
    contrib = dict(zip(feature_names, shap_row, strict=True))
    want_up = risk_level != "low"
    picked = _group_candidates(contrib, want_up, feature_values, quarter, has_history)
    strong = [reason for size, reason in picked if size >= MIN_REASON_EFFECT]
    if strong:
        return strong[:n]
    if picked:  # all below MIN_REASON_EFFECT: keep only the biggest one
        return [picked[0][1]]

    # Fallback: nothing clearly pushes in the expected direction (e.g. a "medium" score that is
    # still below the model's average). Never return an empty list: report the single feature
    # with the largest effect and a neutral, honest sentence.
    main = max(contrib, key=lambda col: abs(contrib[col]))
    text = (
        "Several readings slightly unusual - check this cow again at the next milking"
        if want_up
        else "Readings normal for this cow"
    )
    return [{"feature": main, "effect": f"{contrib[main]:+.2f}", "text": text}]


def advice(risk_level: str, quarter: str | None) -> str:
    """What the farmer should do, by risk level (names the quarter for high risk)."""
    key = _quarter_key(quarter)
    where = f" in the {QUARTER_NAMES[key].lower()} quarter" if key in QUARTER_NAMES else ""
    return ADVICE[risk_level].format(where=where)
