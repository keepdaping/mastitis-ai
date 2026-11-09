"""
Turn SHAP numbers into short, farmer-friendly reasons and advice.

Related features are grouped (e.g. ec_quarter_diff / ec_quarter_ratio / ec_max all say
"one quarter is higher") so the farmer sees one reason per idea, not three copies.
Each reason reports:
  feature - the group member with the biggest contribution (a real model feature name)
  effect  - the group's combined SHAP contribution, signed, 2 decimals ("+0.21")
  text    - plain English, naming the quarter in words
"""
QUARTER_NAMES = {"lf": "Left front", "rf": "Right front", "lr": "Left rear", "rr": "Right rear"}

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
MIN_EFFECT = 0.005           # smaller contributions would print as "+0.00"


def _text(group: str, up: bool, f: dict, quarter: str | None, has_history: bool = True) -> str:
    """One short sentence per factor. f = feature values, up = pushes risk up.
    has_history=False: the baseline is the herd median, so we must not say "this cow's average"."""
    avg = "this cow's 7-day average" if has_history else "the herd average"
    usual = "this cow's usual" if has_history else "usual"
    if group == "quarter_ec":
        if quarter is None:
            return "Quarter conductivity readings missing (check the sensor)"
        if up:
            return (f"{QUARTER_NAMES[quarter]} quarter conductivity much higher than others "
                    f"(+{f['ec_quarter_diff']:.1f} mS/cm)")
        return "Conductivity similar in all four quarters"
    if group == "ec_level":
        if up and f["ec_vs_baseline"] > 3:
            return f"Conductivity {f['ec_vs_baseline']:.0f}% above {avg}"
        if up:
            return f"Conductivity on the high side ({f['ec_mean']:.1f} mS/cm)"
        return "Conductivity normal for this cow"
    if group == "ph":
        if up and f["ph"] > 6.8:
            return f"Milk pH higher than normal ({f['ph']:.2f}; normal is 6.6-6.8)"
        if up:
            return f"Milk pH at the upper end of normal ({f['ph']:.2f})"
        return f"Milk pH normal ({f['ph']:.2f})"
    if group == "temperature":
        if up and f["temp_vs_baseline"] > 0:
            return f"Milk warmer than {usual} ({f['milk_temp_c']:.1f} C)"
        if up:
            return f"Milk temperature slightly unusual ({f['milk_temp_c']:.1f} C)"
        return f"Milk temperature normal ({f['milk_temp_c']:.1f} C)"
    if group == "yield":
        change = f["yield_vs_baseline"]
        if up and change < -3:
            return f"Milk yield {abs(change):.0f}% below {avg}"
        if up:
            return f"Milk yield unusual ({change:+.0f}% vs {avg})"
        return "Milk yield steady"
    if group == "breed":
        return f"{'Exotic' if f['breed_exotic'] else 'Local'} breed cows here get udder infections more often"
    if group == "grazing":
        system = next(g for g in ("zero", "tethered", "open") if f[f"grazing_{g}"])
        label = {"zero": "Zero-grazing", "tethered": "Tethered", "open": "Open-grazing"}[system]
        return f"{label} farms in this herd have more udder infections - keep the cow's area clean and dry"
    if group == "parity":
        return f"Cows with several calvings are more prone to udder infection (this cow: {int(f['parity'])})"
    if group == "lactation_stage":
        dim = int(f["days_in_milk"])
        if dim < 60:
            return f"Early lactation ({dim} days in milk) is a high-risk time for udder infection"
        return f"Stage of lactation ({dim} days in milk) adds some risk"
    return group


def top_reasons(shap_row, feature_names, feature_values: dict, quarter, risk_level, n=3, has_history=True):
    """The n factors that best explain this prediction.

    low risk  -> factors pulling risk DOWN (reassurance)
    otherwise -> factors pushing risk UP (what to check)
    """
    contrib = dict(zip(feature_names, shap_row))
    want_up = risk_level != "low"
    picked = []
    for group, cols in GROUPS.items():
        total = float(sum(contrib[c] for c in cols))
        if abs(total) < MIN_EFFECT or (total > 0) != want_up:
            continue
        if group in RISK_ONLY and total < 0:
            continue
        # The member feature that contributes most in the group's direction
        main = max(cols, key=lambda c: contrib[c] if total > 0 else -contrib[c])
        picked.append((abs(total), {"feature": main, "effect": f"{total:+.2f}",
                                    "text": _text(group, total > 0, feature_values, quarter, has_history)}))
    picked.sort(key=lambda p: p[0], reverse=True)
    if picked:
        return [r for _, r in picked[:n]]

    # Fallback: nothing clearly pushes in the expected direction (e.g. a "medium" score that is
    # still below the model's average). Never return an empty list: report the single feature
    # with the largest effect and a neutral, honest sentence.
    main = max(contrib, key=lambda c: abs(contrib[c]))
    text = ("Several readings slightly unusual - check this cow again at the next milking"
            if want_up else "Readings normal for this cow")
    return [{"feature": main, "effect": f"{contrib[main]:+.2f}", "text": text}]


ADVICE = {
    "low": "No sign of mastitis. Keep teats clean and dry, dip teats after milking, and do a CMT check every week.",
    "medium": "Watch this cow. Do a CMT test at the next milking, milk her last, and check her readings again tomorrow.",
    "high": ("Likely mastitis{where}. Do a CMT test on this cow today and call your vet. Milk her last, "
             "and do not sell milk from a treated cow until the withdrawal period ends."),
}


def advice(risk_level: str, quarter: str | None) -> str:
    where = f" in the {QUARTER_NAMES[quarter].lower()} quarter" if quarter in QUARTER_NAMES else ""
    return ADVICE[risk_level].format(where=where)
