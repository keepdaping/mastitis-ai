"""
SMS message templates, keyed by language.

"en" is complete. "lg" (Luganda) is a placeholder: any missing template falls back to
English until translations are added - just fill in the strings below, keeping the
same {placeholders}.

Messages must fit in ONE SMS (160 characters); render() enforces this.
"""
SMS_MAX = 160

TEMPLATES = {
    "en": {
        "high_risk": "MastitisAI: Cow {cow} (Farm {farm}) HIGH risk.{quarter} Do CMT test today and call your vet.",
        "quarter": " {quarter} quarter.",
    },
    "lg": {
        # TODO: Luganda translations (reviewed by a native speaker / extension officer).
        # Until then these are None and render() uses English.
        "high_risk": None,
        "quarter": None,
    },
}


def _template(key: str, language: str) -> str:
    return TEMPLATES.get(language, {}).get(key) or TEMPLATES["en"][key]


def render(key: str, language: str = "en", **values) -> str:
    """Fill a template; shorten the farm name if needed so the SMS stays <= 160 characters."""
    if values.get("quarter"):
        values["quarter"] = _template("quarter", language).format(quarter=values["quarter"])
    else:
        values["quarter"] = ""
    text = _template(key, language).format(**values)
    while len(text) > SMS_MAX and len(values.get("farm", "")) > 3:   # trim the farm name first
        values["farm"] = values["farm"][:-2] + "."
        text = _template(key, language).format(**values)
    return text[:SMS_MAX]
