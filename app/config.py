"""
Shared constants: the rules and limits used across the whole system, defined once.

Risk bands, alert/episode rules, alert statuses, quarter names, valid input ranges, CMT scales,
SMS and USSD limits. Every layer (ml/, app/, api/) imports them from here. This module imports
nothing from the project, so the ml/ layer can use it without depending on the rest of app/.
"""

from datetime import timedelta

# --------------------------------------------------------------------------- udder quarters
QUARTER_KEYS = ["LF", "RF", "LR", "RR"]  # API keys (docs/API_CONTRACT.md)
QUARTER_NAMES = {"LF": "Left front", "RF": "Right front", "LR": "Left rear", "RR": "Right rear"}

# --------------------------------------------------------------------------- risk bands
# low < MEDIUM_FROM <= medium < threshold <= high. The threshold comes from the trained model.
MEDIUM_FROM = 0.25

# --------------------------------------------------------------------------- alert rules
# Shared by training (ml/episodes.py) and the live service (app/alerts.py).
ALERT_MAX_GAP = timedelta(hours=48)  # two highs further apart than this are not "consecutive"
AUTO_RESOLVE_AFTER = 4  # consecutive non-high readings that close an episode
HISTORY_DAYS = 22  # readings loaded per cow: the clean baseline looks back up to 21 days

# Alert statuses: open | checked | treated | false_alarm | resolved_auto.
# "open" and "checked" are ACTIVE: new high readings join that episode instead of opening a new one.
ACTIVE_ALERT_STATUSES = ["open", "checked"]
CLOSING_STATUSES = ("treated", "false_alarm")  # set by the farmer; need a "re-arm" before a new episode
FARMER_STATUSES = ("checked", "treated", "false_alarm")  # what the dashboard / USSD may set

# --------------------------------------------------------------------------- explanations
MAX_REASONS = 3
MIN_REASON_EFFECT = 0.03  # smaller SHAP effects are not shown (but at least one reason always is)

# --------------------------------------------------------------------------- valid input ranges (-> 422)
EC_RANGE = (0, 20)  # mS/cm
PH_RANGE = (5, 9)
TEMP_RANGE = (30, 45)  # deg C
YIELD_RANGE = (0, 60)  # litres per milking
PARITY_RANGE = (1, 15)
DAYS_IN_MILK_RANGE = (0, 700)
CMT_RANGE = (0, 4)  # stored 0-4 = field scale 1-5 minus 1
ID_MAX_LENGTH = 64
NOTE_MAX_LENGTH = 500

# --------------------------------------------------------------------------- CMT (California Mastitis Test)
CMT_POSITIVE_FROM = 2  # cmt_score >= 2 (field CMT >= 3) = subclinical mastitis
CMT_FIELD_SCALE = {
    0: "1 - Negative",
    1: "2 - Trace",
    2: "3 - Weak positive",
    3: "4 - Distinct positive",
    4: "5 - Strong positive",
}
CMT_USSD_MENU = "1 Negative\n2 Trace\n3 Weak +\n4 Distinct ++\n5 Strong +++"

# --------------------------------------------------------------------------- SMS / USSD
SMS_RATE_LIMIT = timedelta(hours=12)  # at most one SMS per cow in this window
SMS_MAX_CHARS = 160  # one SMS
USSD_MAX_ITEMS = 5  # USSD screens are small (~180 characters)
