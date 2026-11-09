"""
Africa's Talking USSD menu (app/ layer, optional feature). The route is api/routes/ussd.py.

Africa's Talking POSTs: sessionId, serviceCode, phoneNumber, text. `text` holds everything the
user typed so far, joined by "*" (e.g. "2*C005*3"). We answer in plain text:
"CON ..." = show a menu and wait for input, "END ..." = finish.

The farm is identified by the caller's phone number (farms.phone).
  1  My high-risk cows
  2  Enter CMT result      -> cow tag -> score 1-5
  3  Mark alert as treated -> pick the alert
"""

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.alerts import close_alert
from app.config import ACTIVE_ALERT_STATUSES, CMT_USSD_MENU, USSD_MAX_ITEMS
from app.dashboard import herd_rows, save_cmt
from app.models import Alert, Cow, Farm

VALID_FIELD_SCORES = {"1", "2", "3", "4", "5"}


def con(text: str) -> str:
    """A menu screen that waits for input."""
    return "CON " + text


def end(text: str) -> str:
    """A final screen."""
    return "END " + text


def find_cow(session: Session, farm: Farm, typed: str) -> Cow | None:
    """Accept the full tag ("F02-C005") or just its last part ("C005")."""
    typed = typed.strip().upper()
    for cow in session.execute(select(Cow).where(Cow.farm_id == farm.id)).scalars():
        tag = cow.tag.upper()
        if tag == typed or tag.endswith("-" + typed):
            return cow
    return None


def active_alerts(session: Session, farm: Farm) -> list[Alert]:
    """The farm's open/checked alerts, oldest first (at most USSD_MAX_ITEMS)."""
    return (
        session.execute(
            select(Alert)
            .join(Cow)
            .where(Cow.farm_id == farm.id, Alert.status.in_(ACTIVE_ALERT_STATUSES))
            .order_by(Alert.created_at)
        )
        .scalars()
        .all()[:USSD_MAX_ITEMS]
    )


def _high_risk_cows(session: Session, farm: Farm) -> str:
    high = [row for row in herd_rows(session, farm.id) if row["risk_level"] == "high"][:USSD_MAX_ITEMS]
    if not high:
        return end("No high-risk cows right now.")
    return end(
        "High risk:\n" + "\n".join(f"{row['tag']} ({row['risk_score']:.2f})" for row in high) + "\nDo a CMT test today."
    )


def _cmt_result(session: Session, farm: Farm, steps: list[str]) -> str:
    if len(steps) == 1:
        return con("Enter cow tag (e.g. C005):")
    cow = find_cow(session, farm, steps[1])
    if cow is None:
        return end(f"Cow {steps[1]} not found on {farm.name}.")
    if len(steps) == 2:
        return con(f"CMT score for {cow.tag}:\n{CMT_USSD_MENU}")
    if steps[2] not in VALID_FIELD_SCORES:
        return end("Score must be 1 to 5.")
    reading = save_cmt(session, cow, date.today(), int(steps[2]) - 1)  # field 1-5 -> stored 0-4
    if reading is None:
        return end(f"No milking recorded today for {cow.tag}. Not saved.")
    return end(f"Saved: {cow.tag} CMT {steps[2]}. Thank you.")


def _mark_treated(session: Session, farm: Farm, steps: list[str]) -> str:
    alerts = active_alerts(session, farm)
    if not alerts:
        return end("No open alerts.")
    if len(steps) == 1:
        lines = [
            f"{number}. {alert.cow.tag} since {alert.created_at:%d %b}" for number, alert in enumerate(alerts, start=1)
        ]
        return con("Mark as treated:\n" + "\n".join(lines))
    if not steps[1].isdigit() or not 1 <= int(steps[1]) <= len(alerts):
        return end("Invalid choice.")
    alert = alerts[int(steps[1]) - 1]
    close_alert(session, alert, "treated", (alert.note + "; " if alert.note else "") + "marked treated via USSD")
    session.commit()
    return end(f"Alert for {alert.cow.tag} marked as treated.")


MENU = {"1": lambda session, farm, steps: _high_risk_cows(session, farm), "2": _cmt_result, "3": _mark_treated}


def ussd_reply(session: Session, phone_number: str, text: str) -> str:
    """The reply ("CON ..." / "END ...") to one USSD request."""
    steps = text.split("*") if text else []
    farm = session.execute(select(Farm).where(Farm.phone == phone_number.replace(" ", ""))).scalar_one_or_none()
    if farm is None:
        return end("This phone number is not registered with MastitisAI.")
    if not steps:
        return con(f"MastitisAI - {farm.name}\n1. My high-risk cows\n2. Enter CMT result\n3. Mark alert as treated")
    handler = MENU.get(steps[0])
    return handler(session, farm, steps) if handler else end("Invalid choice.")
