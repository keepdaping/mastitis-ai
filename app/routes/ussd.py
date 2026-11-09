"""
Africa's Talking USSD callback (optional feature).

Africa's Talking POSTs form fields: sessionId, serviceCode, phoneNumber, text.
`text` holds everything the user typed so far, joined by "*" (e.g. "2*C005*3").
We answer in plain text: "CON ..." = show a menu and wait for input, "END ..." = finish.

The farm is identified by the caller's phone number (farms.phone).
  1  My high-risk cows
  2  Enter CMT result   -> cow tag -> score 1-5
  3  Mark alert as treated -> pick the alert
"""
from datetime import date

from fastapi import APIRouter, Form
from fastapi.responses import PlainTextResponse
from sqlalchemy import select

from app.db import SessionLocal
from app.models import ACTIVE_ALERT_STATUSES, Alert, Cow, Farm
from app.routes.dashboard import herd_rows, save_cmt
from app.service import close_alert

router = APIRouter()
MAX_ITEMS = 5          # USSD screens are small (~180 characters)


def con(text: str) -> PlainTextResponse:
    return PlainTextResponse("CON " + text)


def end(text: str) -> PlainTextResponse:
    return PlainTextResponse("END " + text)


def find_cow(s, farm: Farm, typed: str) -> Cow | None:
    """Accept the full tag ("F02-C005") or just its last part ("C005")."""
    typed = typed.strip().upper()
    for cow in s.execute(select(Cow).where(Cow.farm_id == farm.id)).scalars():
        tag = cow.tag.upper()
        if tag == typed or tag.endswith("-" + typed):
            return cow
    return None


def active_alerts(s, farm: Farm) -> list[Alert]:
    return s.execute(select(Alert).join(Cow).where(Cow.farm_id == farm.id,
                                                   Alert.status.in_(ACTIVE_ALERT_STATUSES))
                     .order_by(Alert.created_at)).scalars().all()[:MAX_ITEMS]


@router.post("/ussd", response_class=PlainTextResponse)
def ussd(sessionId: str = Form(""), serviceCode: str = Form(""), phoneNumber: str = Form(""),
         text: str = Form("")):
    steps = [t for t in text.split("*")] if text else []
    with SessionLocal() as s:
        farm = s.execute(select(Farm).where(Farm.phone == phoneNumber.replace(" ", ""))).scalar_one_or_none()
        if farm is None:
            return end("This phone number is not registered with MastitisAI.")

        if not steps:
            return con(f"MastitisAI - {farm.name}\n1. My high-risk cows\n2. Enter CMT result\n3. Mark alert as treated")

        # 1 - high-risk cows (latest reading is high)
        if steps[0] == "1":
            high = [r for r in herd_rows(s, farm.id) if r["risk_level"] == "high"][:MAX_ITEMS]
            if not high:
                return end("No high-risk cows right now.")
            return end("High risk:\n" + "\n".join(f"{r['tag']} ({r['risk_score']:.2f})" for r in high)
                       + "\nDo a CMT test today.")

        # 2 - CMT result: tag -> score
        if steps[0] == "2":
            if len(steps) == 1:
                return con("Enter cow tag (e.g. C005):")
            cow = find_cow(s, farm, steps[1])
            if cow is None:
                return end(f"Cow {steps[1]} not found on {farm.name}.")
            if len(steps) == 2:
                return con(f"CMT score for {cow.tag}:\n1 Negative\n2 Trace\n3 Weak +\n4 Distinct ++\n5 Strong +++")
            if steps[2] not in {"1", "2", "3", "4", "5"}:
                return end("Score must be 1 to 5.")
            reading = save_cmt(s, cow, date.today(), int(steps[2]) - 1)     # stored 0-4
            if reading is None:
                return end(f"No milking recorded today for {cow.tag}. Not saved.")
            return end(f"Saved: {cow.tag} CMT {steps[2]}. Thank you.")

        # 3 - mark an open alert as treated
        if steps[0] == "3":
            alerts = active_alerts(s, farm)
            if not alerts:
                return end("No open alerts.")
            if len(steps) == 1:
                lines = [f"{i}. {a.cow.tag} since {a.created_at:%d %b}" for i, a in enumerate(alerts, start=1)]
                return con("Mark as treated:\n" + "\n".join(lines))
            if not steps[1].isdigit() or not 1 <= int(steps[1]) <= len(alerts):
                return end("Invalid choice.")
            alert = alerts[int(steps[1]) - 1]
            close_alert(s, alert, "treated", (alert.note + "; " if alert.note else "") + "marked treated via USSD")
            s.commit()
            return end(f"Alert for {alert.cow.tag} marked as treated.")

    return end("Invalid choice.")
