"""
SMS alerts (app/ layer). Called by app/alerts.py when a NEW alert episode opens.

send_sms(to, message) uses the provider chosen by SMS_PROVIDER in .env:
  console        (default) print the SMS - safe for development and the panel demo
  africastalking Africa's Talking (username from AT_USERNAME, default "sandbox"; key AT_API_KEY)

notify_new_alert() also enforces a safety limit of one SMS per cow per 12 hours, and logs
every attempt in sms_log (sent / failed / rate_limited / no_phone). Texts: app/sms_templates.py.
"""

import os
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import QUARTER_NAMES, SMS_RATE_LIMIT
from app.models import Alert, SmsLog
from app.sms_templates import render


def provider_name() -> str:
    """The configured SMS provider: 'console' (default) or 'africastalking'."""
    return os.getenv("SMS_PROVIDER", "console").strip().lower()


def _send_africastalking(to: str, message: str) -> tuple[str, str]:
    try:
        import africastalking  # imported only when used

        africastalking.initialize(os.getenv("AT_USERNAME", "sandbox"), os.environ["AT_API_KEY"])
        sender = os.getenv("AT_SENDER_ID") or None
        response = africastalking.SMS.send(message, [to], sender_id=sender)
        recipients = response.get("SMSMessageData", {}).get("Recipients", [])
        delivered = bool(recipients) and all(recipient.get("status") == "Success" for recipient in recipients)
        return ("sent" if delivered else "failed"), str(response)
    except Exception as error:  # network error, bad key, ...
        return "failed", f"{type(error).__name__}: {error}"


def send_sms(to: str, message: str) -> tuple[str, str]:
    """Send one SMS. Returns (status, provider_response); status is 'sent' or 'failed'."""
    if provider_name() == "africastalking":
        return _send_africastalking(to, message)
    print(f"[SMS -> {to}] {message}", flush=True)  # console provider (default)
    return "sent", "console: printed"


def _log(session: Session, **fields: object) -> SmsLog:
    entry = SmsLog(provider=provider_name(), **fields)
    session.add(entry)
    return entry


def recently_sent(session: Session, cow_id: int, when: datetime) -> bool:
    """True if this cow already got an SMS in the 12 hours before `when`."""
    query = select(SmsLog.id).where(
        SmsLog.cow_id == cow_id,
        SmsLog.status == "sent",
        SmsLog.created_at > when - SMS_RATE_LIMIT,
        SmsLog.created_at <= when,
    )
    return session.execute(query.limit(1)).first() is not None


def alert_message(alert: Alert, quarter: str | None) -> str:
    """The SMS text for an alert (quarter = API key such as 'LR', or None)."""
    cow, farm = alert.cow, alert.cow.farm
    return render(
        "high_risk",
        os.getenv("SMS_LANGUAGE", "en"),
        cow=cow.tag,
        farm=(farm.name if farm else "unknown"),
        quarter=QUARTER_NAMES.get(quarter or ""),
    )


def notify_new_alert(session: Session, alert: Alert, quarter: str | None, when: datetime) -> SmsLog:
    """SMS the farm about a NEW alert episode. `when` = time of the milking that raised it."""
    message = alert_message(alert, quarter)
    phone = alert.cow.farm.phone if alert.cow.farm else None
    base = {"cow_id": alert.cow_id, "alert_id": alert.id, "to_phone": phone, "message": message, "created_at": when}
    if not phone:
        return _log(session, status="no_phone", provider_response="farm has no phone number", **base)
    if recently_sent(session, alert.cow_id, when):
        return _log(
            session, status="rate_limited", provider_response="already sent to this cow in the last 12 h", **base
        )
    status, response = send_sms(phone, message)
    return _log(session, status=status, provider_response=response, **base)
