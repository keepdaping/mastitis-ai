"""
SMS alerts.

send_sms(to, message) uses the provider chosen by SMS_PROVIDER in .env:
  console        (default) print the SMS - safe for development and the panel demo
  africastalking Africa's Talking (username from AT_USERNAME, default "sandbox"; key AT_API_KEY)

notify_new_alert() is called ONLY when a new alert episode is created. It also enforces a
safety limit of one SMS per cow per 12 hours, and logs every attempt in sms_log
(sent / failed / rate_limited / no_phone).
"""
import os
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Alert, SmsLog
from app.sms_templates import render

RATE_LIMIT = timedelta(hours=12)
QUARTER_WORDS = {"LF": "Left front", "RF": "Right front", "LR": "Left rear", "RR": "Right rear"}


def provider_name() -> str:
    return os.getenv("SMS_PROVIDER", "console").strip().lower()


def send_sms(to: str, message: str) -> tuple[str, str]:
    """Send one SMS. Returns (status, provider_response); status is 'sent' or 'failed'."""
    provider = provider_name()
    if provider == "africastalking":
        try:
            import africastalking                      # imported only when used
            africastalking.initialize(os.getenv("AT_USERNAME", "sandbox"), os.environ["AT_API_KEY"])
            sender = os.getenv("AT_SENDER_ID") or None
            resp = africastalking.SMS.send(message, [to], sender_id=sender)
            recipients = resp.get("SMSMessageData", {}).get("Recipients", [])
            ok = bool(recipients) and all(r.get("status") == "Success" for r in recipients)
            return ("sent" if ok else "failed"), str(resp)
        except Exception as e:                         # network error, bad key, ...
            return "failed", f"{type(e).__name__}: {e}"
    # console provider (default)
    print(f"[SMS -> {to}] {message}", flush=True)
    return "sent", "console: printed"


def _log(session: Session, **fields) -> SmsLog:
    entry = SmsLog(provider=provider_name(), **fields)
    session.add(entry)
    return entry


def recently_sent(session: Session, cow_id: int, when: datetime) -> bool:
    """True if this cow already got an SMS in the 12 hours before `when`."""
    q = select(SmsLog.id).where(SmsLog.cow_id == cow_id, SmsLog.status == "sent",
                                SmsLog.created_at > when - RATE_LIMIT, SmsLog.created_at <= when)
    return session.execute(q.limit(1)).first() is not None


def alert_message(alert: Alert, quarter: str | None) -> str:
    cow, farm = alert.cow, alert.cow.farm
    return render("high_risk", os.getenv("SMS_LANGUAGE", "en"),
                  cow=cow.tag, farm=(farm.name if farm else "unknown"),
                  quarter=QUARTER_WORDS.get(quarter or ""))


def notify_new_alert(session: Session, alert: Alert, quarter: str | None, when: datetime) -> SmsLog:
    """SMS the farm about a NEW alert episode. `when` = time of the milking that raised it."""
    message = alert_message(alert, quarter)
    phone = alert.cow.farm.phone if alert.cow.farm else None
    base = dict(cow_id=alert.cow_id, alert_id=alert.id, to_phone=phone, message=message, created_at=when)
    if not phone:
        return _log(session, status="no_phone", provider_response="farm has no phone number", **base)
    if recently_sent(session, alert.cow_id, when):
        return _log(session, status="rate_limited", provider_response="already sent to this cow in the last 12 h",
                    **base)
    status, response = send_sms(phone, message)
    return _log(session, status=status, provider_response=response, **base)
