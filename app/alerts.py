"""
Alert episodes (app/ layer): the rules that turn per-reading alerts into one alert + one SMS.

Used by app/service.py (every new reading), the dashboard (farmer marks checked / treated /
false alarm), USSD and the seed script. The same rules are replayed over historical data by
ml/episodes.py for training; the constants live in app/config.py.

  alert=true and no active episode  -> new alerts row + SMS
  alert=true and an active episode  -> attach the reading to it (no SMS)
  4 consecutive non-high readings   -> episode auto-resolves ("resolved_auto")
  closed by the farmer (treated / false alarm) -> no new episode until the cow has had at
  least one NON-high reading after the closure ("re-arm")
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import ACTIVE_ALERT_STATUSES, ALERT_MAX_GAP, AUTO_RESOLVE_AFTER, CLOSING_STATUSES
from app.models import Alert, Cow, Reading
from app.notify import notify_new_alert


def is_alert(level: str, timestamp: datetime, previous: Reading | None) -> bool:
    """Contract rule: this AND the cow's previous reading are both 'high', at most 48 h apart."""
    return (
        level == "high"
        and previous is not None
        and previous.risk_level == "high"
        and timestamp - previous.timestamp <= ALERT_MAX_GAP
    )


def latest_reading(session: Session, cow_id: int) -> Reading | None:
    """The cow's most recent reading, or None."""
    return session.execute(
        select(Reading).where(Reading.cow_id == cow_id).order_by(Reading.timestamp.desc()).limit(1)
    ).scalar_one_or_none()


def active_alert(session: Session, cow_id: int) -> Alert | None:
    """The cow's open or checked episode, if any."""
    query = (
        select(Alert).where(Alert.cow_id == cow_id, Alert.status.in_(ACTIVE_ALERT_STATUSES)).order_by(Alert.id.desc())
    )
    return session.execute(query.limit(1)).scalar_one_or_none()


def close_alert(
    session: Session, alert: Alert, status: str, note: str | None = None, when: datetime | None = None
) -> None:
    """Farmer closes an episode (treated / false_alarm) or acknowledges it (checked).
    Used by the dashboard, USSD and the seed script."""
    alert.status = status
    if note:
        alert.note = note
    if status in CLOSING_STATUSES:
        alert.closed_at = when or datetime.now().replace(microsecond=0)
        last = latest_reading(session, alert.cow_id)
        alert.closed_after_reading_id = last.id if last else None
    else:
        alert.closed_at = None


def _last_farmer_closed(session: Session, cow_id: int) -> Alert | None:
    return session.execute(
        select(Alert)
        .where(Alert.cow_id == cow_id, Alert.status.in_(CLOSING_STATUSES), Alert.closed_after_reading_id.is_not(None))
        .order_by(Alert.id.desc())
        .limit(1)
    ).scalar_one_or_none()


def rearmed(session: Session, cow_id: int) -> bool:
    """False while the cow's last farmer-closed episode has not been followed by a non-high
    reading: the cow is still recovering from that episode, so it must not open a new one."""
    closed = _last_farmer_closed(session, cow_id)
    if closed is None:
        return True
    since = session.get(Reading, closed.closed_after_reading_id).timestamp
    normal_after = session.execute(
        select(Reading.id)
        .where(Reading.cow_id == cow_id, Reading.timestamp > since, Reading.risk_level != "high")
        .limit(1)
    ).first()
    return normal_after is not None


def pending_closed_status(session: Session, cow_id: int) -> str | None:
    """'treated' / 'false_alarm' if the cow's last farmer-closed alert hasn't re-armed yet."""
    if rearmed(session, cow_id):
        return None
    return session.execute(
        select(Alert.status)
        .where(Alert.cow_id == cow_id, Alert.status.in_(CLOSING_STATUSES))
        .order_by(Alert.id.desc())
        .limit(1)
    ).scalar_one_or_none()


def _open_episode(session: Session, cow: Cow, reading: Reading, quarter: str | None) -> Alert:
    """New alerts row + SMS to the farm."""
    alert = Alert(
        cow=cow,
        reading_id=reading.id,
        last_reading_id=reading.id,
        reading_count=1,
        created_at=reading.timestamp,
        status="open",
    )
    session.add(alert)
    session.flush()  # get alert.id for the SMS log
    notify_new_alert(session, alert, quarter, when=reading.timestamp)
    return alert


def _auto_resolve_if_recovered(session: Session, cow: Cow, current: Alert, reading: Reading) -> None:
    """Close the episode after AUTO_RESOLVE_AFTER consecutive non-high readings."""
    since = session.get(Reading, current.last_reading_id).timestamp
    recent = (
        session.execute(
            select(Reading.risk_level)
            .where(Reading.cow_id == cow.id, Reading.timestamp > since)
            .order_by(Reading.timestamp.desc())
            .limit(AUTO_RESOLVE_AFTER)
        )
        .scalars()
        .all()
    )
    if len(recent) == AUTO_RESOLVE_AFTER and all(level != "high" for level in recent):
        current.status = "resolved_auto"
        current.closed_at = reading.timestamp
        current.note = f"Auto-resolved after {AUTO_RESOLVE_AFTER} consecutive non-high readings"


def update_episode(session: Session, cow: Cow, reading: Reading, quarter: str | None) -> Alert | None:
    """Create / extend / auto-resolve the cow's alert episode. Returns a NEW alert, if one was made."""
    current = active_alert(session, cow.id)
    if reading.alert:
        if current is None and not rearmed(session, cow.id):
            return None  # still high after treatment: no new SMS
        if current is None:
            return _open_episode(session, cow, reading, quarter)
        if current.last_reading_id != reading.id:  # ongoing episode -> attach
            current.last_reading_id = reading.id
            current.reading_count += 1
        return None
    if current is not None and reading.risk_level != "high":
        _auto_resolve_if_recovered(session, cow, current, reading)
    return None
