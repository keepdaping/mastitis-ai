"""
Dashboard data (app/ layer): the queries and page contexts behind api/routes/dashboard.py.

Each *_page() function returns the dict a Jinja2 template needs; the JSON helpers return what
the /api/... endpoints send. Route handlers only parse input, call one of these, and render.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.alerts import close_alert, latest_reading, pending_closed_status
from app.config import ACTIVE_ALERT_STATUSES, CMT_FIELD_SCALE
from app.models import Alert, Cow, Farm, Reading, SmsLog
from app.notify import provider_name
from app.predictor import get_predictor, medium_floor

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "reports" / "metrics.json"
REPORT_IMAGES = ["confusion_matrix.png", "shap_summary.png", "roc_curves.png"]
COW_PAGE_READINGS = 60  # reading-history table on the cow page
CLOSED_ALERTS_SHOWN = 50
CMT_LABELS_SHOWN = 20
SMS_LOG_SHOWN = 200
API_ALERTS_LIMIT = 500
SMS_CARD_WINDOW = timedelta(days=7)  # "SMS sent this week"


def fmt_time(timestamp: datetime | None) -> str:
    """'09 Nov 06:00', or '-' when there is no time."""
    return timestamp.strftime("%d %b %H:%M") if timestamp else "-"


# --------------------------------------------------------------------------- alerts
def alert_json(alert: Alert) -> dict:
    """An alert episode, with its length (first -> latest alert reading)."""
    return {
        "id": alert.id,
        "cow_id": alert.cow_id,
        "cow_tag": alert.cow.tag,
        "farm": alert.cow.farm.name if alert.cow.farm else None,
        "status": alert.status,
        "note": alert.note,
        "first_reading": alert.first_reading.timestamp.isoformat(),
        "latest_reading": alert.last_reading.timestamp.isoformat(),
        "reading_count": alert.reading_count,
        "episode_hours": round(alert.episode_hours, 1),
        "latest_risk": alert.last_reading.risk_score,
        "closed_at": alert.closed_at.isoformat() if alert.closed_at else None,
    }


def alert_state(has_active_alert: bool, latest_level: str | None, closed_status: str | None = None) -> str | None:
    """What the Alert column shows for a cow:
       open_alert  - active alert episode and the latest reading is still high   (red)
       recovering  - active alert episode, but the latest reading is not high    (grey-blue)
       treated / false_alarm - farmer closed the alert and the cow is still high;
                     no new alert until a normal reading (see app/alerts.rearmed)
       watch       - latest reading is high but no alert yet (first high reading) (amber)
       None        - nothing to show
    closed_status: status of the cow's last farmer-closed alert if it has NOT re-armed yet."""
    if has_active_alert:
        return "open_alert" if latest_level == "high" else "recovering"
    if latest_level == "high":
        return closed_status or "watch"
    return None


def list_alerts(session: Session, status: str) -> list[dict]:
    """Alert episodes for GET /api/alerts: 'active' (open + checked) or 'all'."""
    query = select(Alert).order_by(Alert.created_at.desc())
    if status == "active":
        query = query.where(Alert.status.in_(ACTIVE_ALERT_STATUSES))
    return [alert_json(alert) for alert in session.execute(query.limit(API_ALERTS_LIMIT)).scalars()]


def update_alert(session: Session, alert_id: int, status: str, note: str | None) -> dict | None:
    """Farmer marks an alert checked / treated / false_alarm. None if the alert doesn't exist.
    treated / false_alarm close the episode; a NEW one can start once the cow has had a normal
    reading again. False alarms are kept (not deleted) as negative examples for retraining."""
    alert = session.get(Alert, alert_id)
    if alert is None:
        return None
    close_alert(session, alert, status, note)
    session.commit()
    return alert_json(alert)


# --------------------------------------------------------------------------- herd
def _herd_row(session: Session, cow: Cow, active: dict[int, int]) -> dict:
    reading = latest_reading(session, cow.id)
    level = reading.risk_level if reading else None
    closed_status = (
        pending_closed_status(session, cow.id) if (reading and level == "high" and cow.id not in active) else None
    )
    return {
        "id": cow.id,
        "tag": cow.tag,
        "farm_id": cow.farm_id,
        "farm": cow.farm.name if cow.farm else "-",
        "breed": cow.breed,
        "parity": cow.parity,
        "days_in_milk": reading.days_in_milk if reading else None,
        "latest": reading.timestamp.isoformat() if reading else None,
        "latest_fmt": fmt_time(reading.timestamp if reading else None),
        "risk_score": reading.risk_score if reading else None,
        "risk_level": level,
        "active_alert_id": active.get(cow.id),
        "alert_state": alert_state(cow.id in active, level, closed_status),
    }


def herd_rows(session: Session, farm: str | None = None) -> list[dict]:
    """One row per cow with its latest reading, highest risk first."""
    query = select(Cow).order_by(Cow.tag)
    if farm:
        query = query.where(Cow.farm_id == farm)
    active = {
        alert.cow_id: alert.id
        for alert in session.execute(select(Alert).where(Alert.status.in_(ACTIVE_ALERT_STATUSES))).scalars()
    }
    rows = [_herd_row(session, cow, active) for cow in session.execute(query).scalars()]
    rows.sort(key=lambda row: -1 if row["risk_score"] is None else row["risk_score"], reverse=True)
    return rows


def herd_page(session: Session, farm: str | None) -> dict:
    """Herd overview: summary cards (counted from the SAME rows as the table) + cow table."""
    rows = herd_rows(session, farm)
    farm_cows = select(Cow.id).where(Cow.farm_id == farm) if farm else select(Cow.id)
    cards = {
        "total_cows": len(rows),
        "high_today": sum(1 for row in rows if row["risk_level"] == "high"),  # rows showing HIGH
        "open_alerts": sum(1 for row in rows if row["alert_state"] in ("open_alert", "recovering")),
        "sms_week": session.scalar(
            select(func.count(SmsLog.id)).where(
                SmsLog.status == "sent",
                SmsLog.created_at >= datetime.now() - SMS_CARD_WINDOW,
                SmsLog.cow_id.in_(farm_cows),
            )
        ),
    }
    predictor = get_predictor()
    bands = {"medium": medium_floor(predictor.threshold), "high": predictor.threshold} if predictor else None
    farms = session.execute(select(Farm).order_by(Farm.id)).scalars().all()
    return {"rows": rows, "cards": cards, "farms": farms, "farm": farm, "bands": bands, "page": "herd"}


# --------------------------------------------------------------------------- cow
def cow_page(session: Session, cow_id: int) -> dict | None:
    """Cow detail: latest reasons, alert episodes, reading history. None if the cow doesn't exist."""
    cow = session.get(Cow, cow_id)
    if cow is None:
        return None
    readings = (
        session.execute(
            select(Reading).where(Reading.cow_id == cow.id).order_by(Reading.timestamp.desc()).limit(COW_PAGE_READINGS)
        )
        .scalars()
        .all()
    )
    episodes = [
        alert_json(alert)
        for alert in session.execute(select(Alert).where(Alert.cow_id == cow.id).order_by(Alert.id.desc())).scalars()
    ]
    predictor = get_predictor()
    latest = readings[0] if readings else None
    has_active = any(episode["status"] in ACTIVE_ALERT_STATUSES for episode in episodes)
    return {
        "cow": cow,
        "farm": cow.farm,
        "readings": readings,
        "latest": latest,
        "alert_state": alert_state(
            has_active,
            latest.risk_level if latest else None,
            pending_closed_status(session, cow.id) if not has_active else None,
        ),
        "episodes": episodes,
        "threshold": predictor.threshold if predictor else None,
        "cmt_scale": CMT_FIELD_SCALE,
        "page": "herd",
    }


def cow_readings(session: Session, cow_id: int, days: int) -> list[dict] | None:
    """Chart data for the cow page: the last `days` days of readings. None if the cow doesn't exist."""
    cow = session.get(Cow, cow_id)
    if cow is None:
        return None
    last = latest_reading(session, cow.id)
    since = (last.timestamp if last else datetime.now()) - timedelta(days=days)
    readings = (
        session.execute(
            select(Reading).where(Reading.cow_id == cow.id, Reading.timestamp >= since).order_by(Reading.timestamp)
        )
        .scalars()
        .all()
    )
    return [
        {
            "timestamp": reading.timestamp.isoformat(),
            "label": fmt_time(reading.timestamp),
            "day": reading.timestamp.strftime("%d %b"),  # x-axis ticks: dates only
            "risk_score": reading.risk_score,
            "risk_level": reading.risk_level,
            "alert": reading.alert,
            "ec_LF": reading.ec_LF,
            "ec_RF": reading.ec_RF,
            "ec_LR": reading.ec_LR,
            "ec_RR": reading.ec_RR,
            "ph": reading.ph,
            "milk_temp_c": reading.milk_temp_c,
            "milk_yield_l": reading.milk_yield_l,
            "cmt_score": reading.cmt_score,
        }
        for reading in readings
    ]


# --------------------------------------------------------------------------- other pages
def alerts_page(session: Session) -> dict:
    """Active episodes (with action buttons) + recently closed ones."""
    active = (
        session.execute(select(Alert).where(Alert.status.in_(ACTIVE_ALERT_STATUSES)).order_by(Alert.created_at.desc()))
        .scalars()
        .all()
    )
    closed = (
        session.execute(
            select(Alert)
            .where(Alert.status.not_in(ACTIVE_ALERT_STATUSES))
            .order_by(Alert.created_at.desc())
            .limit(CLOSED_ALERTS_SHOWN)
        )
        .scalars()
        .all()
    )
    return {
        "active": [alert_json(alert) for alert in active],
        "closed": [alert_json(alert) for alert in closed],
        "page": "alerts",
    }


def cmt_page(session: Session, message: str | None = None, ok: bool = True) -> dict:
    """CMT entry form + the latest labels."""
    cows = session.execute(select(Cow).order_by(Cow.tag)).scalars().all()
    labels = (
        session.execute(
            select(Reading)
            .where(Reading.cmt_score.is_not(None))
            .order_by(Reading.timestamp.desc())
            .limit(CMT_LABELS_SHOWN)
        )
        .scalars()
        .all()
    )
    return {
        "cows": cows,
        "labels": labels,
        "scale": CMT_FIELD_SCALE,
        "today": date.today().isoformat(),
        "message": message,
        "ok": ok,
        "page": "cmt",
    }


def save_cmt(session: Session, cow: Cow, day: date, score: int) -> Reading | None:
    """Attach a CMT result (0-4) to the cow's LAST milking on that day. None if there is none."""
    start = datetime.combine(day, datetime.min.time())
    reading = session.execute(
        select(Reading)
        .where(Reading.cow_id == cow.id, Reading.timestamp >= start, Reading.timestamp < start + timedelta(days=1))
        .order_by(Reading.timestamp.desc())
        .limit(1)
    ).scalar_one_or_none()
    if reading is not None:
        reading.cmt_score = score
        session.commit()
    return reading


def record_cmt(session: Session, cow_id: int, day: date, score: int) -> tuple[str, bool]:
    """Save a CMT result from the /cmt form. Returns (message for the farmer, success)."""
    cow = session.get(Cow, cow_id)
    if cow is None:
        return "Unknown cow.", False
    reading = save_cmt(session, cow, day, score)
    if reading is None:
        return (
            f"No milking recorded for {cow.tag} on {day:%d %b %Y}. The CMT result must belong to a recorded milking."
        ), False
    return (
        f"Saved: {cow.tag}, milking of {fmt_time(reading.timestamp)}, "
        f"CMT {CMT_FIELD_SCALE[score]}. Stored as a training label."
    ), True


def model_page() -> dict:
    """Model version, test-farm metrics and report images."""
    predictor = get_predictor()
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    return {
        "predictor": predictor,
        "metrics": metrics,
        "page": "model",
        "floor": medium_floor(predictor.threshold) if predictor else None,
        "has": {name: (ROOT / "reports" / name).exists() for name in REPORT_IMAGES},
    }


def sms_page(session: Session) -> dict:
    """The SMS log with counts per status."""
    log = (
        session.execute(select(SmsLog).order_by(SmsLog.created_at.desc(), SmsLog.id.desc()).limit(SMS_LOG_SHOWN))
        .scalars()
        .all()
    )
    counts = dict(session.execute(select(SmsLog.status, func.count(SmsLog.id)).group_by(SmsLog.status)).all())
    return {"log": log, "counts": counts, "provider": provider_name(), "page": "sms"}
