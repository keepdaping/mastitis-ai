"""
Farmer / vet dashboard (server-rendered Jinja2 + Bootstrap) and the small JSON API it uses.

Pages:  /  /cows/{id}  /alerts  /cmt  /model  /demo  /sms
JSON:   GET /api/cows  GET /api/cows/{id}/readings  GET /api/alerts  PATCH /api/alerts/{id}
        POST /api/demo/start  GET /api/demo/status
"""
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import demo
from app.db import SessionLocal
from app.models import ACTIVE_ALERT_STATUSES, Alert, Cow, Farm, Reading, SmsLog
from app.notify import provider_name
from app.predictor import get_predictor, medium_floor
from app.service import CLOSING_STATUSES, close_alert, rearmed

ROOT = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=ROOT / "templates")
router = APIRouter()

# CMT scores are stored 0-4; farmers know the 1-5 field scale
CMT_FIELD_SCALE = {0: "1 - Negative", 1: "2 - Trace", 2: "3 - Weak positive",
                   3: "4 - Distinct positive", 4: "5 - Strong positive"}


def fmt_time(ts: datetime | None) -> str:
    return ts.strftime("%d %b %H:%M") if ts else "-"


templates.env.filters["t"] = fmt_time


# ------------------------------------------------------------------ queries shared by pages + JSON
def latest_reading(s: Session, cow_id: int) -> Reading | None:
    return s.execute(select(Reading).where(Reading.cow_id == cow_id)
                     .order_by(Reading.timestamp.desc()).limit(1)).scalar_one_or_none()


def alert_json(a: Alert) -> dict:
    """An alert episode, with its length (first -> latest alert reading)."""
    return {"id": a.id, "cow_id": a.cow_id, "cow_tag": a.cow.tag,
            "farm": a.cow.farm.name if a.cow.farm else None, "status": a.status, "note": a.note,
            "first_reading": a.first_reading.timestamp.isoformat(),
            "latest_reading": a.last_reading.timestamp.isoformat(),
            "reading_count": a.reading_count, "episode_hours": round(a.episode_hours, 1),
            "latest_risk": a.last_reading.risk_score,
            "closed_at": a.closed_at.isoformat() if a.closed_at else None}


def alert_state(has_active_alert: bool, latest_level: str | None, closed_status: str | None = None) -> str | None:
    """What the Alert column shows for a cow:
       open_alert  - active alert episode and the latest reading is still high   (red)
       recovering  - active alert episode, but the latest reading is not high    (grey-blue)
       treated / false_alarm - farmer closed the alert and the cow is still high;
                     no new alert until a normal reading (see app/service.rearmed)
       watch       - latest reading is high but no alert yet (first high reading) (amber)
       None        - nothing to show
    closed_status: status of the cow's last farmer-closed alert if it has NOT re-armed yet."""
    if has_active_alert:
        return "open_alert" if latest_level == "high" else "recovering"
    if latest_level == "high":
        return closed_status or "watch"
    return None


def pending_closed_status(s: Session, cow_id: int) -> str | None:
    """'treated' / 'false_alarm' if the cow's last farmer-closed alert hasn't re-armed yet."""
    if rearmed(s, cow_id):
        return None
    return s.execute(select(Alert.status).where(Alert.cow_id == cow_id, Alert.status.in_(CLOSING_STATUSES))
                     .order_by(Alert.id.desc()).limit(1)).scalar_one_or_none()


def herd_rows(s: Session, farm: str | None = None) -> list[dict]:
    """One row per cow with its latest reading, highest risk first."""
    q = select(Cow).order_by(Cow.tag)
    if farm:
        q = q.where(Cow.farm_id == farm)
    active = {a.cow_id: a.id for a in s.execute(
        select(Alert).where(Alert.status.in_(ACTIVE_ALERT_STATUSES))).scalars()}
    rows = []
    for cow in s.execute(q).scalars():
        r = latest_reading(s, cow.id)
        rows.append({
            "id": cow.id, "tag": cow.tag, "farm_id": cow.farm_id,
            "farm": cow.farm.name if cow.farm else "-", "breed": cow.breed, "parity": cow.parity,
            "days_in_milk": r.days_in_milk if r else None,
            "latest": r.timestamp.isoformat() if r else None, "latest_fmt": fmt_time(r.timestamp if r else None),
            "risk_score": r.risk_score if r else None, "risk_level": r.risk_level if r else None,
            "active_alert_id": active.get(cow.id),
            "alert_state": alert_state(
                cow.id in active, r.risk_level if r else None,
                pending_closed_status(s, cow.id) if (r and r.risk_level == "high" and cow.id not in active) else None)})
    rows.sort(key=lambda x: -1 if x["risk_score"] is None else x["risk_score"], reverse=True)
    return rows


def farm_list(s: Session) -> list[Farm]:
    return s.execute(select(Farm).order_by(Farm.id)).scalars().all()


# ------------------------------------------------------------------ pages
@router.get("/", response_class=HTMLResponse)
def herd_overview(request: Request, farm: str | None = None):
    with SessionLocal() as s:
        rows = herd_rows(s, farm)
        cows_q = select(Cow.id).where(Cow.farm_id == farm) if farm else select(Cow.id)
        # Cards are counted from the SAME rows as the table, so the numbers always match it
        cards = {
            "total_cows": len(rows),
            "high_today": sum(1 for r in rows if r["risk_level"] == "high"),           # rows showing HIGH
            "open_alerts": sum(1 for r in rows if r["alert_state"] in ("open_alert", "recovering")),
            "sms_week": s.scalar(select(func.count(SmsLog.id)).where(
                SmsLog.status == "sent", SmsLog.created_at >= datetime.now() - timedelta(days=7),
                SmsLog.cow_id.in_(cows_q))),
        }
        p = get_predictor()
        bands = {"medium": medium_floor(p.threshold), "high": p.threshold} if p else None
        return templates.TemplateResponse(request, "index.html", {
            "rows": rows, "cards": cards, "farms": farm_list(s), "farm": farm, "bands": bands,
            "page": "herd"})


@router.get("/cows/{cow_id}", response_class=HTMLResponse)
def cow_detail(request: Request, cow_id: int):
    with SessionLocal() as s:
        cow = s.get(Cow, cow_id)
        if cow is None:
            raise HTTPException(404, "Cow not found")
        readings = s.execute(select(Reading).where(Reading.cow_id == cow.id)
                             .order_by(Reading.timestamp.desc()).limit(60)).scalars().all()
        episodes = [alert_json(a) for a in s.execute(
            select(Alert).where(Alert.cow_id == cow.id).order_by(Alert.id.desc())).scalars()]
        p = get_predictor()
        latest = readings[0] if readings else None
        has_active = any(e["status"] in ACTIVE_ALERT_STATUSES for e in episodes)
        return templates.TemplateResponse(request, "cow.html", {
            "cow": cow, "farm": cow.farm, "readings": readings, "latest": latest,
            "alert_state": alert_state(has_active, latest.risk_level if latest else None,
                                       pending_closed_status(s, cow.id) if not has_active else None),
            "episodes": episodes, "threshold": p.threshold if p else None,
            "cmt_scale": CMT_FIELD_SCALE, "page": "herd"})


@router.get("/alerts", response_class=HTMLResponse)
def alerts_page(request: Request):
    with SessionLocal() as s:
        active = s.execute(select(Alert).where(Alert.status.in_(ACTIVE_ALERT_STATUSES))
                           .order_by(Alert.created_at.desc())).scalars().all()
        closed = s.execute(select(Alert).where(Alert.status.not_in(ACTIVE_ALERT_STATUSES))
                           .order_by(Alert.created_at.desc()).limit(50)).scalars().all()
        return templates.TemplateResponse(request, "alerts.html", {
            "active": [alert_json(a) for a in active], "closed": [alert_json(a) for a in closed],
            "page": "alerts"})


@router.get("/cmt", response_class=HTMLResponse)
def cmt_form(request: Request, message: str | None = None, ok: bool = True):
    with SessionLocal() as s:
        cows = s.execute(select(Cow).order_by(Cow.tag)).scalars().all()
        labels = s.execute(select(Reading).where(Reading.cmt_score.is_not(None))
                           .order_by(Reading.timestamp.desc()).limit(20)).scalars().all()
        return templates.TemplateResponse(request, "cmt.html", {
            "cows": cows, "labels": labels, "scale": CMT_FIELD_SCALE, "today": date.today().isoformat(),
            "message": message, "ok": ok, "page": "cmt"})


def save_cmt(s: Session, cow: Cow, day: date, score: int) -> Reading | None:
    """Attach a CMT result (0-4) to the cow's LAST milking on that day. None if there is none."""
    start = datetime.combine(day, datetime.min.time())
    reading = s.execute(select(Reading).where(Reading.cow_id == cow.id, Reading.timestamp >= start,
                                              Reading.timestamp < start + timedelta(days=1))
                        .order_by(Reading.timestamp.desc()).limit(1)).scalar_one_or_none()
    if reading is not None:
        reading.cmt_score = score
        s.commit()
    return reading


@router.post("/cmt", response_class=HTMLResponse)
def cmt_submit(request: Request, cow_id: int = Form(...), day: date = Form(...),
               score: int = Form(..., ge=0, le=4)):
    with SessionLocal() as s:
        cow = s.get(Cow, cow_id)
        if cow is None:
            return cmt_form(request, "Unknown cow.", ok=False)
        reading = save_cmt(s, cow, day, score)
        if reading is None:
            return cmt_form(request, f"No milking recorded for {cow.tag} on {day:%d %b %Y}. "
                                     f"The CMT result must belong to a recorded milking.", ok=False)
        return cmt_form(request, f"Saved: {cow.tag}, milking of {fmt_time(reading.timestamp)}, "
                                 f"CMT {CMT_FIELD_SCALE[score]}. Stored as a training label.")


@router.get("/model", response_class=HTMLResponse)
def model_page(request: Request):
    p = get_predictor()
    metrics_path = ROOT / "reports" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    floor = medium_floor(p.threshold) if p else None
    return templates.TemplateResponse(request, "model.html", {
        "p": p, "metrics": metrics, "floor": floor, "page": "model",
        "has": {f: (ROOT / "reports" / f).exists() for f in
                ["confusion_matrix.png", "shap_summary.png", "roc_curves.png"]}})


@router.get("/demo", response_class=HTMLResponse)
def demo_page(request: Request):
    return templates.TemplateResponse(request, "demo.html", {"page": "demo", "farm": demo.DEMO_FARM})


@router.get("/sms", response_class=HTMLResponse)
def sms_page(request: Request):
    with SessionLocal() as s:
        log = s.execute(select(SmsLog).order_by(SmsLog.created_at.desc(), SmsLog.id.desc())
                        .limit(200)).scalars().all()
        counts = dict(s.execute(select(SmsLog.status, func.count(SmsLog.id)).group_by(SmsLog.status)).all())
        return templates.TemplateResponse(request, "sms.html", {
            "log": log, "counts": counts, "provider": provider_name(), "page": "sms"})


# ------------------------------------------------------------------ JSON API used by the pages
@router.get("/api/cows")
def api_cows(farm: str | None = None):
    with SessionLocal() as s:
        return herd_rows(s, farm)


@router.get("/api/cows/{cow_id}/readings")
def api_cow_readings(cow_id: int, days: int = 30):
    with SessionLocal() as s:
        cow = s.get(Cow, cow_id)
        if cow is None:
            raise HTTPException(404, "Cow not found")
        last = latest_reading(s, cow.id)
        since = (last.timestamp if last else datetime.now()) - timedelta(days=days)
        rows = s.execute(select(Reading).where(Reading.cow_id == cow.id, Reading.timestamp >= since)
                         .order_by(Reading.timestamp)).scalars().all()
        return [{"timestamp": r.timestamp.isoformat(), "label": fmt_time(r.timestamp),
                 "day": r.timestamp.strftime("%d %b"),           # x-axis ticks: dates only
                 "risk_score": r.risk_score, "risk_level": r.risk_level, "alert": r.alert,
                 "ec_LF": r.ec_LF, "ec_RF": r.ec_RF, "ec_LR": r.ec_LR, "ec_RR": r.ec_RR,
                 "ph": r.ph, "milk_temp_c": r.milk_temp_c, "milk_yield_l": r.milk_yield_l,
                 "cmt_score": r.cmt_score} for r in rows]


@router.get("/api/alerts")
def api_alerts(status: Literal["active", "all"] = "active"):
    with SessionLocal() as s:
        q = select(Alert).order_by(Alert.created_at.desc())
        if status == "active":
            q = q.where(Alert.status.in_(ACTIVE_ALERT_STATUSES))
        return [alert_json(a) for a in s.execute(q.limit(500)).scalars()]


class AlertUpdate(BaseModel):
    # Farmers can only set these; "open" is set by the system, "resolved_auto" by auto-close
    status: Literal["checked", "treated", "false_alarm"]
    note: str | None = Field(None, max_length=500)


@router.patch("/api/alerts/{alert_id}")
def api_update_alert(alert_id: int, body: AlertUpdate):
    with SessionLocal() as s:
        a = s.get(Alert, alert_id)
        if a is None:
            raise HTTPException(404, "Alert not found")
        # treated / false_alarm close the episode. A NEW episode can start once the cow has had a
        # normal reading again. False alarms are kept (not deleted) as negative examples for retraining.
        close_alert(s, a, body.status, body.note)
        s.commit()
        return alert_json(a)


@router.post("/api/demo/start")
def api_demo_start(interval: float = 2.0):
    started = demo.start(max(0.0, min(interval, 10.0)))
    return {"started": started, **demo_status()}


@router.get("/api/demo/status")
def demo_status():
    return {k: demo.STATE[k] for k in ["running", "step", "total", "error", "sick_cow", "log"]}
