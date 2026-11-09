"""
Dashboard routes (api/ layer): server-rendered pages (Jinja2 + Bootstrap) and the small JSON API they use.

Pages:  /  /cows/{id}  /alerts  /cmt  /model  /demo  /sms
JSON:   GET /api/cows  GET /api/cows/{id}/readings  GET /api/alerts  PATCH /api/alerts/{id}
        POST /api/demo/start  GET /api/demo/status
Handlers only parse input, call app/dashboard.py (or app/demo.py) and render the result.
"""

from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.schemas import AlertUpdate
from app import dashboard, demo
from app.config import CMT_RANGE
from app.db import SessionLocal

ROOT = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=ROOT / "templates")
templates.env.filters["t"] = dashboard.fmt_time
router = APIRouter()
NOT_FOUND = 404
DEFAULT_CHART_DAYS = 30
DEFAULT_DEMO_INTERVAL_S = 2.0


def _render(request: Request, template: str, context: dict) -> HTMLResponse:
    return templates.TemplateResponse(request, template, context)


# --------------------------------------------------------------------------- pages
@router.get("/", response_class=HTMLResponse)
def herd_overview(request: Request, farm: str | None = None) -> HTMLResponse:
    """Herd overview: summary cards + cows sorted by latest risk."""
    with SessionLocal() as session:
        return _render(request, "index.html", dashboard.herd_page(session, farm))


@router.get("/cows/{cow_id}", response_class=HTMLResponse)
def cow_detail(request: Request, cow_id: int) -> HTMLResponse:
    """One cow: reasons, alert episodes, charts and reading history."""
    with SessionLocal() as session:
        context = dashboard.cow_page(session, cow_id)
        if context is None:
            raise HTTPException(NOT_FOUND, "Cow not found")
        return _render(request, "cow.html", context)


@router.get("/alerts", response_class=HTMLResponse)
def alerts_page(request: Request) -> HTMLResponse:
    """Open alerts with Checked / Treated / False alarm buttons."""
    with SessionLocal() as session:
        return _render(request, "alerts.html", dashboard.alerts_page(session))


@router.get("/cmt", response_class=HTMLResponse)
def cmt_form(request: Request, message: str | None = None, ok: bool = True) -> HTMLResponse:
    """CMT entry form."""
    with SessionLocal() as session:
        return _render(request, "cmt.html", dashboard.cmt_page(session, message, ok))


@router.post("/cmt", response_class=HTMLResponse)
def cmt_submit(
    request: Request,
    cow_id: int = Form(...),
    day: date = Form(...),
    score: int = Form(..., ge=CMT_RANGE[0], le=CMT_RANGE[1]),
) -> HTMLResponse:
    """Save a CMT result as a training label, then show the form again with a message."""
    with SessionLocal() as session:
        message, ok = dashboard.record_cmt(session, cow_id, day, score)
        return _render(request, "cmt.html", dashboard.cmt_page(session, message, ok))


@router.get("/model", response_class=HTMLResponse)
def model_page(request: Request) -> HTMLResponse:
    """Model version, test-farm metrics, confusion matrix and SHAP images."""
    return _render(request, "model.html", dashboard.model_page())


@router.get("/demo", response_class=HTMLResponse)
def demo_page(request: Request) -> HTMLResponse:
    """The live demo page."""
    return _render(request, "demo.html", {"page": "demo", "farm": demo.DEMO_FARM})


@router.get("/sms", response_class=HTMLResponse)
def sms_page(request: Request) -> HTMLResponse:
    """The SMS log."""
    with SessionLocal() as session:
        return _render(request, "sms.html", dashboard.sms_page(session))


# --------------------------------------------------------------------------- JSON used by the pages
@router.get("/api/cows", response_model=None)
def api_cows(farm: str | None = None) -> list[dict]:
    """Cows with their latest risk (same rows as the herd table)."""
    with SessionLocal() as session:
        return dashboard.herd_rows(session, farm)


@router.get("/api/cows/{cow_id}/readings", response_model=None)
def api_cow_readings(cow_id: int, days: int = DEFAULT_CHART_DAYS) -> list[dict]:
    """Chart data for one cow."""
    with SessionLocal() as session:
        readings = dashboard.cow_readings(session, cow_id, days)
    if readings is None:
        raise HTTPException(NOT_FOUND, "Cow not found")
    return readings


@router.get("/api/alerts", response_model=None)
def api_alerts(status: Literal["active", "all"] = "active") -> list[dict]:
    """Alert episodes: active (open + checked) or all."""
    with SessionLocal() as session:
        return dashboard.list_alerts(session, status)


@router.patch("/api/alerts/{alert_id}", response_model=None)
def api_update_alert(alert_id: int, body: AlertUpdate) -> dict:
    """Mark an alert checked / treated / false_alarm (+ optional note)."""
    with SessionLocal() as session:
        updated = dashboard.update_alert(session, alert_id, body.status, body.note)
    if updated is None:
        raise HTTPException(NOT_FOUND, "Alert not found")
    return updated


@router.post("/api/demo/start", response_model=None)
def api_demo_start(interval: float = DEFAULT_DEMO_INTERVAL_S) -> dict:
    """Start the live demo replay in the background."""
    started = demo.start(interval)
    return {"started": started, **demo.status()}


@router.get("/api/demo/status", response_model=None)
def demo_status() -> dict:
    """Progress of the live demo."""
    return demo.status()
