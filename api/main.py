"""
The FastAPI app: prediction API + farmer dashboard + SMS alerts + USSD, in one service.

API (contract: docs/API_CONTRACT.md):
  POST /predict          - risk for one milking; saves the reading, manages alerts + SMS
  GET  /health           - liveness + model loaded?
  GET  /model-info       - model name, version, threshold, risk bands, metrics
  GET  /history/{cow_id} - stored readings for a cow tag
Dashboard (app/routes/dashboard.py): /, /cows/{id}, /alerts, /cmt, /model, /demo, /sms + /api/*
USSD (app/routes/ussd.py): POST /ussd

Run:  uvicorn api.main:app --reload        Docs: http://127.0.0.1:8000/docs
Data: SQLite via SQLAlchemy (data/app.db, or MASTITIS_DB in .env).
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from api.schemas import ErrorResponse, HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse
from app.db import SessionLocal, init_db
from app.models import Cow, Reading
from app.predictor import MEDIUM_FROM, get_predictor, medium_floor, risk_level  # noqa: F401 (re-exported)
from app.routes.dashboard import router as dashboard_router
from app.routes.ussd import router as ussd_router
from app.service import ALERT_MAX_GAP, InputProblem, ModelNotLoaded, handle_reading

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "reports" / "metrics.json"
METADATA_PATH = ROOT / "models" / "metadata.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()            # create tables if needed
    get_predictor()      # load the model once at startup
    yield


app = FastAPI(title="Mastitis early-warning", version="1.2.0", lifespan=lifespan,
              description="Subclinical mastitis risk from in-line milk sensors "
                          "(prototype, trained on SIMULATED data). Contract: docs/API_CONTRACT.md")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
(ROOT / "reports").mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory=ROOT / "reports"), name="reports")   # plots for /model
app.include_router(dashboard_router)
app.include_router(ussd_router)


def error_422(message: str, detail: list) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "validation_error", "message": message,
                                                  "detail": detail})


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """422 with one readable message (e.g. "ph: Input should be less than or equal to 9")
    plus the standard per-field list."""
    errors = jsonable_encoder(exc.errors())
    parts = []
    for e in errors:
        loc = ".".join(str(x) for x in e.get("loc", []) if x != "body") or "body"
        parts.append(f"{loc}: {e.get('msg')}")
    return error_422("; ".join(parts), errors)


@app.post("/predict", response_model=PredictResponse, responses={422: {"model": ErrorResponse}})
def predict(req: PredictRequest):
    with SessionLocal() as session:
        try:
            response, _ = handle_reading(session, req)
        except ModelNotLoaded as e:
            raise HTTPException(503, str(e))
        except InputProblem as e:
            session.rollback()
            return error_422(str(e), [{"loc": ["body", e.field], "msg": e.message, "type": "value_error"}])
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    p = get_predictor()
    return HealthResponse(status="ok", model_loaded=p is not None, model_version=p.version if p else None)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    p = get_predictor()
    if p is None:
        raise HTTPException(503, "Model not loaded")
    b = p.bundle
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    meta = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    floor = medium_floor(b["threshold"])
    return ModelInfoResponse(
        model_name=b["model_name"], version=b["version"], trained_at=b["trained_at"],
        threshold=b["threshold"],
        risk_bands={"low": f"< {floor}", "medium": f"{floor} - {b['threshold']:.3f}",
                    "high": f">= {b['threshold']:.3f}"},
        alert_rule=f"alert=true when this and the previous reading of the cow are both 'high' "
                   f"(at most {int(ALERT_MAX_GAP.total_seconds() // 3600)} h apart)",
        features=b["features"], positive_definition=b["positive_definition"], data=b["data"],
        test_metrics=metrics.get("test_metrics"), cv_results=meta.get("cv_results"))


@app.get("/history/{cow_id}")
def history(cow_id: str, limit: int = 50):
    """Stored readings for a cow tag, newest first."""
    with SessionLocal() as session:
        cow = session.execute(select(Cow).where(Cow.tag == cow_id)).scalar_one_or_none()
        if cow is None:
            return []
        rows = session.execute(select(Reading).where(Reading.cow_id == cow.id)
                               .order_by(Reading.timestamp.desc()).limit(limit)).scalars().all()
        return [reading_json(r, cow) for r in rows]


def reading_json(r: Reading, cow: Cow) -> dict:
    return {"cow_id": cow.tag, "timestamp": r.timestamp.isoformat(), "ec_ms_cm": r.ec_ms_cm,
            "ec_quarters": {q: getattr(r, f"ec_{q}") for q in ["LF", "RF", "LR", "RR"]},
            "milk_temp_c": r.milk_temp_c, "ph": r.ph, "milk_yield_l": r.milk_yield_l,
            "days_in_milk": r.days_in_milk, "cmt_score": r.cmt_score, "risk_score": r.risk_score,
            "risk_level": r.risk_level, "alert": r.alert, "top_reasons": r.top_reasons,
            "model_version": r.model_version}
