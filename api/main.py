"""
The FastAPI app (api/ layer): assembles the prediction API, farmer dashboard and USSD into one service.

  api/routes/predict.py    POST /predict, GET /health, GET /model-info, GET /history/{cow_id}
  api/routes/dashboard.py  /, /cows/{id}, /alerts, /cmt, /model, /demo, /sms + /api/...
  api/routes/ussd.py       POST /ussd
Contract: docs/API_CONTRACT.md. Business logic lives in app/, the model pipeline in ml/.

Run:  uvicorn api.main:app --reload        Docs: http://127.0.0.1:8000/docs
Data: SQLite via SQLAlchemy (data/app.db, or MASTITIS_DB in .env).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.errors import validation_error
from api.routes.dashboard import router as dashboard_router
from api.routes.predict import router as predict_router
from api.routes.ussd import router as ussd_router
from app.db import init_db
from app.predictor import get_predictor

ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create tables if needed and load the model once at startup."""
    init_db()
    get_predictor()
    yield


app = FastAPI(
    title="Mastitis early-warning",
    version="1.2.0",
    lifespan=lifespan,
    description="Subclinical mastitis risk from in-line milk sensors "
    "(prototype, trained on SIMULATED data). Contract: docs/API_CONTRACT.md",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_exception_handler(RequestValidationError, validation_error)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
(ROOT / "reports").mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory=ROOT / "reports"), name="reports")  # plots for /model
app.include_router(dashboard_router)
app.include_router(ussd_router)
app.include_router(predict_router)
