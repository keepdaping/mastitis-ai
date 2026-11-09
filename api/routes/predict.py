"""
Prediction API routes (api/ layer): POST /predict, GET /health, GET /model-info, GET /history/{cow_id}.
Contract: docs/API_CONTRACT.md. The work is done by app/service.py.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.errors import error_422
from api.schemas import ErrorResponse, HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse
from app import service
from app.db import SessionLocal

router = APIRouter()
SERVICE_UNAVAILABLE = 503
DEFAULT_HISTORY_LIMIT = 50


@router.post("/predict", response_model=PredictResponse, responses={422: {"model": ErrorResponse}})
def predict(request: PredictRequest) -> PredictResponse | JSONResponse:
    """Risk for one milking; saves the reading and manages alerts + SMS."""
    with SessionLocal() as session:
        try:
            response, _ = service.handle_reading(session, request)
        except service.ModelNotLoadedError as error:
            raise HTTPException(SERVICE_UNAVAILABLE, str(error)) from error
        except service.InvalidReadingError as error:
            session.rollback()
            detail = [{"loc": ["body", error.field], "msg": error.message, "type": "value_error"}]
            return error_422(str(error), detail)
    return response


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + whether a model is loaded."""
    return HealthResponse(**service.health())


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Model name, version, threshold, risk bands, alert rule and metrics."""
    try:
        return ModelInfoResponse(**service.model_info())
    except service.ModelNotLoadedError as error:
        raise HTTPException(SERVICE_UNAVAILABLE, "Model not loaded") from error


@router.get("/history/{cow_id}", response_model=None)
def history(cow_id: str, limit: int = DEFAULT_HISTORY_LIMIT) -> list[dict]:
    """Stored readings for a cow tag, newest first."""
    with SessionLocal() as session:
        return service.cow_history(session, cow_id, limit)
