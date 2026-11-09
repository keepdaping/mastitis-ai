"""
Error responses (api/ layer): the 422 format shared by the validation handler (api/main.py)
and /predict (api/routes/predict.py), as documented in docs/API_CONTRACT.md.
"""

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

UNPROCESSABLE = 422


def error_422(message: str, detail: list) -> JSONResponse:
    """{"error": "validation_error", "message": ..., "detail": [...]} with status 422."""
    return JSONResponse(
        status_code=UNPROCESSABLE, content={"error": "validation_error", "message": message, "detail": detail}
    )


async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with one readable message (e.g. "ph: Input should be less than or equal to 9")
    plus the standard per-field list."""
    errors = jsonable_encoder(exc.errors())
    parts = []
    for error in errors:
        location = ".".join(str(part) for part in error.get("loc", []) if part != "body") or "body"
        parts.append(f"{location}: {error.get('msg')}")
    return error_422("; ".join(parts), errors)
