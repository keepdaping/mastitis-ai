"""
USSD route (api/ layer): Africa's Talking USSD callback POST /ussd. The menu is in app/ussd.py.
"""

from fastapi import APIRouter, Form
from fastapi.responses import PlainTextResponse

from app.db import SessionLocal
from app.ussd import ussd_reply

router = APIRouter()


# The form field names (sessionId, serviceCode, phoneNumber) are fixed by Africa's Talking
# (camelCase is allowed for this file in pyproject.toml).
@router.post("/ussd", response_class=PlainTextResponse)
def ussd(
    sessionId: str = Form(""),
    serviceCode: str = Form(""),
    phoneNumber: str = Form(""),
    text: str = Form(""),
) -> PlainTextResponse:
    """Answer one USSD step with "CON ..." (menu) or "END ..." (final screen)."""
    with SessionLocal() as session:
        return PlainTextResponse(ussd_reply(session, phoneNumber, text))
