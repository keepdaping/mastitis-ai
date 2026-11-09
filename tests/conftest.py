"""Shared test setup: make the project importable and give the API a throw-away database."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Must be set BEFORE the app is imported: tests use a throw-away database and never send real SMS
# (load_dotenv() does not override variables that are already set).
os.environ["MASTITIS_DB"] = os.path.join(tempfile.mkdtemp(prefix="mastitis_test_"), "app.db")
os.environ["SMS_PROVIDER"] = "console"
os.environ["SMS_LANGUAGE"] = "en"

MODEL = ROOT / "models" / "mastitis_model.joblib"


@pytest.fixture(scope="session")
def client():
    if not MODEL.exists():
        pytest.skip("No trained model - run `python -m ml.train` first")
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def healthy_reading():
    """A normal reading in the docs/API_CONTRACT.md format (each test sets its own cow_id)."""
    return {
        "cow_id": "TEST-HEALTHY", "farm_id": "TEST", "timestamp": "2026-03-01T06:00:00",
        "ec_ms_cm": 5.2, "ec_quarters": {"LF": 5.2, "RF": 5.1, "LR": 5.3, "RR": 5.2},
        "milk_temp_c": 37.5, "ph": 6.70, "milk_yield_l": 5.5, "cmt_score": None,
        "breed": "exotic", "parity": 3, "days_in_milk": 90, "grazing_system": "tethered",
    }


@pytest.fixture
def spike_reading(healthy_reading):
    """Same cow profile, but the left rear quarter's EC is far above the others (-> high risk)."""
    return {**healthy_reading, "ec_ms_cm": 5.9, "ph": 6.95, "milk_temp_c": 38.1,
            "ec_quarters": {"LF": 5.2, "RF": 5.1, "LR": 8.2, "RR": 5.2}}
