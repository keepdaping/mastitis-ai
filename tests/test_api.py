"""API contract, validation, alert logic and model behaviour (need a trained model in models/).

Every test uses its own cow_id so tests don't share history.
"""
import json
from pathlib import Path

import pytest

from api.schemas import PredictRequest, PredictResponse

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def post(client, reading, **changes):
    r = client.post("/predict", json={**reading, **changes})
    assert r.status_code == 200, r.text
    return PredictResponse.model_validate(r.json())      # raises if the contract is broken


def check_contract(body: PredictResponse, raw: dict):
    assert set(raw) == {"cow_id", "risk_score", "risk_level", "top_reasons", "advice", "model_version", "alert"}
    assert 0.0 <= body.risk_score <= 1.0
    assert 1 <= len(body.top_reasons) <= 3
    for reason in body.top_reasons:
        assert set(reason.model_dump()) == {"feature", "effect", "text"}
        assert reason.effect[0] in "+-" and len(reason.effect.split(".")[1]) == 2
        assert reason.text
    assert body.advice
    assert body.model_version == json.loads((EXAMPLES.parent / "models" / "metadata.json").read_text())["version"]


# ---------------------------------------------------------------- basics
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["version"] and 0 < info["threshold"] < 1
    assert info["risk_bands"]["low"] == "< 0.25" and info["risk_bands"]["medium"].startswith("0.25 - ")
    assert "alert" in info["alert_rule"]


# ---------------------------------------------------------------- contract
def test_normal_cow_with_history_matches_contract(client, healthy_reading):
    for ts in ["2026-03-01T06:00:00", "2026-03-01T18:00:00", "2026-03-02T06:00:00"]:
        r = client.post("/predict", json={**healthy_reading, "cow_id": "NORMAL", "timestamp": ts})
        assert r.status_code == 200
    check_contract(PredictResponse.model_validate(r.json()), r.json())
    assert r.json()["risk_level"] == "low" and r.json()["alert"] is False
    assert len(client.get("/history/NORMAL").json()) == 3          # history is stored


def test_cow_with_no_history_matches_contract(client, healthy_reading):
    r = client.post("/predict", json={**healthy_reading, "cow_id": "BRAND-NEW"})
    assert r.status_code == 200
    check_contract(PredictResponse.model_validate(r.json()), r.json())


def test_null_sensor_values_match_contract(client, healthy_reading):
    reading = {**healthy_reading, "cow_id": "SENSOR-FAULT", "ec_ms_cm": None, "ph": None,
               "milk_temp_c": None, "milk_yield_l": None,
               "ec_quarters": {"LF": 5.2, "RF": None, "LR": None, "RR": 5.3}}
    r = client.post("/predict", json=reading)
    assert r.status_code == 200, r.text
    check_contract(PredictResponse.model_validate(r.json()), r.json())


def test_all_ec_missing_or_omitted_never_crashes(client, healthy_reading):
    no_quarters = {k: v for k, v in healthy_reading.items() if k != "ec_quarters"}
    post(client, no_quarters, cow_id="ONLY-WHOLE-MILK-EC")                      # ec_ms_cm fallback
    body = post(client, no_quarters, cow_id="NO-EC-AT-ALL", ec_ms_cm=None)      # herd-median fallback
    assert body.risk_level in {"low", "medium", "high"}


def test_contract_examples_are_valid():
    PredictRequest.model_validate(json.loads((EXAMPLES / "predict_request.json").read_text()))
    PredictResponse.model_validate(json.loads((EXAMPLES / "predict_response.json").read_text()))


def test_cmt_score_is_stored(client, healthy_reading):
    post(client, healthy_reading, cow_id="CMT-COW", cmt_score=3)
    stored = client.get("/history/CMT-COW").json()
    assert stored[0]["cmt_score"] == 3


# ---------------------------------------------------------------- validation (422)
@pytest.mark.parametrize("field, value, mentions", [
    ("breed", "unicorn", "breed"),
    ("grazing_system", "indoor", "grazing_system"),
    ("ph", 9.5, "ph"),
    ("ph", 4.0, "ph"),
    ("ec_ms_cm", 25.0, "ec_ms_cm"),
    ("milk_temp_c", 50.0, "milk_temp_c"),
    ("milk_temp_c", 20.0, "milk_temp_c"),
    ("cmt_score", 5, "cmt_score"),
])
def test_invalid_values_rejected_with_clear_message(client, healthy_reading, field, value, mentions):
    r = client.post("/predict", json={**healthy_reading, field: value})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_error"
    assert mentions in body["message"]


def test_invalid_quarter_ec_rejected(client, healthy_reading):
    r = client.post("/predict", json={**healthy_reading, "ec_quarters": {"LF": -1, "RF": 5, "LR": 5, "RR": 5}})
    assert r.status_code == 422
    assert "ec_quarters.LF" in r.json()["message"]


# ---------------------------------------------------------------- model behaviour
def test_high_ec_quarter_gives_higher_risk(client, healthy_reading, spike_reading):
    healthy = post(client, healthy_reading, cow_id="COW-A")
    sick = post(client, spike_reading, cow_id="COW-B")
    assert sick.risk_score > healthy.risk_score
    assert sick.risk_level == "high" and healthy.risk_level == "low"
    assert "Left rear" in sick.top_reasons[0].text        # quarter named in words
    assert "left rear" in sick.advice


def test_risk_bands():
    from api.main import risk_level
    assert [risk_level(s, 0.6) for s in (0.1, 0.24, 0.25, 0.59, 0.6, 0.9)] == \
        ["low", "low", "medium", "medium", "high", "high"]
    # threshold at or below 0.25 -> medium band starts at threshold / 2 so it never disappears
    assert [risk_level(s, 0.2) for s in (0.05, 0.1, 0.19, 0.2)] == ["low", "medium", "medium", "high"]


# ---------------------------------------------------------------- alert logic
def test_single_spike_does_not_alert(client, healthy_reading, spike_reading):
    cow = "SPIKE-ONCE"
    post(client, healthy_reading, cow_id=cow, timestamp="2026-03-01T06:00:00")
    spike = post(client, spike_reading, cow_id=cow, timestamp="2026-03-01T18:00:00")
    after = post(client, healthy_reading, cow_id=cow, timestamp="2026-03-02T06:00:00")
    assert spike.risk_level == "high" and spike.alert is False
    assert after.risk_level != "high" and after.alert is False


def test_two_consecutive_highs_alert(client, spike_reading):
    cow = "SPIKE-TWICE"
    first = post(client, spike_reading, cow_id=cow, timestamp="2026-03-01T06:00:00")
    second = post(client, spike_reading, cow_id=cow, timestamp="2026-03-01T18:00:00")
    assert first.risk_level == "high" and first.alert is False
    assert second.risk_level == "high" and second.alert is True


def test_highs_separated_by_normal_reading_do_not_alert(client, healthy_reading, spike_reading):
    cow = "SPIKE-GAP"
    post(client, spike_reading, cow_id=cow, timestamp="2026-03-01T06:00:00")
    post(client, healthy_reading, cow_id=cow, timestamp="2026-03-01T18:00:00")
    third = post(client, spike_reading, cow_id=cow, timestamp="2026-03-02T06:00:00")
    assert third.risk_level == "high" and third.alert is False


def test_highs_days_apart_do_not_alert(client, spike_reading):
    cow = "SPIKE-STALE"
    post(client, spike_reading, cow_id=cow, timestamp="2026-03-01T06:00:00")
    later = post(client, spike_reading, cow_id=cow, timestamp="2026-03-05T06:00:00")
    assert later.risk_level == "high" and later.alert is False


def test_reasons_never_empty():
    """A 'medium' score where every SHAP value is negative must still return one reason."""
    from api.reasons import top_reasons
    from ml.features import FEATURES
    values = {f: 0.0 for f in FEATURES} | {"breed_exotic": 1, "grazing_zero": 1, "parity": 2, "ph": 6.7}
    out = top_reasons([-0.01] * len(FEATURES), FEATURES, values, "lf", "medium")
    assert len(out) == 1 and out[0]["effect"] == "-0.01" and out[0]["text"]
