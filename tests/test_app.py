"""Database, alert episodes, SMS, dashboard pages and CMT form (need a trained model in models/).

Each test uses its own farm / cow ids, so tests don't share history.
Readings are 12 hours apart, like twice-daily milking.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Alert, Cow, Farm, Reading, SmsLog

T0 = datetime(2026, 3, 1, 6, 0)


# ---------------------------------------------------------------- helpers
def make_farm(farm_id: str, phone: str | None = "+256700000123", name: str = "Test Farm"):
    with SessionLocal() as s:
        s.merge(Farm(id=farm_id, name=name, district="Test", owner_name="Tester", phone=phone))
        s.commit()


def send(client, reading, cow, farm, hours, **changes):
    """POST one reading `hours` after T0; returns the response JSON."""
    body = {**reading, "cow_id": cow, "farm_id": farm, "timestamp": (T0 + timedelta(hours=hours)).isoformat(),
            **changes}
    r = client.post("/predict", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def alerts_for(cow_tag):
    with SessionLocal() as s:
        cow = s.execute(select(Cow).where(Cow.tag == cow_tag)).scalar_one()
        return s.execute(select(Alert).where(Alert.cow_id == cow.id).order_by(Alert.id)).scalars().all()


def sms_for(cow_tag):
    with SessionLocal() as s:
        cow = s.execute(select(Cow).where(Cow.tag == cow_tag)).scalar_one()
        return s.execute(select(SmsLog).where(SmsLog.cow_id == cow.id).order_by(SmsLog.id)).scalars().all()


def readings_for(cow_tag):
    with SessionLocal() as s:
        cow = s.execute(select(Cow).where(Cow.tag == cow_tag)).scalar_one()
        return s.execute(select(Reading).where(Reading.cow_id == cow.id)
                         .order_by(Reading.timestamp)).scalars().all()


# ---------------------------------------------------------------- /predict saves + alerts
def test_predict_saves_reading_and_alerts_only_on_two_highs(client, spike_reading):
    make_farm("TF-A")
    first = send(client, spike_reading, "EP-A", "TF-A", 0)
    assert first["risk_level"] == "high" and first["alert"] is False
    assert len(readings_for("EP-A")) == 1 and alerts_for("EP-A") == []      # saved, no alert yet
    second = send(client, spike_reading, "EP-A", "TF-A", 12)
    assert second["alert"] is True
    saved = readings_for("EP-A")
    assert len(saved) == 2 and saved[-1].alert and saved[-1].top_reasons and saved[-1].model_version
    assert len(alerts_for("EP-A")) == 1


def test_five_consecutive_highs_make_one_alert_and_one_sms(client, spike_reading):
    make_farm("TF-B")
    for i in range(5):
        send(client, spike_reading, "EP-B", "TF-B", 12 * i)
    alerts = alerts_for("EP-B")
    assert len(alerts) == 1
    assert alerts[0].status == "open"
    assert alerts[0].reading_count == 4                     # readings 2-5 carry alert=true
    assert [m.status for m in sms_for("EP-B")] == ["sent"]


def test_episode_auto_resolves_after_four_normal_readings(client, spike_reading, healthy_reading):
    make_farm("TF-C")
    send(client, spike_reading, "EP-C", "TF-C", 0)
    send(client, spike_reading, "EP-C", "TF-C", 12)
    for i in range(3):
        send(client, healthy_reading, "EP-C", "TF-C", 24 + 12 * i)
    assert alerts_for("EP-C")[0].status == "open"           # 3 normal readings: still open
    send(client, healthy_reading, "EP-C", "TF-C", 60)
    alert = alerts_for("EP-C")[0]
    assert alert.status == "resolved_auto" and alert.closed_at is not None


def test_resolved_then_high_again_starts_new_episode_with_new_sms(client, spike_reading, healthy_reading):
    make_farm("TF-D")
    send(client, spike_reading, "EP-D", "TF-D", 0)
    send(client, spike_reading, "EP-D", "TF-D", 12)
    first = alerts_for("EP-D")[0]
    assert client.patch(f"/api/alerts/{first.id}", json={"status": "treated", "note": "antibiotics"}).status_code == 200
    send(client, healthy_reading, "EP-D", "TF-D", 24)
    send(client, healthy_reading, "EP-D", "TF-D", 36)
    send(client, spike_reading, "EP-D", "TF-D", 48)
    send(client, spike_reading, "EP-D", "TF-D", 60)
    alerts = alerts_for("EP-D")
    assert [a.status for a in alerts] == ["treated", "open"]
    assert [m.status for m in sms_for("EP-D")] == ["sent", "sent"]


# ---------------------------------------------------------------- SMS
def test_sms_rate_limited_to_one_per_cow_per_12_hours(client, spike_reading, healthy_reading):
    make_farm("TF-E")
    send(client, spike_reading, "EP-E", "TF-E", 0)
    send(client, spike_reading, "EP-E", "TF-E", 2)          # SMS sent at T0+2h
    first = alerts_for("EP-E")[0]
    client.patch(f"/api/alerts/{first.id}", json={"status": "false_alarm"})
    send(client, healthy_reading, "EP-E", "TF-E", 4)
    send(client, spike_reading, "EP-E", "TF-E", 6)
    send(client, spike_reading, "EP-E", "TF-E", 8)          # new episode, but only 6 h after the last SMS
    assert len(alerts_for("EP-E")) == 2
    assert [m.status for m in sms_for("EP-E")] == ["sent", "rate_limited"]


def test_sms_text_is_short_and_names_the_quarter(client, spike_reading):
    make_farm("TF-F", name="A Very Long Farm Name In Kayunga District Near The Lake Shore Road")
    send(client, spike_reading, "COW-012", "TF-F", 0)
    send(client, spike_reading, "COW-012", "TF-F", 12)
    msg = sms_for("COW-012")[0]
    assert msg.status == "sent" and msg.to_phone == "+256700000123"
    assert len(msg.message) <= 160
    assert msg.message.startswith("MastitisAI: Cow COW-012") and "Left rear quarter" in msg.message


def test_no_phone_is_logged_not_sent(client, spike_reading):
    make_farm("TF-G", phone=None)
    send(client, spike_reading, "EP-G", "TF-G", 0)
    send(client, spike_reading, "EP-G", "TF-G", 12)
    assert [m.status for m in sms_for("EP-G")] == ["no_phone"]


def test_luganda_falls_back_to_english():
    from app.sms_templates import render
    assert render("high_risk", "lg", cow="1", farm="X", quarter="Left rear") == \
        render("high_risk", "en", cow="1", farm="X", quarter="Left rear")


# ---------------------------------------------------------------- alerts API + CMT form
def test_patch_alert_status(client, spike_reading):
    make_farm("TF-H")
    send(client, spike_reading, "EP-H", "TF-H", 0)
    send(client, spike_reading, "EP-H", "TF-H", 12)
    alert_id = alerts_for("EP-H")[0].id
    r = client.patch(f"/api/alerts/{alert_id}", json={"status": "checked", "note": "CMT done: ++"})
    assert r.status_code == 200 and r.json()["status"] == "checked" and r.json()["note"] == "CMT done: ++"
    assert any(a["id"] == alert_id for a in client.get("/api/alerts").json())     # still active
    assert client.patch(f"/api/alerts/{alert_id}", json={"status": "resolved_auto"}).status_code == 422
    assert client.patch("/api/alerts/999999", json={"status": "treated"}).status_code == 404
    assert client.patch(f"/api/alerts/{alert_id}", json={"status": "false_alarm"}).json()["status"] == "false_alarm"


def test_cmt_form_saves_training_label(client, healthy_reading):
    send(client, healthy_reading, "CMT-FORM", "TF-I", 0)
    send(client, healthy_reading, "CMT-FORM", "TF-I", 12)           # same day, 18:00
    cow_id = readings_for("CMT-FORM")[0].cow_id
    r = client.post("/cmt", data={"cow_id": cow_id, "day": T0.date().isoformat(), "score": 3})
    assert r.status_code == 200 and "Saved" in r.text
    saved = readings_for("CMT-FORM")
    assert saved[1].cmt_score == 3 and saved[0].cmt_score is None   # the day's LAST milking
    r = client.post("/cmt", data={"cow_id": cow_id, "day": "2020-01-01", "score": 2})
    assert "No milking recorded" in r.text


def test_days_in_milk_from_calving_date(client, healthy_reading):
    send(client, healthy_reading, "DIM-COW", "TF-J", 0, days_in_milk=100)
    no_dim = {k: v for k, v in healthy_reading.items() if k != "days_in_milk"}
    send(client, no_dim, "DIM-COW", "TF-J", 48)
    assert readings_for("DIM-COW")[-1].days_in_milk == 102
    r = client.post("/predict", json={**no_dim, "cow_id": "NEVER-SEEN"})
    assert r.status_code == 422 and "days_in_milk" in r.json()["message"]


# ---------------------------------------------------------------- pages after seeding
def test_dashboard_pages_after_seeding(client):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from seed_db import seed
    counts = seed(cows_per_farm=1, days=3, quiet=True)
    assert counts["readings"] > 0
    cows = client.get("/api/cows").json()
    seeded = next(c for c in cows if c["farm_id"] == "F02")
    pages = ["/", "/?farm=F02", f"/cows/{seeded['id']}", "/alerts", "/cmt", "/model", "/demo", "/sms",
             "/api/cows?farm=F02", f"/api/cows/{seeded['id']}/readings", "/api/alerts?status=all",
             "/api/demo/status", "/static/app.css", "/reports/shap_summary.png"]
    for page in pages:
        r = client.get(page)
        assert r.status_code == 200, f"{page}: {r.status_code}"
    assert "SIMULATED DATA" in client.get("/").text
    assert client.get("/cows/999999").status_code == 404


# ---------------------------------------------------------------- USSD (optional feature)
def ussd(client, text, phone="+256700000777"):
    r = client.post("/ussd", data={"sessionId": "s1", "serviceCode": "*384#", "phoneNumber": phone, "text": text})
    assert r.status_code == 200
    return r.text


def test_ussd_menu_cmt_and_treat(client, spike_reading):
    make_farm("TF-U", phone="+256700000777", name="Ussd Farm")
    today = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
    for i, ts in enumerate([today - timedelta(hours=12), today]):
        r = client.post("/predict", json={**spike_reading, "cow_id": "TF-U-C042", "farm_id": "TF-U",
                                          "timestamp": ts.isoformat()})
        assert r.status_code == 200
    assert ussd(client, "").startswith("CON MastitisAI - Ussd Farm")
    assert "TF-U-C042" in ussd(client, "1")                                    # high-risk list
    assert ussd(client, "2*C042").startswith("CON CMT score for TF-U-C042")   # short tag accepted
    assert ussd(client, "2*C042*4").startswith("END Saved")
    assert readings_for("TF-U-C042")[-1].cmt_score == 3                        # field 4 -> stored 3
    assert "1. TF-U-C042" in ussd(client, "3")
    assert ussd(client, "3*1") == "END Alert for TF-U-C042 marked as treated."
    assert alerts_for("TF-U-C042")[0].status == "treated"
    assert ussd(client, "", phone="+256711111111").startswith("END This phone number is not registered")
