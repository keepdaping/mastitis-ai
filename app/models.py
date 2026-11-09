"""
Database tables (SQLAlchemy ORM).

farms     - one row per farm; `id` is the farm code sent by the sensor (e.g. "F02", "FARM-01")
cows      - one row per cow; `tag` is the cow_id sent to /predict (e.g. "F02-C005")
readings  - one row per milking (+ the model's result for it)
alerts    - one row per alert EPISODE (a run of consecutive high readings)
sms_log   - every SMS attempt, whatever the outcome
"""
from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

# Alert statuses. "open" and "checked" (farmer has looked, not finished) are ACTIVE:
# while a cow has an active alert, new high readings join that episode instead of
# creating a new alert and a new SMS.
ALERT_STATUSES = ["open", "checked", "treated", "false_alarm", "resolved_auto"]
ACTIVE_ALERT_STATUSES = ["open", "checked"]


class Farm(Base):
    __tablename__ = "farms"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)          # farm code
    name: Mapped[str] = mapped_column(String(120))
    district: Mapped[str | None] = mapped_column(String(80))
    owner_name: Mapped[str | None] = mapped_column(String(120))
    phone: Mapped[str | None] = mapped_column(String(32))                  # SMS goes here, e.g. +2567...

    cows: Mapped[list["Cow"]] = relationship(back_populates="farm")


class Cow(Base):
    __tablename__ = "cows"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    farm_id: Mapped[str | None] = mapped_column(ForeignKey("farms.id"))
    tag: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # = cow_id in the API
    breed: Mapped[str] = mapped_column(String(16))                          # exotic | local
    parity: Mapped[int] = mapped_column(Integer)
    calving_date: Mapped[date | None] = mapped_column(Date)                 # -> days in milk
    grazing_system: Mapped[str] = mapped_column(String(16))                 # zero | tethered | open

    farm: Mapped[Farm | None] = relationship(back_populates="cows")
    readings: Mapped[list["Reading"]] = relationship(back_populates="cow", order_by="Reading.timestamp")

    def days_in_milk(self, when: datetime) -> int | None:
        return (when.date() - self.calving_date).days if self.calving_date else None


class Reading(Base):
    __tablename__ = "readings"
    __table_args__ = (UniqueConstraint("cow_id", "timestamp"),)   # re-sent milking replaces the old one
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cow_id: Mapped[int] = mapped_column(ForeignKey("cows.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    # Sensor values (any may be missing)
    ec_ms_cm: Mapped[float | None] = mapped_column(Float)
    ec_LF: Mapped[float | None] = mapped_column(Float)
    ec_RF: Mapped[float | None] = mapped_column(Float)
    ec_LR: Mapped[float | None] = mapped_column(Float)
    ec_RR: Mapped[float | None] = mapped_column(Float)
    milk_temp_c: Mapped[float | None] = mapped_column(Float)
    ph: Mapped[float | None] = mapped_column(Float)
    milk_yield_l: Mapped[float | None] = mapped_column(Float)
    days_in_milk: Mapped[int | None] = mapped_column(Integer)      # at the time of this milking
    # Label (from the sensor request or the /cmt form): 0-4 = field CMT 1-5 minus 1
    cmt_score: Mapped[int | None] = mapped_column(Integer)
    # Model output
    risk_score: Mapped[float | None] = mapped_column(Float)
    risk_level: Mapped[str | None] = mapped_column(String(8))
    alert: Mapped[bool] = mapped_column(Boolean, default=False)
    # True = not part of the cow's "normal" (high risk, or during an active alert episode).
    # Used by ml/features.py for the clean baseline, exactly as in training.
    baseline_exclude: Mapped[bool] = mapped_column(Boolean, default=False)
    top_reasons: Mapped[list | None] = mapped_column(JSON)
    model_version: Mapped[str | None] = mapped_column(String(40))

    cow: Mapped[Cow] = relationship(back_populates="readings")


class Alert(Base):
    """One alert EPISODE: starts at the first alert=true reading, grows while the cow stays high."""
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cow_id: Mapped[int] = mapped_column(ForeignKey("cows.id"), index=True)
    reading_id: Mapped[int] = mapped_column(ForeignKey("readings.id"))        # first reading of the episode
    last_reading_id: Mapped[int] = mapped_column(ForeignKey("readings.id"))   # latest alert reading
    reading_count: Mapped[int] = mapped_column(Integer, default=1)            # alert readings in the episode
    created_at: Mapped[datetime] = mapped_column(DateTime)                    # = time of the first reading
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    note: Mapped[str | None] = mapped_column(Text)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime)
    # The cow's latest reading when a farmer closed the alert (treated / false alarm). A new
    # episode can only start after a NON-high reading that comes after this one ("re-arm"),
    # so a cow that is still high while antibiotics work doesn't trigger a new alert + SMS.
    closed_after_reading_id: Mapped[int | None] = mapped_column(ForeignKey("readings.id"))

    cow: Mapped[Cow] = relationship()
    first_reading: Mapped[Reading] = relationship(foreign_keys=[reading_id])
    last_reading: Mapped[Reading] = relationship(foreign_keys=[last_reading_id])

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_ALERT_STATUSES

    @property
    def episode_hours(self) -> float:
        """Episode length from first to latest alert reading, in hours."""
        return (self.last_reading.timestamp - self.first_reading.timestamp).total_seconds() / 3600


class SmsLog(Base):
    __tablename__ = "sms_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cow_id: Mapped[int | None] = mapped_column(ForeignKey("cows.id"), index=True)   # for the 12 h limit
    alert_id: Mapped[int | None] = mapped_column(ForeignKey("alerts.id"))
    to_phone: Mapped[str | None] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24))      # sent | failed | rate_limited | no_phone
    provider: Mapped[str] = mapped_column(String(24))
    provider_response: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime)

    cow: Mapped[Cow | None] = relationship()
