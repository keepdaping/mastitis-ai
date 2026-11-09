"""
Database setup: SQLite via SQLAlchemy.

The file is data/app.db by default; override with MASTITIS_DB (a file path) in .env.
Tables are defined in app/models.py and created on startup by init_db().
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")          # does NOT override variables that are already set

DB_PATH = Path(os.getenv("MASTITIS_DB", ROOT / "data" / "app.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# check_same_thread=False: FastAPI may use the connection from a worker thread
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")     # enforce cow -> farm etc.
    cur.execute("PRAGMA journal_mode=WAL")    # dashboard can read while /predict writes
    cur.execute("PRAGMA synchronous=NORMAL")  # safe with WAL; avoids a disk sync on every commit
    cur.close()


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# Columns added after the first release: (table, column, SQL type). init_db() adds them to
# an existing database so it keeps working without a reset.
ADDED_COLUMNS = [
    ("readings", "baseline_exclude", "BOOLEAN DEFAULT 0"),
    ("alerts", "closed_after_reading_id", "INTEGER REFERENCES readings(id)"),
]


def init_db():
    """Create all tables that don't exist yet, and add columns introduced later."""
    from app import models  # noqa: F401  (registers the tables on Base)
    Base.metadata.create_all(engine)
    with engine.begin() as con:
        for table, column, sql_type in ADDED_COLUMNS:
            existing = {row[1] for row in con.exec_driver_sql(f"PRAGMA table_info({table})")}
            if column not in existing:
                con.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")


def reset_db():
    """Drop and recreate every table (used by scripts/seed_db.py --reset)."""
    from app import models  # noqa: F401
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
