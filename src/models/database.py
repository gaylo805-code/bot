"""DB engine/session helpers. SQLite default; Postgres via DATABASE_URL."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import get_settings
from src.models.job import Base


def _connect_args(database_url: str) -> dict:
    if database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def get_engine(database_url: str | None = None):
    """Create a SQLAlchemy engine for the given (or configured) URL."""
    settings = get_settings()
    url = database_url or settings.database_url
    return create_engine(url, connect_args=_connect_args(url), future=True)


def _ensure_columns(engine) -> None:
    """Lightweight migration: add columns missing from an older DB.

    create_all() never alters existing tables, so each new Job column
    needs an explicit ALTER here to keep old ./storage/dubbing.db working.
    """
    from sqlalchemy import inspect as sa_inspect
    from sqlalchemy import text as sa_text

    insp = sa_inspect(engine)
    if "jobs" not in insp.get_table_names():
        return
    existing = {c["name"] for c in insp.get_columns("jobs")}
    for name, ddl in (("burn_subs", "INTEGER DEFAULT 0"),
                      ("lipsync", "INTEGER DEFAULT 1")):
        if name not in existing:
            with engine.begin() as conn:
                conn.execute(sa_text(f"ALTER TABLE jobs ADD COLUMN {name} {ddl}"))


def init_db(database_url: str | None = None) -> None:
    """Create tables, migrate old ones, and ensure storage dirs exist."""
    settings = get_settings()
    settings.ensure_dirs()
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    _ensure_columns(engine)


def get_session_factory(database_url: str | None = None) -> sessionmaker:
    """Return a session factory bound to an engine."""
    engine = get_engine(database_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
