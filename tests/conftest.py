"""Shared pytest fixtures: isolated temp storage + test DB."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Isolate storage BEFORE importing app modules.
TMP = Path("/tmp/opencode/autodub_tests")
TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("STORAGE_PATH", str(TMP / "storage"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{TMP}/test.db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
os.environ.setdefault("API_KEY", "")
os.environ.setdefault("PUBLIC_HOSTNAME", "")

from src.config import get_settings  # noqa: E402

get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clean_env(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    store = tmp_path / "storage"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("STORAGE_PATH", str(store))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
