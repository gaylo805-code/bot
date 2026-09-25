"""API tests: upload/get/download/size-limit/health with TestClient."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from src.main import create_app


def _video_bytes(seconds: float = 1.0) -> bytes:
    out = Path("/tmp/opencode/autodub_tests/sample.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}",
         "-f", "lavfi", "-i", f"color=c=green:s=160x120:d={seconds}",
         "-shortest", str(out)], check=True)
    return out.read_bytes()


def _client(monkeypatch, max_mb: int = 500) -> TestClient:
    monkeypatch.setenv("MAX_UPLOAD_MB", str(max_mb))
    from src.config import get_settings
    get_settings.cache_clear()
    app = create_app()
    return TestClient(app)


def test_health(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_get_download_delete(monkeypatch):
    c = _client(monkeypatch)
    data = _video_bytes()
    # Avoid real Celery: patch delay to no-op
    import src.api.routes_upload as ru
    monkeypatch.setattr(ru, "process_dubbing_job", None, raising=False)
    import src.worker.tasks as _t  # noqa: F401
    # Patch at dispatch site via monkeypatching Celery task attribute
    from unittest.mock import MagicMock
    with monkeypatch.context() as m:
        # routes_upload imports inside function; patch celery task object
        try:
            from src.worker.tasks import process_dubbing_job as task
            m.setattr(task, "delay", MagicMock(return_value=None))
        except Exception:
            pass
        r = c.post("/api/upload", files={"file": ("t.mp4", data, "video/mp4")})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    r2 = c.get(f"/api/jobs/{job_id}")
    assert r2.status_code == 200
    assert r2.json()["job_id"] == job_id

    r3 = c.get("/api/jobs?page=1&page_size=5")
    assert r3.status_code == 200 and r3.json()["total"] >= 1

    r4 = c.get(f"/api/jobs/{job_id}/download")
    assert r4.status_code == 404  # no output yet

    r5 = c.delete(f"/api/jobs/{job_id}")
    assert r5.status_code == 200
    assert c.get(f"/api/jobs/{job_id}").status_code == 404


def test_upload_rejects_extension(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/upload",
               files={"file": ("evil.exe", b"xx", "application/octet-stream")})
    assert r.status_code == 400


def test_upload_size_limit(monkeypatch):
    c = _client(monkeypatch, max_mb=0)
    data = _video_bytes()
    r = c.post("/api/upload", files={"file": ("t.mp4", data, "video/mp4")})
    assert r.status_code in (413, 200)  # 0MB forces 413 for any payload
    if r.status_code == 200:
        raise AssertionError("expected 413 with MAX_UPLOAD_MB=0")
