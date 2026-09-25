"""Tests: bot helpers (pure) + DubbingAPI client (mocked httpx)."""

from __future__ import annotations

import httpx
import pytest

from src.bot import api_client as api_mod
from src.bot.api_client import DubbingAPI, DubbingAPIError
from src.bot.helpers import (
    human_size,
    is_terminal,
    needs_link,
    progress_bar,
    public_download_url,
    result_filename,
    stage_label,
    status_color,
    truncate,
    validate_attachment,
)


# --- helpers ------------------------------------------------------------
def test_progress_bar_edges():
    assert progress_bar(0) == "░░░░░░░░░░░░ 0.0%"
    assert progress_bar(100) == "████████████ 100.0%"
    assert progress_bar(50, width=4) == "██░░ 50.0%"
    assert progress_bar(-5).endswith("0.0%")
    assert progress_bar(999).endswith("100.0%")


def test_human_size():
    assert human_size(0) == "0B"
    assert human_size(512) == "512B"
    assert human_size(2048) == "2.0KB"
    assert human_size(5 * 1024 * 1024) == "5.0MB"
    assert human_size(2 * 1024**3) == "2.0GB"


def test_validate_attachment():
    cap = 10 * 1024 * 1024
    assert validate_attachment("a.mp4", 1000, cap) is None
    assert validate_attachment("A.MKV", 1000, cap) is None
    assert "không hỗ trợ" in (validate_attachment("a.exe", 100, cap) or "")
    assert "không hỗ trợ" in (validate_attachment("noext", 100, cap) or "")
    assert "rỗng" in (validate_attachment("a.mp4", 0, cap) or "")
    assert "quá lớn" in (validate_attachment("a.mp4", cap + 1, cap) or "")


def test_stage_label_and_color():
    assert "ASR" in stage_label("asr")
    assert "Hoàn thành" in stage_label("done")
    assert stage_label("weird-stage") == "weird-stage"
    assert status_color("done") == 0x2ECC71
    assert status_color("failed") == 0xE74C3C
    assert status_color("nope") == 0x5865F2  # blurple fallback


def test_is_terminal_and_needs_link():
    assert is_terminal("done") and is_terminal("FAILED")
    assert not is_terminal("tts") and not is_terminal("")
    assert needs_link(0, 100) is True
    assert needs_link(101, 100) is True
    assert needs_link(50, 100) is False


def test_result_filename():
    assert result_filename("My Video!.mp4", "abc") == "My Video_dubbed_vi.mp4"
    assert result_filename("", "abc") == "video_dubbed_vi.mp4"
    assert result_filename("a.mkv", "abc").endswith(".mp4")


def test_public_download_url():
    u = public_download_url("http://api:8000", "dub.example.com", "j1")
    assert u == "https://dub.example.com/api/jobs/j1/download"
    u2 = public_download_url("http://localhost:8000/", "", "j1")
    assert u2 == "http://localhost:8000/api/jobs/j1/download"


def test_truncate():
    assert truncate("abc", 10) == "abc"
    assert truncate("x" * 50, 10).endswith("…") and len(truncate("x" * 50, 10)) == 10
    assert truncate("", 5) == ""


# --- DubbingAPI with fake httpx -----------------------------------------
class _FakeResp:
    def __init__(self, payload=None, content=b"", status=200, headers=None):
        self._payload = payload
        self.content = content
        self.status_code = status
        self.headers = headers or {}
        self.text = str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)


class _FakeClient:
    """Records calls; canned responses per method."""

    last_kwargs: dict = {}

    def __init__(self, *args, **kwargs):
        _FakeClient.last_kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, files=None, data=None, headers=None):
        assert url.endswith("/api/upload")
        assert files["file"][0] == "a.mp4"
        assert data["keep_background"] == "true"
        assert headers == {"X-API-Key": "secret"}
        return _FakeResp({"job_id": "j1", "status": "queued"})

    def get(self, url, params=None, headers=None):
        if url.endswith("/download"):
            return _FakeResp(content=b"VID",
                             headers={"content-disposition": 'attachment; filename="r.mp4"'})
        if "/api/jobs/" in url:
            return _FakeResp({"job_id": "j1", "status": "done", "progress": 100.0})
        return _FakeResp({"total": 1, "jobs": []})

    def delete(self, url, headers=None):
        return _FakeResp({"job_id": "j1", "deleted": True})


@pytest.fixture()
def _fake_http(monkeypatch):
    monkeypatch.setattr(api_mod.httpx, "Client", _FakeClient)
    yield


def test_api_upload_get_list_delete_download(_fake_http):
    api = DubbingAPI("http://api:8000/", api_key="secret")
    assert api.base_url == "http://api:8000"  # trailing slash stripped
    created = api.upload_video(b"xx", "a.mp4", keep_background=True)
    assert created["job_id"] == "j1"
    assert api.get_job("j1")["status"] == "done"
    assert api.list_jobs()["total"] == 1
    assert api.delete_job("j1")["deleted"] is True
    data, fname = api.download("j1")
    assert data == b"VID" and fname == "r.mp4"


def test_api_error_raises(_fake_http, monkeypatch):
    class _Bad(_FakeClient):
        def get(self, url, params=None, headers=None):
            return _FakeResp({"detail": "Job not found"}, status=404)

    monkeypatch.setattr(api_mod.httpx, "Client", _Bad)
    api = DubbingAPI("http://api:8000")
    with pytest.raises(DubbingAPIError, match="404"):
        api.get_job("missing")


def test_api_no_key_sends_no_header(monkeypatch):
    seen: dict = {}

    class _Spy(_FakeClient):
        def get(self, url, params=None, headers=None):
            seen.update(headers or {})
            return _FakeResp({"job_id": "j", "status": "queued"})

    monkeypatch.setattr(api_mod.httpx, "Client", _Spy)
    DubbingAPI("http://x").get_job("j")
    assert seen == {}
