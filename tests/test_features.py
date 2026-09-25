"""Tests: migration, srt/thumbnail/stats/retry endpoints, burn-in, new helpers."""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.main import create_app


def _client(monkeypatch) -> TestClient:
    from src.config import get_settings
    get_settings.cache_clear()
    return TestClient(create_app())


def _make_video(path: Path, seconds: float = 2.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}",
         "-f", "lavfi", "-i", f"color=c=blue:s=160x120:d={seconds}",
         "-shortest", str(path)], check=True)
    return path


def _seed_job(monkeypatch, job_id="jx1", with_segments=True,
              status="done", source: Path | None = None,
              output: Path | None = None) -> str:
    from src.config import get_settings
    from src.models.database import get_session_factory, init_db
    from src.models.job import Job, Segment

    get_settings.cache_clear()
    init_db()
    factory = get_session_factory()
    with factory() as db:
        db.add(Job(id=job_id, filename="v.mp4", status=status, progress=100.0,
                   source_lang="en", target_lang="vi", voice_id="",
                   keep_background=0, burn_subs=0, current_stage=status,
                   source_path=str(source or ""), output_path=str(output or "")))
        if with_segments:
            db.add(Segment(job_id=job_id, idx=0, start=0.0, end=1.0,
                           source_text="Hello", translated_text="Xin chào"))
            db.add(Segment(job_id=job_id, idx=1, start=1.0, end=2.0,
                           source_text="Bye", translated_text=""))
        db.commit()
    return job_id


# --- migration ----------------------------------------------------------
def test_migration_adds_burn_subs(monkeypatch, tmp_path: Path):
    db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE jobs (id VARCHAR(36) PRIMARY KEY, "
                 "filename VARCHAR(512), status VARCHAR(32))")
    conn.execute("INSERT INTO jobs VALUES ('a','v.mp4','done')")
    conn.commit()
    conn.close()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    from src.config import get_settings
    from src.models.database import init_db
    get_settings.cache_clear()
    init_db()  # must not crash on old schema; must add burn_subs
    conn = sqlite3.connect(db_file)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    conn.close()
    assert "burn_subs" in cols
    get_settings.cache_clear()


# --- srt endpoint -------------------------------------------------------
def test_srt_vi_and_src(monkeypatch):
    c = _client(monkeypatch)
    jid = _seed_job(monkeypatch)
    r = c.get(f"/api/jobs/{jid}/srt?lang=vi")
    assert r.status_code == 200
    assert "Xin chào" in r.text  # translated preferred
    assert "Bye" in r.text       # fallback to source when untranslated
    r2 = c.get(f"/api/jobs/{jid}/srt?lang=src")
    assert r2.status_code == 200 and "Hello" in r2.text
    assert c.get("/api/jobs/nope/srt").status_code == 404
    jid2 = _seed_job(monkeypatch, job_id="jx2", with_segments=False)
    assert c.get(f"/api/jobs/{jid2}/srt").status_code == 404


# --- thumbnail ----------------------------------------------------------
def test_thumbnail_from_output(monkeypatch, tmp_path: Path):
    video = _make_video(tmp_path / "src.mp4")
    c = _client(monkeypatch)
    jid = _seed_job(monkeypatch, output=video)
    r = c.get(f"/api/jobs/{jid}/thumbnail")
    assert r.status_code == 200
    assert r.content[:2] == b"\xff\xd8"  # JPEG magic
    # Cached second call
    assert c.get(f"/api/jobs/{jid}/thumbnail").status_code == 200
    assert c.get("/api/jobs/nope/thumbnail").status_code == 404


# --- stats + retry ------------------------------------------------------
def test_stats(monkeypatch):
    c = _client(monkeypatch)
    _seed_job(monkeypatch, job_id="s1", status="done")
    _seed_job(monkeypatch, job_id="s2", status="failed")
    r = c.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 2 and body["done"] >= 1 and body["failed"] >= 1


def test_retry_resets_and_redispatches(monkeypatch, tmp_path: Path):
    video = _make_video(tmp_path / "src.mp4")
    c = _client(monkeypatch)
    jid = _seed_job(monkeypatch, status="failed", source=video)
    from src.worker.tasks import process_dubbing_job as task
    monkeypatch.setattr(task, "delay", MagicMock(return_value=None))
    r = c.post(f"/api/jobs/{jid}/retry")
    assert r.status_code == 200 and r.json()["status"] == "queued"
    assert c.get(f"/api/jobs/{jid}").json()["status"] == "queued"
    task.delay.assert_called_once()


def test_retry_missing_source_gone(monkeypatch, tmp_path: Path):
    c = _client(monkeypatch)
    jid = _seed_job(monkeypatch, status="failed",
                    source=tmp_path / "gone.mp4")
    assert c.post(f"/api/jobs/{jid}/retry").status_code == 410
    assert c.post("/api/jobs/nope/retry").status_code == 404


# --- burn-in merger -----------------------------------------------------
def test_burn_subtitles(tmp_path: Path):
    from src.asr.srt_utils import SimpleSegment, segments_to_srt
    from src.video.merger import burn_subtitles

    video = _make_video(tmp_path / "v.mp4")
    srt = segments_to_srt(
        [SimpleSegment(idx=0, start=0.0, end=1.5, text="Xin chào")],
        tmp_path / "a.srt")
    out = burn_subtitles(video, srt, tmp_path / "burned.mp4")
    assert out.exists() and out.stat().st_size > 1000
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True)
    assert probe.stdout.strip() == "h264"


def test_merge_with_burn_flag(tmp_path: Path):
    import wave
    from src.asr.srt_utils import SimpleSegment, segments_to_srt
    from src.video.merger import merge_dubbed_video

    video = _make_video(tmp_path / "v.mp4")
    wav = tmp_path / "d.wav"
    import math
    import struct
    n = 44100 * 2
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(struct.pack("<" + "h" * n, *([1000] * n)))
    srt = segments_to_srt(
        [SimpleSegment(idx=0, start=0.0, end=1.0, text="Chào")],
        tmp_path / "b.srt")
    out = merge_dubbed_video(video, wav, tmp_path / "m.mp4",
                             burn_subs=True, subs_srt=srt)
    assert out.exists()
    # burn requested but SRT missing -> graceful fallback, still outputs
    out2 = merge_dubbed_video(video, wav, tmp_path / "m2.mp4",
                              burn_subs=True,
                              subs_srt=tmp_path / "missing.srt")
    assert out2.exists()


# --- new helpers --------------------------------------------------------
def test_duration_vi():
    from src.bot.helpers import duration_vi
    assert duration_vi(45) == "45 giây"
    assert duration_vi(125) == "2 phút 5 giây"
    assert duration_vi(120) == "2 phút"
    assert duration_vi(3720) == "1 giờ 2 phút"


def test_eta_text():
    from src.bot.helpers import eta_text
    assert eta_text(0, 10) == "đang ước tính..."
    assert eta_text(50, 60) == "còn ~1 phút"
    assert "còn ~" in eta_text(99, 300)


def test_is_valid_rate():
    from src.bot.helpers import is_valid_rate
    assert is_valid_rate("+20%") and is_valid_rate("-10%")
    assert is_valid_rate("+0%")
    assert not is_valid_rate("20%") and not is_valid_rate("fast")
    assert not is_valid_rate("")


def test_transcript_snippet_and_queue():
    from src.bot.helpers import queue_text, transcript_snippet
    segs = [{"translated": "Xin chào", "source": "Hello"},
            {"translated": "", "source": "Bye"}]
    snip = transcript_snippet(segs)
    assert "Xin chào" in snip and "Bye" in snip
    assert "chưa có" in transcript_snippet([])
    assert "không phải chờ" in queue_text(1)
    assert "2 job" in queue_text(3)


def test_edge_ssml_rate():
    from src.tts.edge_client import EdgeTTSClient
    c = EdgeTTSClient(default_voice="female", rate="+20%")
    ssml = c.to_ssml("Chào <bạn>", "vi-VN-HoaiMyNeural")
    assert "+20%" in ssml and "&lt;b" in ssml and "vi-VN-HoaiMyNeural" in ssml


# --- bot api_client new methods -----------------------------------------
class _FeatResp:
    def __init__(self, payload=None, content=b"", status=200):
        self._payload = payload
        self.content = content
        self.status_code = status
        self.headers = {}
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("e", request=None, response=self)


class _FeatClient:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params=None, headers=None):
        if url.endswith("/health"):
            return _FeatResp({"status": "ok", "redis": "ok",
                              "database": "ok"})
        if url.endswith("/segments"):
            return _FeatResp({"segments": [{"translated": "Chào"}]})
        if url.endswith("/srt"):
            assert params == {"lang": "vi"}
            return _FeatResp(content=b"1\n00:00:00,000 --> 1")
        if url.endswith("/thumbnail"):
            return _FeatResp(content=b"\xff\xd8fakejpg")
        if url.endswith("/api/stats"):
            return _FeatResp({"total": 3, "active": 1, "done": 2,
                              "failed": 0})
        raise AssertionError(url)

    def post(self, url, headers=None):
        assert url.endswith("/retry")
        return _FeatResp({"job_id": "j", "status": "queued"})


def test_bot_client_new_methods(monkeypatch):
    from src.bot import api_client as api_mod
    from src.bot.api_client import DubbingAPI

    monkeypatch.setattr(api_mod.httpx, "Client", _FeatClient)
    api = DubbingAPI("http://x")
    payload, ms = api.health()
    assert payload["status"] == "ok" and ms >= 0
    assert api.get_segments("j") == [{"translated": "Chào"}]
    data, fname = api.get_srt("j")
    assert fname == "j_vi.srt" and data.startswith(b"1\n")
    assert api.get_thumbnail("j") == b"\xff\xd8fakejpg"
    assert api.get_stats()["total"] == 3
    assert api.retry_job("j")["status"] == "queued"
