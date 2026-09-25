"""Tests: Wav2Lip+GFPGAN post-processing (mocked heavy stages)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_clip(path: Path, seconds: float = 2.0,
               color: str = "blue") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}",
         "-f", "lavfi", "-i", f"color=c={color}:s=160x120:d={seconds}",
         "-shortest", str(path)], check=True)
    return path


def _make_wav(path: Path, seconds: float = 2.0) -> Path:
    import math
    import struct
    import wave
    n = int(seconds * 44100)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(struct.pack(
            "<" + "h" * n,
            *([int(8000 * math.sin(2 * i / 100)) for i in range(n)])))
    return path


# --- availability / describe ---------------------------------------------
def test_is_available_missing(monkeypatch, tmp_path: Path):
    from src.config import get_settings
    from src.video import lipsync as lip

    monkeypatch.setenv("WAV2LIP_DIR", str(tmp_path / "nothere"))
    monkeypatch.setenv("WAV2LIP_CHECKPOINT", str(tmp_path / "no.pth"))
    get_settings.cache_clear()
    ok, reason = lip.is_available()
    assert ok is False and "missing" in reason
    get_settings.cache_clear()


def test_describe_states(monkeypatch, tmp_path: Path):
    from src.config import get_settings
    from src.video import lipsync as lip

    monkeypatch.setenv("WAV2LIP_ENABLED", "false")
    get_settings.cache_clear()
    assert "OFF" in lip.describe_lipsync()
    monkeypatch.setenv("WAV2LIP_ENABLED", "true")
    monkeypatch.setenv("WAV2LIP_DIR", str(tmp_path / "x"))
    get_settings.cache_clear()
    assert "unavailable" in lip.describe_lipsync()
    get_settings.cache_clear()


# --- face pre-check --------------------------------------------------------
def test_has_face_negative_real_detector(tmp_path: Path):
    from src.video.lipsync import has_face
    clip = _make_clip(tmp_path / "blue.mp4")
    assert has_face(clip, samples=3) is False


def test_has_face_positive_mocked(tmp_path: Path, monkeypatch):
    from src.video import lipsync as lip

    clip = _make_clip(tmp_path / "blue.mp4")

    class _FakeDetector:
        def detect_faces(self, frame):
            import numpy as np
            return np.array([[10, 10, 90, 90, 0.99] + [0] * 10])

    monkeypatch.setattr(lip, "_get_detector", lambda *a, **k: _FakeDetector())
    assert lip.has_face(clip, samples=2) is True


def test_has_face_detector_broken_fail_open(tmp_path: Path, monkeypatch):
    from src.video import lipsync as lip

    clip = _make_clip(tmp_path / "blue.mp4")

    def _broken(*a, **k):
        raise ImportError("No module named 'facexlib'")

    monkeypatch.setattr(lip, "_get_detector", _broken)
    assert lip.has_face(clip) is True


# --- wav2lip command / runner ----------------------------------------------
def test_build_wav2lip_cmd(tmp_path: Path, monkeypatch):
    from src.config import get_settings
    from src.video import lipsync as lip

    monkeypatch.setenv("WAV2LIP_CHECKPOINT", str(tmp_path / "w.pth"))
    get_settings.cache_clear()
    cmd = lip.build_wav2lip_cmd(tmp_path / "f.mp4", tmp_path / "a.wav",
                                tmp_path / "o.mp4")
    assert cmd[1] == "inference.py"
    assert "--checkpoint_path" in cmd and "--face" in cmd
    assert "--static" not in cmd
    cmd2 = lip.build_wav2lip_cmd(tmp_path / "f.mp4", tmp_path / "a.wav",
                                 tmp_path / "o.mp4", static=True)
    assert "--static" in cmd2
    get_settings.cache_clear()


def test_run_wav2lip_success_and_failure(tmp_path: Path, monkeypatch):
    from src.video import lipsync as lip

    face = _make_clip(tmp_path / "f.mp4")
    audio = _make_wav(tmp_path / "a.wav")
    out = tmp_path / "o.mp4"

    class _OK:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        Path(out).write_bytes(b"lipvideo")
        return _OK()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert lip.run_wav2lip(face, audio, out) == out

    class _Bad:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Bad())
    with pytest.raises(RuntimeError, match="boom"):
        lip.run_wav2lip(face, audio, tmp_path / "o2.mp4")

    with pytest.raises(FileNotFoundError):
        lip.run_wav2lip(tmp_path / "missing.mp4", audio, out)


# --- orchestrator fallbacks -------------------------------------------------
def test_post_process_disabled_or_missing(tmp_path: Path):
    from src.video import lipsync as lip

    merged = _make_clip(tmp_path / "m.mp4")
    wav = _make_wav(tmp_path / "a.wav")
    out = tmp_path / "final.mp4"
    assert lip.post_process_lipsync(merged, wav, out, enable=False) == merged
    assert lip.post_process_lipsync(tmp_path / "no.mp4", wav, out) == tmp_path / "no.mp4"


def test_post_process_no_face_skips(tmp_path: Path, monkeypatch):
    from src.video import lipsync as lip

    merged = _make_clip(tmp_path / "m.mp4")
    wav = _make_wav(tmp_path / "a.wav")
    monkeypatch.setattr(lip, "has_face", lambda *a, **k: False)
    monkeypatch.setattr(lip, "is_available", lambda: (True, "ok"))
    out = tmp_path / "final.mp4"
    assert lip.post_process_lipsync(merged, wav, out) == merged
    assert not out.exists()


def test_post_process_wav2lip_crash_keeps_dub(tmp_path: Path, monkeypatch):
    from src.video import lipsync as lip

    merged = _make_clip(tmp_path / "m.mp4")
    wav = _make_wav(tmp_path / "a.wav")
    monkeypatch.setattr(lip, "has_face", lambda *a, **k: True)
    monkeypatch.setattr(lip, "is_available", lambda: (True, "ok"))

    def _boom(*a, **k):
        raise RuntimeError("cuda oom")

    monkeypatch.setattr(lip, "run_wav2lip", _boom)
    out = tmp_path / "final.mp4"
    assert lip.post_process_lipsync(merged, wav, out) == merged


def _write_lip(output: Path, payload: bytes = b"lip") -> Path:
    output.write_bytes(payload)
    return output


def test_post_process_gfpgan_crash_keeps_lip(tmp_path: Path, monkeypatch):
    from src.video import lipsync as lip

    merged = _make_clip(tmp_path / "m.mp4")
    wav = _make_wav(tmp_path / "a.wav")
    monkeypatch.setattr(lip, "has_face", lambda *a, **k: True)
    monkeypatch.setattr(lip, "is_available", lambda: (True, "ok"))
    monkeypatch.setattr(lip, "run_wav2lip",
                        lambda f, a, o, **k: _write_lip(Path(o)))

    def _gfail(*a, **k):
        raise RuntimeError("gfpgan exploded")

    monkeypatch.setattr(lip, "run_gfpgan", _gfail)
    out = tmp_path / "final.mp4"
    assert lip.post_process_lipsync(merged, wav, out) == out
    assert out.exists()


def test_post_process_success(tmp_path: Path, monkeypatch):
    from src.video import lipsync as lip

    merged = _make_clip(tmp_path / "m.mp4")
    wav = _make_wav(tmp_path / "a.wav")
    monkeypatch.setattr(lip, "has_face", lambda *a, **k: True)
    monkeypatch.setattr(lip, "is_available", lambda: (True, "ok"))
    monkeypatch.setattr(lip, "run_wav2lip",
                        lambda f, a, o, **k: _write_lip(Path(o)))
    import shutil
    # run_ffmpeg is used for audio mux: stub it to just copy input video.
    monkeypatch.setattr(lip, "run_ffmpeg",
                        lambda args: shutil.copy(args[5], args[-1]))
    monkeypatch.setattr(lip, "run_gfpgan",
                        lambda i, o, **k: _write_lip(Path(o), b"sharp"))
    out = tmp_path / "final.mp4"
    assert lip.post_process_lipsync(merged, wav, out) == out
    assert out.read_bytes() == b"sharp"


# --- gfpgan loader paths -----------------------------------------------------
def test_run_gfpgan_missing_package(monkeypatch, tmp_path: Path):
    from src.video import lipsync as lip

    video = _make_clip(tmp_path / "v.mp4")

    def _no_pkg(*a, **k):
        raise ImportError("No module named 'gfpgan'")

    monkeypatch.setattr(lip, "_load_gfpgan", _no_pkg)
    with pytest.raises(RuntimeError, match="gfpgan package missing"):
        lip.run_gfpgan(video, tmp_path / "o.mp4")


def test_ensure_weights_existing_and_download(monkeypatch, tmp_path: Path):
    from src.config import get_settings
    from src.video import lipsync as lip

    big = tmp_path / "G.pth"
    big.write_bytes(b"x" * 20_000_000)
    monkeypatch.setenv("GFPGAN_PATH", str(big))
    get_settings.cache_clear()
    assert lip.ensure_gfpgan_weights() == big

    small = tmp_path / "small.pth"
    small.write_bytes(b"tiny")
    monkeypatch.setenv("GFPGAN_PATH", str(small))
    get_settings.cache_clear()
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve",
                        lambda url, dest: Path(dest).write_bytes(b"weights"))
    assert lip.ensure_gfpgan_weights() == small
    get_settings.cache_clear()


# --- upload toggle persisted ---------------------------------------------------
def test_upload_lipsync_flag(monkeypatch):
    from fastapi.testclient import TestClient
    from src.main import create_app
    from src.config import get_settings
    from src.models.database import get_session_factory
    from src.models.job import Job
    from src.worker.tasks import process_dubbing_job as task

    get_settings.cache_clear()
    monkeypatch.setattr(task, "delay", MagicMock(return_value=None))
    c = TestClient(create_app())
    r = c.post("/api/upload",
               files={"file": ("t.mp4", b"\x00" * 100, "video/mp4")},
               data={"lipsync": "false"})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    with get_session_factory()() as db:
        assert db.get(Job, jid).lipsync == 0
    r2 = c.post("/api/upload",
                files={"file": ("t.mp4", b"\x00" * 100, "video/mp4")})
    with get_session_factory()() as db:
        assert db.get(Job, r2.json()["job_id"]).lipsync == 1


# --- bot stage label -----------------------------------------------------------
def test_bot_lipsync_stage():
    from src.bot.helpers import stage_label, status_color
    assert "khớp môi" in stage_label("lipsync").lower()
    assert status_color("lipsync") == 0xFF6B9D
