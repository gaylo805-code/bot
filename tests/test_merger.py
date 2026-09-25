"""Tests: merged video keeps video codec + has new audio track."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

from src.video.merger import merge_dubbed_video


def _make_video(path: Path, seconds: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}",
         "-f", "lavfi", "-i", f"color=c=red:s=160x120:d={seconds}",
         "-shortest", str(path)], check=True)


def _make_wav(path: Path, seconds: float = 2.0) -> None:
    import math
    import struct
    n = int(seconds * 44100)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(struct.pack(
            "<" + "h" * n,
            *([int(8000 * math.sin(2 * math.pi * 220 * i / 44100))
               for i in range(n)])))


def _probe(path: Path, selector: str) -> str:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", selector,
         "-show_entries", "stream=codec_name",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return p.stdout.strip()


def test_merge_dub_only(tmp_path: Path):
    src = tmp_path / "src.mp4"
    _make_video(src)
    dub = tmp_path / "dub.wav"
    _make_wav(dub)
    out = tmp_path / "out.mp4"
    res = merge_dubbed_video(src, dub, out, keep_background=False)
    assert res.exists()
    assert _probe(out, "v:0") != ""          # video track kept
    assert _probe(out, "a:0") != ""          # new audio present


def test_merge_missing_input_raises(tmp_path: Path):
    try:
        merge_dubbed_video(tmp_path / "no.mp4", tmp_path / "no.wav",
                           tmp_path / "o.mp4")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
