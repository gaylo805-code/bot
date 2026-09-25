"""Tests: extract_audio, clean_srt, SRT format."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.asr.srt_utils import SimpleSegment, clean_srt, extract_audio, segments_to_srt


def _make_video(path: Path, seconds: float = 2.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
           "-f", "lavfi", "-i", f"color=c=blue:s=160x120:d={seconds}",
           "-shortest", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return path


def test_extract_audio(tmp_path: Path):
    video = _make_video(tmp_path / "in.mp4")
    wav = extract_audio(video, tmp_path / "out.wav")
    assert wav.exists() and wav.stat().st_size > 1000


def test_segments_to_srt_format(tmp_path: Path):
    segs = [SimpleSegment(idx=0, start=1.0, end=2.5, text="Hello"),
            SimpleSegment(idx=1, start=3.0, end=4.0, text="World",
                          translated="Thế giới")]
    p = segments_to_srt(segs, tmp_path / "a.srt")
    txt = p.read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:02,500" in txt
    assert "Thế giới" in txt  # translated preferred
    assert txt.strip().splitlines()[0] == "1"


def test_clean_srt_filters_and_merges():
    segs = [
        SimpleSegment(idx=0, start=0.0, end=1.0, text="Hello"),
        SimpleSegment(idx=1, start=1.05, end=2.0, text="world"),  # gap 0.05 -> merge
        SimpleSegment(idx=2, start=5.0, end=5.1, text=""),        # empty+short -> drop
        SimpleSegment(idx=3, start=6.0, end=5.5, text="bad"),     # invalid -> drop
        SimpleSegment(idx=4, start=6.0, end=7.0, text="ok"),
    ]
    out = clean_srt(segs)
    texts = [s.text for s in out]
    assert any("Hello" in t and "world" in t for t in texts)
    assert all(s.end > s.start for s in out)
    assert [s.idx for s in out] == list(range(len(out)))


def test_clean_srt_fixes_overlap():
    segs = [SimpleSegment(idx=0, start=0.0, end=2.0, text="a"),
            SimpleSegment(idx=1, start=1.5, end=3.0, text="b")]
    out = clean_srt(segs, merge_gap=0.0)
    assert out[1].start >= out[0].end
