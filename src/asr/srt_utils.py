"""Audio extraction + SRT helpers built on ffmpeg + srt libs."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimpleSegment:
    """Lightweight segment used across ASR/translate/TTS pipeline."""

    idx: int
    start: float
    end: float
    text: str
    language: str = ""
    translated: str = ""


def extract_audio(video_path: str | Path, wav_path: str | Path) -> Path:
    """Extract mono 16kHz WAV from video for Whisper.

    Args:
        video_path: input video file.
        wav_path: destination wav file.

    Returns:
        Path to the created wav file.

    Raises:
        FileNotFoundError: if input video missing.
        RuntimeError: if ffmpeg fails.
    """
    video_path = Path(video_path)
    wav_path = Path(wav_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_path),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not wav_path.exists():
        raise RuntimeError(f"ffmpeg extract_audio failed: {proc.stderr.strip()}")
    return wav_path


def _fmt_ts(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[SimpleSegment], srt_path: str | Path) -> Path:
    """Write segments to a standard SRT file.

    Args:
        segments: list of segments with start/end/text (translated preferred
            if present else source text).
        srt_path: destination .srt file.

    Returns:
        Path to srt file.
    """
    srt_path = Path(srt_path)
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        text = seg.translated or seg.text
        lines.append(str(i))
        lines.append(f"{_fmt_ts(seg.start)} --> {_fmt_ts(seg.end)}")
        lines.append(text.strip() or "...")
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


def clean_srt(
    segments: list[SimpleSegment],
    min_duration: float = 0.5,
    merge_gap: float = 0.2,
) -> list[SimpleSegment]:
    """Clean segments: drop too-short empties, merge close gaps, fix overlaps.

    - Drop segments with empty text and duration < min_duration.
    - Merge consecutive segments when gap < merge_gap (text joined with space).
    - Clamp overlaps: next.start = max(next.start, prev.end).

    Args:
        segments: raw segments sorted by start.
        min_duration: minimum duration to keep an empty segment.
        merge_gap: gap threshold (seconds) to merge neighbours.

    Returns:
        New cleaned list with re-indexed idx.
    """
    segs = sorted(segments, key=lambda s: s.start)
    # Fix overlaps first
    for prev, cur in zip(segs, segs[1:]):
        if cur.start < prev.end:
            cur.start = prev.end
            if cur.end <= cur.start:
                cur.end = cur.start + min_duration
    # Drop degenerate empty segments
    kept: list[SimpleSegment] = []
    for s in segs:
        dur = s.end - s.start
        if dur <= 0:
            continue
        if not s.text.strip() and dur < min_duration:
            continue
        kept.append(s)
    # Merge close neighbours
    merged: list[SimpleSegment] = []
    for s in kept:
        if merged and (s.start - merged[-1].end) < merge_gap:
            last = merged[-1]
            last.text = (last.text + " " + s.text).strip()
            if s.translated or last.translated:
                last.translated = (
                    (last.translated + " " + s.translated).strip()
                )
            last.end = max(last.end, s.end)
        else:
            merged.append(
                SimpleSegment(
                    idx=len(merged), start=s.start, end=s.end,
                    text=s.text.strip(), language=s.language,
                    translated=s.translated,
                )
            )
    for i, s in enumerate(merged):
        s.idx = i
    return merged
