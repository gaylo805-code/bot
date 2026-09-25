"""ffmpeg subprocess wrapper with logging."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from loguru import logger


class FFmpegError(RuntimeError):
    """Raised when ffmpeg exits non-zero."""


def run_ffmpeg(args: list[str], timeout: int | None = None) -> tuple[str, str]:
    """Run ffmpeg with check + stderr logging.

    Args:
        args: full argv (starting with 'ffmpeg').
        timeout: optional timeout seconds.

    Returns:
        (stdout, stderr).

    Raises:
        FFmpegError: on non-zero exit or timeout.
    """
    logger.debug(f"Running: {' '.join(args)}")
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffmpeg timed out: {exc}") from exc
    if proc.returncode != 0:
        logger.error(f"ffmpeg failed: {proc.stderr[-2000:]}")
        raise FFmpegError(f"ffmpeg exited {proc.returncode}: "
                          f"{proc.stderr.strip()[-1000:]}")
    return proc.stdout, proc.stderr


# -- A/V sync probing --------------------------------------------------------
SYNC_TOLERANCE_S = 0.1


def probe_stream_durations(path: str | Path) -> dict[str, float]:
    """Return {video: seconds, audio: seconds} for a media file.

    Args:
        path: any file ffprobe can read.

    Returns:
        Mapping with 'video' and 'audio' keys; a missing stream reads 0.0.
    """
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {proc.stderr[-200:]}")
    out: dict[str, float] = {"video": 0.0, "audio": 0.0}
    for stream in json.loads(proc.stdout or "{}").get("streams", []):
        kind = stream.get("codec_type")
        if kind in out:
            try:
                out[kind] = max(out[kind], float(stream.get("duration") or 0.0))
            except (TypeError, ValueError):
                pass
    return out


def probe_duration(path: str | Path) -> float:
    """Return the longer of the video/audio stream durations."""
    d = probe_stream_durations(path)
    return max(d["video"], d["audio"])
