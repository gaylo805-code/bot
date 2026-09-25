"""ffmpeg subprocess wrapper with logging."""

from __future__ import annotations

import subprocess

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
