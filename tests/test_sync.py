"""A/V sync checks: the drift guard for lip-synced output.

Why this exists: on a real 13-minute Facebook clip the final output came
out 790s of video against 754s of audio - 36s of drift, so the lips pulled
away from the voice. `-c:v copy` cannot cut a stream at a non-keyframe, so
`-shortest` only ended the video at the *next* keyframe.

Where the keyframe lands is encoder-dependent, so these tests pin the
deterministic parts: the decision to re-encode, and the sync check itself.
The overshoot itself was reproduced off-fixture (240s AV1 with a 240-frame
GOP, 200s audio: copy gave 230s/200s, re-encode gave 200.0s/200.0s).
"""
import subprocess

import pytest

from src.video.ffmpeg_utils import SYNC_TOLERANCE_S
from src.video.lipsync import av_sync_drift, verify_av_sync


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args],
                   capture_output=True, check=True, timeout=300)


@pytest.fixture
def video6(tmp_path):
    """6s of video with no audio track."""
    p = tmp_path / "v6.mp4"
    _ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=6",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p))
    return p


@pytest.fixture
def video_audio(video6, tmp_path):
    """A 6s file whose audio is deliberately 2s short (no -shortest)."""
    short = tmp_path / "short.m4a"
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:a", "aac",
        str(short))
    out = tmp_path / "mismatched.mp4"
    _ff("-i", str(video6), "-i", str(short), "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0", str(out))
    return out


def test_av_sync_drift_detects_mismatch(video_audio):
    """A 6s/4s file must read as roughly 2s of drift."""
    drift = av_sync_drift(video_audio)
    assert 1.5 < drift < 2.5, drift


def test_verify_av_sync_fails_on_mismatch(video_audio):
    ok, drift = verify_av_sync(video_audio)
    assert ok is False
    assert drift > SYNC_TOLERANCE_S


def test_verify_av_sync_passes_when_lengths_match(tmp_path):
    """Same-length streams must come back clean, not merely unprobed."""
    both = tmp_path / "ok.m4a"
    _ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=5",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", str(both))
    ok, drift = verify_av_sync(both)
    assert ok, f"drift {drift:.3f}s"
    assert drift <= SYNC_TOLERANCE_S


def test_verify_reports_probe_failure_as_not_ok(tmp_path):
    """A missing file must not be mistaken for a pass."""
    ok, drift = verify_av_sync(tmp_path / "does-not-exist.mp4")
    assert ok is False
    assert drift == float("inf")


def test_merge_reencodes_only_when_dub_is_shorter(video6, tmp_path):
    """The cheap `-c:v copy` path must stay for already-matching lengths."""
    from src.video.merger import _needs_reencode_for_trim

    short = tmp_path / "short.m4a"
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:a", "aac",
        str(short))
    assert _needs_reencode_for_trim(video6, short) is True

    same = tmp_path / "same.m4a"
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-c:a", "aac",
        str(same))
    assert _needs_reencode_for_trim(video6, same) is False


def test_reencode_path_lands_within_tolerance(video6, tmp_path):
    """End to end: re-encoding must bring the merge inside tolerance."""
    short = tmp_path / "short.m4a"
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:a", "aac",
        str(short))
    out = tmp_path / "merged.mp4"
    _ff("-i", str(video6), "-i", str(short), "-c:v", "libx264",
        "-preset", "veryfast", "-c:a", "aac", "-map", "0:v:0",
        "-map", "1:a:0", "-shortest", str(out))
    ok, drift = verify_av_sync(out)
    assert ok, f"drift {drift:.3f}s exceeds {SYNC_TOLERANCE_S}s"
