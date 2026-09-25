"""Lip-sync post-processing: Wav2Lip (mouth) + GFPGAN (face restore).

Runs AFTER merge_video(). Every stage degrades gracefully:
  - disabled / missing checkpoint -> return merged video unchanged
  - no face detected              -> skip, keep dubbed video as-is
  - stage crash                   -> keep previous artifact, never fail the job

Wav2Lip runs as an isolated subprocess (its deps stay out of our venv);
GFPGAN runs in-process. Both use CUDA when available.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

from loguru import logger

from src.config import get_settings
from src.video.ffmpeg_utils import run_ffmpeg

GFPGAN_RELEASE_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)


# -- availability ---------------------------------------------------------
def is_available() -> tuple[bool, str]:
    """Check Wav2Lip repo + checkpoint presence.

    Returns:
        (ok, reason) - reason is human-readable when not ok.
    """
    settings = get_settings()
    repo = settings.wav2lip_dir
    if not (repo / "inference.py").exists():
        return False, f"Wav2Lip repo missing at {repo} (run scripts/setup_lipsync.sh)"
    if not settings.wav2lip_checkpoint.exists():
        return False, (f"Checkpoint missing at {settings.wav2lip_checkpoint} "
                       "(run scripts/setup_lipsync.sh)")
    return True, "ok"


def describe_lipsync() -> str:
    """One-liner for docs/status (no secrets)."""
    settings = get_settings()
    if not settings.wav2lip_enabled:
        return "Lip-sync: OFF (WAV2LIP_ENABLED=false)"
    ok, reason = is_available()
    if not ok:
        return f"Lip-sync: unavailable ({reason})"
    gfpgan = " + GFPGAN" if settings.gfpgan_enabled else " (GFPGAN off)"
    return f"Lip-sync: Wav2Lip{gfpgan} on CUDA"


# -- face pre-check --------------------------------------------------------
_retinaface = None


def _get_detector(device: str = "cuda"):
    """Lazy RetinaFace detector (downloads ~104MB weights on first use).

    Probes half precision at init: some torch builds fail only at detect
    time ('requires grad' numpy error), so fall back to full precision.
    """
    global _retinaface
    if _retinaface is None:
        from facexlib.detection import init_detection_model

        import numpy as np
        import torch

        last_exc: Exception | None = None
        for half in (True, False):
            try:
                det = init_detection_model(
                    "retinaface_resnet50", half=half, device=device)
                with torch.no_grad():
                    det.detect_faces(np.zeros((64, 64, 3), dtype=np.uint8))
                _retinaface = det
                logger.info(f"RetinaFace ready (half={half}, {device})")
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"RetinaFace half={half} unusable: {exc}")
                last_exc = exc
        if _retinaface is None:
            raise RuntimeError(f"RetinaFace unavailable: {last_exc}")
    return _retinaface


def has_face(
    video_path: str | Path, samples: int = 5, min_hits: int = 1,
    device: str = "cuda",
) -> bool:
    """RetinaFace check on sampled frames (same family Wav2Lip trusts).

    Args:
        video_path: video to inspect.
        samples: frames to sample evenly.
        min_hits: frames with a face required.
        device: 'cuda' (L4) or 'cpu'.

    Returns:
        True when a face is found. Fail-open True when the detector
        itself cannot load, so real faces are never skipped
        (Wav2Lip then decides and errors loudly if truly faceless).
    """
    import cv2

    video_path = Path(video_path)
    try:
        detector = _get_detector(device)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Face detector unavailable ({exc}); assuming present")
        return True
    import torch

    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return True  # unreadable -> let Wav2Lip decide
        hits = 0
        for i in range(samples):
            pos = int((i + 0.5) / samples * total)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            try:
                # facexlib returns one Nx15 array (empty when faceless).
                with torch.no_grad():
                    found = detector.detect_faces(frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Face detect failed ({exc}); assuming present")
                return True
            if found is not None and len(found) > 0:
                hits += 1
                if hits >= min_hits:
                    return True
        return hits >= min_hits
    finally:
        cap.release()


# -- Wav2Lip (subprocess, isolated deps) ------------------------------------
def get_video_fps(video_path: str | Path, default: float = 25.0) -> float:
    """Probe a video's frame rate via ffprobe (avg_frame_rate, as a fraction).

    Wav2Lip assumes 25fps internally unless told otherwise; on sources with
    a different fps (24, 29.97, 60...) that mismatch alone is enough to
    make the generated mouth movement drift out of sync with the audio
    over the length of the clip. Always pass the real fps explicitly.
    """
    import subprocess as _sp

    proc = _sp.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    raw = proc.stdout.strip()
    try:
        if "/" in raw:
            num, den = raw.split("/")
            den = float(den)
            return float(num) / den if den else default
        return float(raw) if raw else default
    except (ValueError, ZeroDivisionError):
        return default


def build_wav2lip_cmd(
    face_video: Path, audio_wav: Path, out_mp4: Path, static: bool = False
) -> list[str]:
    """Build the Wav2Lip inference argv (cwd must be the repo dir)."""
    settings = get_settings()
    fps = get_video_fps(face_video)
    cmd = [
        sys.executable, "inference.py",
        "--checkpoint_path", str(settings.wav2lip_checkpoint.resolve()),
        "--face", str(face_video.resolve()),
        "--audio", str(audio_wav.resolve()),
        "--outfile", str(out_mp4.resolve()),
        "--pads", "0", "10", "0", "0",
        "--face_det_batch_size", "16",
        "--wav2lip_batch_size", "128",
        "--resize_factor", "1",
        # Explicit fps: without this Wav2Lip defaults to 25fps internally,
        # which silently drifts the generated mouth timing on any source
        # that isn't actually 25fps (24, 29.97, 60...).
        "--fps", f"{fps:.3f}",
    ]
    if static or settings.wav2lip_static:
        cmd += ["--static", "True"]
    return cmd


def run_wav2lip(
    face_video: str | Path,
    audio_wav: str | Path,
    out_mp4: str | Path,
    timeout: int | None = 1800,
) -> Path:
    """Run Wav2Lip inference, raising RuntimeError with log tail on failure.

    Args:
        face_video: merged dubbed video (face source).
        audio_wav: dubbed timeline wav (mouth driver).
        out_mp4: lip-synced video destination.
        timeout: subprocess timeout seconds.

    Returns:
        Path to lip-synced video.
    """
    settings = get_settings()
    face_video, audio_wav, out_mp4 = (
        Path(face_video), Path(audio_wav), Path(out_mp4))
    for p, name in ((face_video, "face video"), (audio_wav, "audio")):
        if not p.exists():
            raise FileNotFoundError(f"Wav2Lip {name} not found: {p}")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_wav2lip_cmd(face_video, audio_wav, out_mp4)
    logger.info(f"Wav2Lip inference -> {out_mp4.name} "
                f"(CWD={settings.wav2lip_dir})")
    try:
        proc = subprocess.run(
            cmd, cwd=str(settings.wav2lip_dir),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Wav2Lip timed out after {timeout}s") from exc
    tail = (proc.stderr or proc.stdout or "")[-1500:]
    if proc.returncode != 0 or not out_mp4.exists():
        raise RuntimeError(f"Wav2Lip failed (exit {proc.returncode}): {tail}")
    if "face not detected" in tail.lower() or "no face" in tail.lower():
        raise RuntimeError(f"Wav2Lip found no face: {tail}")
    logger.info(f"Wav2Lip done: {out_mp4} ({out_mp4.stat().st_size} bytes)")
    return out_mp4


# -- GFPGAN (in-process face restore) ---------------------------------------
def ensure_gfpgan_weights() -> Path:
    """Download GFPGANv1.4 weights once (cached under ./models)."""
    settings = get_settings()
    dest = settings.gfpgan_path
    if dest.exists() and dest.stat().st_size > 10_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading GFPGAN weights -> {dest} (~350MB)")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(GFPGAN_RELEASE_URL, str(tmp))
    tmp.replace(dest)
    return dest


def _load_gfpgan(device: str = "cuda"):
    """Instantiate GFPGANer (import-guarded for testability)."""
    from gfpgan import GFPGANer

    weights = ensure_gfpgan_weights()
    try:
        return GFPGANer(
            model_path=str(weights), upscale=2, arch="clean",
            channel_multiplier=2, bg_upsampler=None, device=device,
        )
    except TypeError:
        # Older gfpgan without a device kwarg (uses CUDA when available).
        return GFPGANer(
            model_path=str(weights), upscale=2, arch="clean",
            channel_multiplier=2, bg_upsampler=None,
        )


def run_gfpgan(
    video_in: str | Path,
    video_out: str | Path,
    device: str = "cuda",
) -> Path:
    """Restore every frame with GFPGAN, keep original size + audio.

    Args:
        video_in: lip-synced video (face may be soft).
        video_out: sharpened destination.
        device: 'cuda' (L4) or 'cpu'.

    Returns:
        Path to restored video.

    Raises:
        RuntimeError: when the enhancer cannot run (caller falls back).
    """
    import cv2

    video_in, video_out = Path(video_in), Path(video_out)
    if not video_in.exists():
        raise FileNotFoundError(f"GFPGAN input not found: {video_in}")
    video_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        enhancer = _load_gfpgan(device)
    except ImportError as exc:
        raise RuntimeError(f"gfpgan package missing ({exc}); "
                           "pip install gfpgan") from exc

    cap = cv2.VideoCapture(str(video_in))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video dims: {video_in}")
    silent = video_out.with_name(f"{video_out.stem}_silent.mp4")
    writer = cv2.VideoWriter(
        str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frames = enhanced = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames += 1
            try:
                _, _, restored = enhancer.enhance(
                    frame[:, :, ::-1], has_aligned=False,
                    only_center_face=False, paste_back=True)
                restored_bgr = cv2.resize(restored[:, :, ::-1],
                                          (width, height))
                enhanced += 1
            except Exception as exc:  # noqa: BLE001 - keep original frame
                logger.warning(f"GFPGAN frame {frames} failed: {exc}")
                restored_bgr = frame
            writer.write(restored_bgr)
    finally:
        cap.release()
        writer.release()
    logger.info(f"GFPGAN enhanced {enhanced}/{frames} frames")
    if frames == 0 or not silent.exists():
        raise RuntimeError("GFPGAN produced no frames")
    # Re-attach original audio (frame loop is video-only).
    run_ffmpeg([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(silent), "-i", str(video_in),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-shortest", str(video_out),
    ])
    silent.unlink(missing_ok=True)
    return video_out


# -- orchestrator (called by the Celery pipeline) ----------------------------
def post_process_lipsync(
    merged_mp4: str | Path,
    dubbed_wav: str | Path,
    out_mp4: str | Path,
    enable: bool = True,
) -> Path:
    """Wav2Lip + GFPGAN after merge. Never raises: returns best artifact.

    Args:
        merged_mp4: dubbed video from merge_video().
        dubbed_wav: timeline wav driving the mouth.
        out_mp4: final lip-synced destination.
        enable: per-job toggle (global kill-switch is WAV2LIP_ENABLED).

    Returns:
        out_mp4 on success, else merged_mp4 (or input when inputs missing).
    """
    settings = get_settings()
    merged, wav, out = Path(merged_mp4), Path(dubbed_wav), Path(out_mp4)
    if not enable or not settings.wav2lip_enabled:
        return merged
    if not merged.exists() or not wav.exists():
        logger.warning("Lip-sync skipped: inputs missing")
        return merged
    ok, reason = is_available()
    if not ok:
        logger.warning(f"Lip-sync skipped: {reason}")
        return merged
    try:
        if not has_face(merged):
            logger.info("Lip-sync skipped: no face detected, keeping dub as-is")
            return merged
    except Exception as exc:  # noqa: BLE001 - pre-check must not block
        logger.warning(f"Face pre-check failed ({exc}); trying Wav2Lip anyway")

    work = out.parent / f"{out.stem}_lipwork"
    work.mkdir(parents=True, exist_ok=True)
    try:
        lip_raw = run_wav2lip(merged, wav, work / "wav2lip.mp4")
    except Exception as exc:  # noqa: BLE001 - keep dubbed video
        logger.error(f"Wav2Lip failed, keeping dub as-is: {exc}")
        return merged
    # Wav2Lip generated the mouth movement to match `wav` (dubbed_wav)
    # exactly, NOT `merged` (which, when keep_background=True, is a mix
    # of the dub + lowered background music). Re-attaching audio from
    # `merged` there would play a *mixed/offset* track against lip motion
    # that was driven by the clean dub -> visible mouth/audio mismatch.
    # Always re-mux the same `wav` file that drove Wav2Lip.
    # Also re-encode video (no `-c:v copy`): copying a stream out of
    # Wav2Lip's output while pairing it with another container's timebase
    # can drift the two out of sync over the length of the clip.
    lip_muxed = work / "wav2lip_muxed.mp4"
    try:
        run_ffmpeg([
            "ffmpeg", "-y", "-v", "error",
            "-i", str(lip_raw), "-i", str(wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-shortest", str(lip_muxed),
        ])
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Lip-sync audio mux failed ({exc}); using raw output")
        lip_muxed = lip_raw

    if not settings.gfpgan_enabled:
        lip_muxed.replace(out)
        return out
    try:
        return run_gfpgan(lip_muxed, out)
    except Exception as exc:  # noqa: BLE001 - soft face beats no video
        logger.error(f"GFPGAN failed, keeping Wav2Lip output: {exc}")
        try:
            lip_muxed.replace(out)
        except OSError:
            return merged
        return out
