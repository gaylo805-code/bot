"""Merge dubbed audio back into video with optional background music."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from loguru import logger

from src.video.ffmpeg_utils import run_ffmpeg


def _has_audio_stream(video: Path) -> bool:
    import subprocess

    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    return bool(proc.stdout.strip())


def _escape_subs_path(srt: Path) -> str:
    """Escape an .srt path for the ffmpeg subtitles filter."""
    return (
        str(srt.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    )


def burn_subtitles(
    video_in: str | Path, srt_path: str | Path, video_out: str | Path
) -> Path:
    """Burn an SRT into video (libass, DejaVu Sans, bottom-center).

    Args:
        video_in: muxed video.
        srt_path: Vietnamese subtitle file.
        video_out: destination (re-encoded with libx264).

    Returns:
        Path to subtitled video.

    Raises:
        FileNotFoundError: missing inputs. FFmpegError: filter failed.
    """
    video_in, srt_path, video_out = Path(video_in), Path(srt_path), Path(video_out)
    if not video_in.exists():
        raise FileNotFoundError(f"Video not found: {video_in}")
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT not found: {srt_path}")
    video_out.parent.mkdir(parents=True, exist_ok=True)
    style = (
        "FontName=DejaVu Sans,FontSize=16,"
        "PrimaryColour=&HFFFFFF&,OutlineColour=&H80000000&,"
        "BorderStyle=1,Outline=2,Shadow=0,MarginV=28"
    )
    run_ffmpeg([
        "ffmpeg", "-y", "-v", "error", "-i", str(video_in),
        "-vf", f"subtitles='{_escape_subs_path(srt_path)}':force_style='{style}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "copy", str(video_out),
    ])
    return video_out


def merge_dubbed_video(
    video_path: str | Path,
    dubbed_audio_path: str | Path,
    output_path: str | Path,
    keep_background: bool = False,
    bg_volume: float = 0.15,
    burn_subs: bool = False,
    subs_srt: str | Path | None = None,
) -> Path:
    """Mux dubbed audio onto video (copy video codec).

    Steps:
      1. If keep_background: extract original audio as background bed
         (demucs separation is optional; we fall back to low-volume mix
         of the original track when demucs is unavailable).
      2. ffmpeg -i video -i dubbed -c:v copy -map 0:v:0 -map 1:a:0
         -shortest output.mp4
      3. If keep_background: amix dubbed + lowered original.

    Args:
        video_path: source video.
        dubbed_audio_path: timeline wav from audio_builder.
        output_path: final mp4.
        keep_background: whether to keep original audio at low volume.
        bg_volume: background volume 0-1.
        burn_subs: burn subs_srt into the picture (libass re-encode).
        subs_srt: Vietnamese .srt for burn-in (required when burn_subs).

    Returns:
        Path to output video.
    """
    video_path = Path(video_path)
    dubbed_audio_path = Path(dubbed_audio_path)
    output_path = Path(output_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not dubbed_audio_path.exists():
        raise FileNotFoundError(f"Dubbed audio not found: {dubbed_audio_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = output_path
    if burn_subs:
        # Mux to a temp file first, burn-in as the final pass.
        merged = output_path.with_name(f"{output_path.stem}_nomux.mp4")

    if not keep_background or not _has_audio_stream(video_path):
        if keep_background:
            logger.warning("No source audio stream; merging dub only")
        run_ffmpeg([
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video_path), "-i", str(dubbed_audio_path),
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", str(merged),
        ])
        return _maybe_burn(merged, output_path, burn_subs, subs_srt)

    # keep_background=True: try demucs separation, else plain original.
    bg_track = output_path.parent / f"{output_path.stem}_bg.aac"
    demucs_dir = output_path.parent / "demucs_out"
    bg_src: Path = video_path
    try:
        import demucs  # noqa: F401
        from src.video.ffmpeg_utils import run_ffmpeg as _rf
        logger.info("demucs found - attempting vocal/accompaniment split")
        # sys.executable: same interpreter that has demucs installed.
        _rf([sys.executable, "-m", "demucs", "--two-stems=vocals",
             "-o", str(demucs_dir),
             str(video_path)])
        # `--two-stems=vocals` emits "no_vocals.wav" (never "accompaniment.wav"),
        # so match the real stem name; keep the old name as a safety net.
        candidates = [
            p for p in demucs_dir.rglob("*.wav")
            if p.stem in {"no_vocals", "accompaniment"}
        ]
        if candidates:
            bg_src = candidates[0]
            logger.info(f"Background bed from demucs: {bg_src.name}")
        else:
            logger.warning(
                "demucs produced no accompaniment stem; using original audio"
            )
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.warning(f"demucs unavailable, using original audio as bg: {exc}")

    # Extract (possibly demucs) background bed ...
    tmp_bg = output_path.parent / f"{output_path.stem}_bg_raw.m4a"
    run_ffmpeg(["ffmpeg", "-y", "-v", "error", "-i", str(bg_src),
                "-map", "0:a:0", "-c:a", "aac", str(tmp_bg)])
    # ... then mix: dubbed full volume + bg at bg_volume
    run_ffmpeg([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_path), "-i", str(dubbed_audio_path),
        "-i", str(tmp_bg),
        "-filter_complex",
        f"[1:a]volume=1.0[d];[2:a]volume={bg_volume}[b];"
        "[d][b]amix=inputs=2:duration=shortest:dropout_transition=0[m]",
        "-map", "0:v:0", "-map", "[m]",
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(merged),
    ])
    for p in (tmp_bg, bg_track):
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass
    # Stems are large (tens of MB); drop them so storage/outputs stays small.
    try:
        shutil.rmtree(demucs_dir, ignore_errors=True)
    except OSError:
        pass
    return _maybe_burn(merged, output_path, burn_subs, subs_srt)


def _maybe_burn(
    merged: Path,
    output_path: Path,
    burn_subs: bool,
    subs_srt: str | Path | None,
) -> Path:
    """Finalize output: burn subtitles when requested, else rename temp."""
    if not burn_subs:
        if merged != output_path:
            merged.replace(output_path)
        return output_path
    if not subs_srt or not Path(subs_srt).exists():
        logger.warning("burn_subs requested but no SRT found; skipping burn-in")
        if merged != output_path:
            merged.replace(output_path)
        return output_path
    try:
        burn_subtitles(merged, Path(subs_srt), output_path)
    finally:
        if merged != output_path:
            merged.unlink(missing_ok=True)
    return output_path
