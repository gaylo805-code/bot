"""Build a dubbed timeline WAV from per-segment TTS + atempo correction."""

from __future__ import annotations

import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from src.asr.srt_utils import SimpleSegment
from src.config import get_settings
from src.tts.base import TTSClient
from src.tts.factory import get_tts_client
from src.tts.text_cleaner import clean_for_tts


def get_wav_duration(path: str | Path) -> float:
    """Return duration (seconds) of a WAV/MP3-ish file via wave or ffprobe."""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate() or 44100
            return frames / float(rate)
    except wave.Error:
        pass
    # Fallback: ffprobe for mp3 outputs
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def _atempo_chain(ratio: float) -> list[str]:
    """Build a chain of atempo filters (each supports 0.5-2.0)."""
    filters: list[str] = []
    r = ratio
    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r /= 0.5
    filters.append(f"atempo={r:.3f}")
    return filters


def stretch_to_fit(
    src: Path, dst: Path, target_dur: float, tolerance: float = 0.15
) -> Path:
    """Speed-adjust audio with ffmpeg atempo if deviation > tolerance.

    Args:
        src: TTS segment file.
        dst: destination adjusted file (may equal src copy).
        target_dur: desired duration (end - start).
        tolerance: relative deviation allowed before adjusting.

    Returns:
        Path to (possibly new) file with ~target_dur duration.
    """
    actual = get_wav_duration(src)
    if actual <= 0 or target_dur <= 0:
        return src
    dev = abs(actual - target_dur) / target_dur
    if dev <= tolerance:
        if src != dst:
            dst.write_bytes(src.read_bytes())
        return dst
    # tempo = actual/target: audio longer than slot -> speed up (>1.0)
    tempo = max(0.25, min(4.0, actual / target_dur))
    filt = ",".join(_atempo_chain(tempo))
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
           "-filter:a", filt, str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning(f"atempo failed ({proc.stderr.strip()}), using raw TTS")
        if src != dst:
            dst.write_bytes(src.read_bytes())
        return dst
    return dst


def _silent_wav(path: Path, duration: float, sr: int = 44100) -> None:
    import struct

    n = max(1, int(duration * sr))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


def _read_mono(wav_path: Path, sr: int = 44100) -> tuple[bytes, int]:
    """Read/convert any audio to mono 16-bit PCM bytes at sr via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(wav_path),
         "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le",
         "-f", "s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {wav_path}")
    return proc.stdout, sr


def build_dubbed_audio(
    segments: list[SimpleSegment],
    voice_id: str,
    output_wav: str | Path,
    client: TTSClient | None = None,
    max_workers: int = 4,
    sample_rate: int = 44100,
) -> Path:
    """Synthesize each segment (parallel) and lay them on the right timeline.

    - Each segment -> TTS file -> atempo if off by >15% -> placed at
      timestamp with silence padding.
    - One segment failure inserts silence instead of crashing pipeline.
    - Output mono 44.1kHz WAV.

    Args:
        segments: translated segments.
        voice_id: provider voice id (VieNeu id or edge voice/alias).
        output_wav: final timeline wav path.
        client: optional TTS client (provider from env if None).
        max_workers: ThreadPoolExecutor size.
        sample_rate: output sample rate.

    Returns:
        Path to output wav.
    """
    from src.config import get_settings as _gs
    settings = _gs()
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    client = client or get_tts_client()
    seg_dir = output_wav.parent / f"{output_wav.stem}_segs"
    seg_dir.mkdir(parents=True, exist_ok=True)

    total = max((s.end for s in segments), default=0.0)
    total = max(total, 0.5)

    def _one(seg: SimpleSegment) -> tuple[int, Path | None]:
        text = clean_for_tts(seg.translated or seg.text)
        raw = seg_dir / f"seg_{seg.idx:04d}.mp3"
        fit = seg_dir / f"seg_{seg.idx:04d}_fit.wav"
        if not text:
            return seg.idx, None
        try:
            client.synthesize(text, voice_id, raw)
            target = max(0.2, seg.end - seg.start)
            stretch_to_fit(raw, fit, target)
            return seg.idx, fit
        except Exception as exc:  # noqa: BLE001 - per-segment resilience
            logger.error(f"TTS segment {seg.idx} failed: {exc}")
            return seg.idx, None

    fitted: dict[int, Path | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, s): s.idx for s in segments}
        for fut in as_completed(futs):
            idx, path = fut.result()
            fitted[idx] = path

    import struct

    n_total = int(total * sample_rate)
    timeline = bytearray(n_total * 2)  # 16-bit mono silence
    for seg in segments:
        p = fitted.get(seg.idx)
        if p is None or not Path(p).exists():
            continue  # leave silence
        try:
            pcm, _ = _read_mono(Path(p), sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not place segment {seg.idx}: {exc}")
            continue
        start_sample = int(seg.start * sample_rate)
        # Truncate if TTS still overruns into next slot
        max_len = n_total - start_sample
        if max_len <= 0:
            continue
        chunk = pcm[:max_len * 2]
        timeline[start_sample * 2:start_sample * 2 + len(chunk)] = chunk

    with wave.open(str(output_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(timeline))
    logger.info(f"Dubbed audio written: {output_wav} ({total:.1f}s)")
    _ = settings
    return output_wav
