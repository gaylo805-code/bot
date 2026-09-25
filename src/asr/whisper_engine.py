"""Whisper ASR engine based on faster-whisper with VAD + GPU fallback."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from src.asr.srt_utils import (
    SimpleSegment,
    clean_srt,
    collapse_repeated_text,
    extract_audio,
    suppress_repeated_segments,
)
from src.config import get_settings


class WhisperEngine:
    """Transcribe video audio to timestamped segments.

    Uses faster-whisper (large-v3 default), auto language detection,
    VAD filter, beam_size=5, temperature=0. Falls back to CPU when
    CUDA is unavailable or model load fails.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        beam_size: int | None = None,
    ) -> None:
        """Init engine config (model is lazy-loaded on first transcribe)."""
        settings = get_settings()
        self.model_name = model_name or settings.whisper_model
        self.device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type
        self.beam_size = beam_size or settings.whisper_beam_size
        self._model: Any | None = None

    def _load_model(self) -> Any:
        """Lazy-load faster-whisper model with GPU->CPU fallback."""
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run `uv sync` first."
            ) from exc
        for device, compute in (
            (self.device, self.compute_type),
            ("cpu", "int8"),
        ):
            try:
                logger.info(
                    f"Loading Whisper model {self.model_name} "
                    f"on {device}/{compute}"
                )
                self._model = WhisperModel(
                    self.model_name, device=device, compute_type=compute
                )
                self.device = device
                return self._model
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Whisper load on {device} failed: {exc}")
                self._model = None
        raise RuntimeError("Could not load faster-whisper model on cuda nor cpu")

    def transcribe(
        self,
        video_path: str | Path,
        on_progress: Callable[[float], None] | None = None,
        language: str | None = None,
        tmp_wav: str | Path | None = None,
    ) -> list[SimpleSegment]:
        """Transcribe a video file to cleaned segments.

        Args:
            video_path: path to source video.
            on_progress: optional callback receiving percent 0-100.
            language: force language code or None for auto-detect.
            tmp_wav: optional wav path override.

        Returns:
            List of cleaned SimpleSegment.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        settings = get_settings()
        wav = Path(tmp_wav) if tmp_wav else (
            settings.tmp_dir / f"{video_path.stem}_16k.wav"
        )
        extract_audio(video_path, wav)
        model = self._load_model()
        # faster-whisper yields (segments_generator, info)
        seg_iter, info = model.transcribe(
            str(wav),
            language=language,
            beam_size=self.beam_size,
            temperature=0.0,
            # Do not let a previous hallucinated phrase become context for
            # the next decoding window; this is a common source of loops.
            condition_on_previous_text=False,
            max_initial_timestamp=0.0,
            hallucination_silence_threshold=2.0,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        detected = getattr(info, "language", "") or ""
        logger.info(
            f"Detected language: {detected} "
            f"(prob={getattr(info, 'language_probability', 0):.2f})"
        )
        raw: list[SimpleSegment] = []
        # Stream the generator so on_progress fires while transcribing.
        # Buffering it into a list first (to count segments) left the job
        # pinned at 1% for the whole ASR stage - on a 10-minute video that
        # is a silent 15+ minutes. Segment end times are in the original
        # audio timeline, so they scale against info.duration.
        total_dur = float(getattr(info, "duration", 0.0) or 0.0)
        for i, s in enumerate(seg_iter):
            raw.append(
                SimpleSegment(
                    idx=i, start=float(s.start), end=float(s.end),
                    text=str(s.text).strip(), language=detected,
                )
            )
            if on_progress:
                if total_dur > 0:
                    on_progress(round(min(100.0, float(s.end) / total_dur * 100.0), 1))
                else:
                    on_progress(100.0)
        # Suppress pathological decoder loops before translation/TTS. Empty
        # timed segments are retained so the rest of the timeline stays aligned.
        for segment in raw:
            segment.text = collapse_repeated_text(segment.text)
        suppress_repeated_segments(raw)
        # Keep Whisper's natural timing boundaries. Merging every short gap
        # into one multi-minute TTS request makes the voice sound stretched
        # and lets one bad phrase poison the entire video.
        cleaned = clean_srt(raw, merge_gap=0.0)
        if on_progress:
            on_progress(100.0)
        try:
            wav.unlink(missing_ok=True)
        except OSError:
            pass
        return cleaned
