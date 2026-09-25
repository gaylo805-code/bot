"""Whisper ASR engine based on faster-whisper with VAD + GPU fallback."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from src.asr.srt_utils import SimpleSegment, clean_srt, extract_audio
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
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        detected = getattr(info, "language", "") or ""
        logger.info(
            f"Detected language: {detected} "
            f"(prob={getattr(info, 'language_probability', 0):.2f})"
        )
        raw: list[SimpleSegment] = []
        # Materialize generator to count progress; segments carry start/end.
        collected = list(seg_iter)
        total = len(collected) if collected else 1
        for i, s in enumerate(collected):
            raw.append(
                SimpleSegment(
                    idx=i, start=float(s.start), end=float(s.end),
                    text=str(s.text).strip(), language=detected,
                )
            )
            if on_progress:
                on_progress(round((i + 1) / total * 100, 1))
        cleaned = clean_srt(raw)
        if on_progress:
            on_progress(100.0)
        try:
            wav.unlink(missing_ok=True)
        except OSError:
            pass
        return cleaned
