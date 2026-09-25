"""Free Vietnamese TTS via Microsoft Edge online voices (no API key).

Voices: vi-VN-HoaiMyNeural (female, default), vi-VN-NamMinhNeural (male).
Note: unofficial endpoint - free but no SLA; keep VieNeu for paid/SLA use.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.video.ffmpeg_utils import run_ffmpeg

EDGE_VOICES = {
    "female": "vi-VN-HoaiMyNeural",
    "male": "vi-VN-NamMinhNeural",
    "vi-VN-HoaiMyNeural": "vi-VN-HoaiMyNeural",
    "vi-VN-NamMinhNeural": "vi-VN-NamMinhNeural",
}

# Edge endpoint caps a single request; split long segments by sentence.
MAX_CHARS_PER_REQUEST = 800


def split_for_edge(text: str, limit: int = MAX_CHARS_PER_REQUEST) -> list[str]:
    """Split text into sentence chunks <= limit chars.

    Args:
        text: input utterance.
        limit: max chars per chunk.

    Returns:
        Non-empty chunk list (single item when short enough).
    """
    sentences = [s.strip() for s in
                 re.split(r"(?<=[.!?…。！？])\s+", text.strip()) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        candidate = (current + " " + sent).strip()
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Hard-split pathological single sentences.
            while len(sent) > limit:
                chunks.append(sent[:limit])
                sent = sent[limit:]
            current = sent
    if current:
        chunks.append(current)
    return chunks or [text]


class EdgeTTSClient:
    """Free TTS client with the same .synthesize() shape as VieNeuClient."""

    def __init__(
        self,
        default_voice: str | None = None,
        rate: str | None = None,
    ) -> None:
        """Init client.

        Args:
            default_voice: edge voice name or female/male alias.
            rate: SSML prosody rate like '+20%'/'-10%' (env EDGE_RATE).
        """
        from src.config import get_settings

        settings = get_settings()
        self.default_voice = self.resolve_voice(
            default_voice or settings.edge_voice
        )
        self.rate = rate or settings.edge_rate

    @staticmethod
    def resolve_voice(voice_id: str | None) -> str:
        """Map alias/custom id to a real edge voice (fallback: female)."""
        if not voice_id:
            return EDGE_VOICES["female"]
        key = voice_id.strip()
        if key in EDGE_VOICES:
            return EDGE_VOICES[key]
        logger.warning(f"Unknown edge voice '{key}', using default female")
        return EDGE_VOICES["female"]

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def synthesize(
        self,
        text: str,
        voice_id: str | None,
        output_path: str | Path,
    ) -> Path:
        """Speak text to file (ffmpeg-concat when chunked).

        Args:
            text: Vietnamese text.
            voice_id: edge voice override (alias accepted).
            output_path: destination .mp3.

        Returns:
            Path to written audio file.
        """
        import edge_tts  # lazy: optional dep, mockable in tests

        text = (text or "").strip()
        if not text:
            raise ValueError("synthesize() got empty text")
        voice = self.resolve_voice(voice_id) if voice_id else self.default_voice
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        chunks = split_for_edge(text)
        logger.info(f"Edge-TTS {len(text)} chars voice={voice} "
                    f"rate={self.rate} ({len(chunks)} chunk(s))")
        ssml_chunks = [self.to_ssml(c, voice) for c in chunks]
        if len(ssml_chunks) == 1:
            asyncio.run(self._save(edge_tts, ssml_chunks[0], voice, output_path))
        else:
            parts = [output_path.with_name(f"{output_path.stem}_p{i}.mp3")
                     for i in range(len(chunks))]
            for chunk, part in zip(ssml_chunks, parts):
                asyncio.run(self._save(edge_tts, chunk, voice, part))
            self._concat(parts, output_path)
            for p in parts:
                p.unlink(missing_ok=True)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Edge-TTS produced empty file: {output_path}")
        return output_path

    def to_ssml(self, text: str, voice: str) -> str:
        """Wrap text in SSML with configured speech rate.

        Args:
            text: plain chunk text.
            voice: resolved edge voice name.

        Returns:
            SSML string (edge auto-detects SSML payloads).
        """
        import xml.sax.saxutils as saxutils

        safe = saxutils.escape(text)
        return (
            f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
            f"xml:lang='vi-VN'><voice name='{voice}'>"
            f"<prosody rate='{self.rate}'>{safe}</prosody>"
            f"</voice></speak>"
        )

    @staticmethod
    async def _save(module, text: str, voice: str, dest: Path) -> None:
        """Single edge-tts request (module injected for testability)."""
        await module.Communicate(text, voice).save(str(dest))

    @staticmethod
    def _concat(parts: list[Path], dest: Path) -> None:
        """Concat mp3 parts with ffmpeg (same codec, no re-encode)."""
        lst = dest.with_suffix(".txt")
        lst.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in parts),
            encoding="utf-8",
        )
        try:
            run_ffmpeg(["ffmpeg", "-y", "-v", "error", "-f", "concat",
                        "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(dest)])
        finally:
            lst.unlink(missing_ok=True)
