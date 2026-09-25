"""Free EN->VI translation via Google Translate (no API key).

Uses deep_translator (unofficial free endpoint). Quality is solid for
straight narration; keep the LLM provider for slang/idioms/nuance.
Resilient: one segment failure falls back to source text, never the batch.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.asr.srt_utils import SimpleSegment


def _hash(text: str, source: str, target: str) -> str:
    h = hashlib.sha256()
    h.update(f"google|{source}|{target}|{text}".encode("utf-8"))
    return h.hexdigest()


class GoogleFreeTranslator:
    """Stateless-ish free translator with in-memory cache + retry."""

    def __init__(
        self,
        batch_size: int = 25,
        target_lang: str = "vi",
        request_delay: float = 0.25,
    ) -> None:
        """Init translator.

        Args:
            batch_size: segments per progress step (calls are 1:1).
            target_lang: target language code.
            request_delay: seconds between calls. Free endpoint allows
                ~5 req/s, so default 0.25s stays under the limit.
        """
        self.batch_size = max(1, batch_size)
        self.target_lang = target_lang
        self.request_delay = request_delay
        self._cache: dict[str, str] = {}

    def translate_srt(
        self,
        segments: list[SimpleSegment],
        source_lang: str = "en",
        target_lang: str = "vi",
        on_progress: Any | None = None,
    ) -> list[SimpleSegment]:
        """Translate segments one-by-one with cache + fallback.

        Args:
            segments: source segments.
            source_lang: source code ('auto' ok).
            target_lang: target code.
            on_progress: optional callback(percent).

        Returns:
            Same objects with .translated filled.
        """
        self.target_lang = target_lang or self.target_lang
        src = (source_lang or "auto").strip() or "auto"
        total = len(segments)
        for start in range(0, total, self.batch_size):
            for seg in segments[start:start + self.batch_size]:
                key = _hash(seg.text, src, self.target_lang)
                if key in self._cache:
                    seg.translated = self._cache[key]
                    continue
                if not seg.text.strip():
                    seg.translated = ""
                    continue
                try:
                    out = self._translate_one(seg.text, src, self.target_lang)
                    seg.translated = out
                    self._cache[key] = out
                except Exception as exc:  # noqa: BLE001 - fallback, no crash
                    logger.warning(f"Free translate failed [{seg.idx}]: {exc}")
                    seg.translated = seg.translated or seg.text
                if self.request_delay:
                    time.sleep(self.request_delay)
            if on_progress:
                done = min(start + self.batch_size, total)
                on_progress(round(done / max(total, 1) * 100, 1))
        return segments

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _translate_one(self, text: str, source: str, target: str) -> str:
        """Single Google Translate call with retry (lazy import)."""
        from deep_translator import GoogleTranslator

        try:
            translator = GoogleTranslator(source=source, target=target)
        except Exception:
            # Unknown source code -> let Google auto-detect.
            logger.warning(f"Unsupported source '{source}', using auto")
            translator = GoogleTranslator(source="auto", target=target)
        result = translator.translate(text)
        result = (result or "").strip()
        if not result:
            raise RuntimeError("Empty translation response")
        return result
