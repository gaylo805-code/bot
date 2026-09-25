"""LLM translation via LiteLLM with batching, context window, cache + retry."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.asr.srt_utils import SimpleSegment
from src.config import get_settings
from src.translate.prompts import TRANSLATE_SYSTEM, build_translate_prompt


def _hash(text: str, source: str, target: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(f"{model}|{source}|{target}|{text}".encode("utf-8"))
    return h.hexdigest()


class LLMTranslator:
    """Translate subtitle segments preserving timestamps.

    - Batches 20-30 segments keeping context (3 prev + 3 next sentences).
    - Returns JSON [{index, translated}]; only text changes.
    - Retries 3x with exponential backoff; in-memory cache by source hash.
    - One segment failure never crashes the whole batch (fallback = source).
    """

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 25,
        target_lang: str = "vi",
    ) -> None:
        """Init translator.

        Args:
            model: LiteLLM model string (env LLM_MODEL if None).
            batch_size: segments per LLM call (20-30 recommended).
            target_lang: target language code.
        """
        settings = get_settings()
        self.model = model or settings.llm_model
        self.batch_size = max(1, min(batch_size, 30))
        self.target_lang = target_lang
        self._cache: dict[str, str] = {}

    # -- public API -----------------------------------------------------
    def translate_srt(
        self,
        segments: list[SimpleSegment],
        source_lang: str = "en",
        target_lang: str = "vi",
        on_progress: Any | None = None,
    ) -> list[SimpleSegment]:
        """Translate all segments in batches.

        Args:
            segments: source segments.
            source_lang: source language code/name.
            target_lang: target language code.
            on_progress: optional callback(percent).

        Returns:
            Same segment objects with .translated filled.
        """
        self.target_lang = target_lang or self.target_lang
        total = len(segments)
        if total == 0:
            return segments
        out = list(segments)
        for start in range(0, total, self.batch_size):
            batch = out[start:start + self.batch_size]
            try:
                translated_map = self._translate_batch(batch, source_lang)
            except Exception as exc:  # noqa: BLE001 - per-segment fallback
                logger.error(f"Batch translate failed, keeping source: {exc}")
                translated_map = {}
            for seg in batch:
                key = _hash(seg.text, source_lang, self.target_lang, self.model)
                if seg.idx in translated_map:
                    seg.translated = translated_map[seg.idx]
                    self._cache[key] = seg.translated
                elif key in self._cache:
                    seg.translated = self._cache[key]
                elif not seg.translated:
                    seg.translated = seg.text  # fallback: keep source
            if on_progress:
                done = min(start + self.batch_size, total)
                on_progress(round(done / total * 100, 1))
        return out

    # -- internals ------------------------------------------------------
    def _batch_payload(
        self, batch: list[SimpleSegment], full: list[SimpleSegment] | None = None
    ) -> list[dict[str, Any]]:
        """Build payload with 3-prev/3-next context window per item."""
        full = full or batch
        pos = {id(s): i for i, s in enumerate(full)}
        payload: list[dict[str, Any]] = []
        for seg in batch:
            i = pos.get(id(seg), 0)
            prev = [s.text for s in full[max(0, i - 3):i]]
            nxt = [s.text for s in full[i + 1:i + 4]]
            payload.append(
                {"index": seg.idx, "text": seg.text,
                 "prev": prev, "next": nxt}
            )
        return payload

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def _call_llm(self, prompt: str) -> str:
        """Call LiteLLM completion with retry. Never logs secrets."""
        import litellm  # lazy import so tests can mock easily

        settings = get_settings()
        # Wire provider keys from env without logging values.
        extra: dict[str, Any] = {}
        if settings.gemini_api_key and "gemini" in self.model:
            extra["api_key"] = settings.gemini_api_key
        elif settings.openai_api_key and "gpt" in self.model:
            extra["api_key"] = settings.openai_api_key
        elif settings.deepseek_api_key and "deepseek" in self.model:
            extra["api_key"] = settings.deepseek_api_key
        if self.model.startswith("ollama/") and settings.ollama_base_url:
            extra["api_base"] = settings.ollama_base_url
        resp = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": TRANSLATE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            **extra,
        )
        content = resp.choices[0].message.content or "[]"
        return content.strip()

    def _translate_batch(
        self, batch: list[SimpleSegment], source_lang: str
    ) -> dict[int, str]:
        """Translate one batch, using cache to skip repeated texts."""
        # Serve fully-cached batches without an API call.
        uncached = [
            s for s in batch
            if _hash(s.text, source_lang, self.target_lang, self.model)
            not in self._cache
        ]
        if not uncached:
            return {s.idx: self._cache[
                _hash(s.text, source_lang, self.target_lang, self.model)]
                for s in batch}
        payload = self._batch_payload(batch)
        prompt = build_translate_prompt(
            source_lang, json.dumps(payload, ensure_ascii=False)
        )
        raw = self._call_llm(prompt)
        parsed = self._parse_response(raw)
        result: dict[int, str] = {}
        for item in parsed:
            try:
                idx = int(item.get("index"))
                txt = str(item.get("translated", "")).strip()
                if txt:
                    result[idx] = txt
            except (ValueError, TypeError, AttributeError):
                continue
        # Cache what we got
        by_idx = {s.idx: s for s in batch}
        for idx, txt in result.items():
            seg = by_idx.get(idx)
            if seg is not None:
                self._cache[_hash(
                    seg.text, source_lang, self.target_lang, self.model)] = txt
        return result

    @staticmethod
    def _parse_response(raw: str) -> list[dict[str, Any]]:
        """Parse JSON array from LLM output, tolerating code fences."""
        text = raw.strip()
        if text.startswith("```"):
            # strip ```json ... ``` fences
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract the first [...] block
            start, end = text.find("["), text.rfind("]")
            if start >= 0 and end > start:
                data = json.loads(text[start:end + 1])
            else:
                logger.warning(f"Unparseable translate response: {raw[:200]}")
                return []
        if isinstance(data, dict):
            data = data.get("translations", data.get("results", []))
        return data if isinstance(data, list) else []
