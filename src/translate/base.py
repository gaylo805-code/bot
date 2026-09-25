"""Translator abstraction: free Google (no key) default, LLM for quality."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.asr.srt_utils import SimpleSegment


@runtime_checkable
class Translator(Protocol):
    """Common interface every translation provider must implement."""

    def translate_srt(
        self,
        segments: list[SimpleSegment],
        source_lang: str = "en",
        target_lang: str = "vi",
        on_progress: Any | None = None,
    ) -> list[SimpleSegment]:
        """Fill .translated on each segment, preserving timestamps."""
        ...
