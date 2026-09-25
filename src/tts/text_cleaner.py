"""Normalize transcript text before TTS so URLs/markup are not spoken."""

from __future__ import annotations

import re

# URLs are metadata in a video transcript, not narration. Remove them before TTS.
_URL_RE = re.compile(
    r"(?i)(?:https?://|www\.)[^\s<>]+"
    r"|(?<![\w@])(?:[a-z0-9-]+\.)+(?:com|net|org|io|tv|co|me|ly|vn)(?:/[^\s<>]*)?"
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def clean_for_tts(text: str) -> str:
    """Remove URLs and leftover URL punctuation while preserving narration."""
    text = str(text or "").strip()
    if not text:
        return ""
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    # URL removal can leave isolated punctuation or repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n-–—,;:|/")
    return text.strip()
