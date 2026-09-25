"""TTS provider abstraction: free Edge-TTS default, VieNeu for paid quality."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class TTSClient(Protocol):
    """Common interface every TTS provider must implement."""

    def synthesize(
        self,
        text: str,
        voice_id: str | None,
        output_path: str | Path,
    ) -> Path:
        """Speak text to an audio file. Returns the file path."""
        ...
