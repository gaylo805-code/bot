"""VieNeu TTS client over an OpenAI-compatible endpoint."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings


class VieNeuClient:
    """Thin wrapper around OpenAI SDK pointed at VieNeu base_url.

    Calls client.audio.speech.create(model="vieneu-v4", voice=...).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        default_voice: str | None = None,
    ) -> None:
        """Init client config (OpenAI client created lazily)."""
        settings = get_settings()
        self.api_key = api_key or settings.vieneu_api_key
        self.base_url = (base_url or settings.vieneu_base_url).rstrip("/")
        self.model = model or settings.vieneu_model
        self.default_voice = default_voice or settings.vieneu_voice_id
        self._client = None  # lazy OpenAI client

    def _get_client(self):
        """Create OpenAI client lazily (import inside for testability)."""
        if self._client is not None:
            return self._client
        from openai import OpenAI

        if not self.api_key:
            raise RuntimeError("VIENEU_API_KEY is not set")
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def synthesize(
        self,
        text: str,
        voice_id: str | None,
        output_path: str | Path,
    ) -> Path:
        """Synthesize text to an audio file.

        Args:
            text: Vietnamese text to speak.
            voice_id: voice id override (default voice if None).
            output_path: destination file (.mp3/.wav).

        Returns:
            Path to written audio file.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("synthesize() got empty text")
        voice = voice_id or self.default_voice
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        client = self._get_client()
        logger.info(f"TTS synthesize {len(text)} chars voice={voice}")
        resp = client.audio.speech.create(
            model=self.model, voice=voice, input=text
        )
        # openai>=1.x response has .write_to_file or .content/stream
        if hasattr(resp, "write_to_file"):
            resp.write_to_file(str(output_path))
        elif hasattr(resp, "content") and resp.content:
            output_path.write_bytes(resp.content)
        else:  # streaming fallback
            with open(output_path, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"TTS produced empty file: {output_path}")
        return output_path

    def clone_voice(self, audio_sample_path: str | Path) -> str:
        """Voice-clone stub: VieNeu v4 manages voices server-side.

        Args:
            audio_sample_path: reference audio file.

        Returns:
            Currently configured voice id (clone API not public).
        """
        # VieNeu public API does not expose cloning; keep the hook so the
        # pipeline/UI can accept a sample file without breaking.
        sample = Path(audio_sample_path)
        if not sample.exists():
            raise FileNotFoundError(f"Voice sample not found: {sample}")
        logger.warning("clone_voice() not supported by VieNeu API; "
                       "returning default voice")
        return self.default_voice
