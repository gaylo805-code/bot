"""TTS provider factory: TTS_PROVIDER=edge (free) | vieneu (paid)."""

from __future__ import annotations

from src.tts.base import TTSClient


def get_tts_client(
    provider: str | None = None,
    default_voice: str | None = None,
) -> TTSClient:
    """Build the configured TTS client.

    Args:
        provider: 'edge' | 'vieneu' (env TTS_PROVIDER when None).
        default_voice: voice override for this job.

    Returns:
        Client implementing .synthesize(text, voice_id, output_path).
    """
    from src.config import get_settings

    settings = get_settings()
    name = (provider or settings.tts_provider).strip().lower()
    if name == "vieneu":
        from src.tts.vieneu_client import VieNeuClient

        return VieNeuClient(default_voice=default_voice or None)  # type: ignore[return-value]
    from src.tts.edge_client import EdgeTTSClient

    return EdgeTTSClient(default_voice=default_voice)  # type: ignore[return-value]


def describe_tts() -> str:
    """One-liner for /voices: provider + default voice (no secrets)."""
    from src.config import get_settings

    settings = get_settings()
    if settings.tts_provider.strip().lower() == "vieneu":
        return f"VieNeu (trả phí) — voice `{settings.vieneu_voice_id}`"
    from src.tts.edge_client import EdgeTTSClient

    return (f"Edge-TTS (free) — voice "
            f"`{EdgeTTSClient.resolve_voice(settings.edge_voice)}`")
