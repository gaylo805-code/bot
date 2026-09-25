"""Translator factory: TRANSLATE_PROVIDER=google (free) | llm (paid)."""

from __future__ import annotations

from src.translate.base import Translator


def get_translator(
    provider: str | None = None,
    model: str | None = None,
    batch_size: int = 25,
    target_lang: str = "vi",
) -> Translator:
    """Build the configured translator.

    Args:
        provider: 'google' | 'llm' (env TRANSLATE_PROVIDER when None).
        model: LiteLLM model string (llm provider only).
        batch_size: segments per batch/progress step.
        target_lang: target language code.

    Returns:
        Object with .translate_srt(segments, source_lang, target_lang).
    """
    from src.config import get_settings

    settings = get_settings()
    name = (provider or settings.translate_provider).strip().lower()
    if name == "llm":
        from src.translate.llm_translator import LLMTranslator

        return LLMTranslator(  # type: ignore[return-value]
            model=model, batch_size=batch_size, target_lang=target_lang
        )
    from src.translate.google_translator import GoogleFreeTranslator

    return GoogleFreeTranslator(  # type: ignore[return-value]
        batch_size=batch_size, target_lang=target_lang
    )


def describe_translator() -> str:
    """One-liner for docs/status (no secrets)."""
    from src.config import get_settings

    settings = get_settings()
    if settings.translate_provider.strip().lower() == "llm":
        return f"LLM ({settings.llm_model})"
    return "Google Translate (free)"
