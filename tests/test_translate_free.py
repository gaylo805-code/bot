"""Tests: free Google translator (mocked network) + translator factory."""

from __future__ import annotations

import pytest

from src.asr.srt_utils import SimpleSegment
from src.translate.base import Translator
from src.translate.factory import describe_translator, get_translator
from src.translate.google_translator import GoogleFreeTranslator


def _segs() -> list[SimpleSegment]:
    return [SimpleSegment(idx=0, start=0.0, end=1.0, text="Hello world"),
            SimpleSegment(idx=1, start=1.0, end=2.0, text="Good morning")]


class _FakeGT:
    calls = 0

    def __init__(self, source="auto", target="vi"):
        self.source, self.target = source, target

    def translate(self, text: str) -> str:
        _FakeGT.calls += 1
        return f"VI:{text}"


@pytest.fixture()
def _fake_google(monkeypatch):
    import deep_translator

    _FakeGT.calls = 0
    monkeypatch.setattr(deep_translator, "GoogleTranslator", _FakeGT)
    yield


def test_google_translate_and_cache(_fake_google):
    tr = GoogleFreeTranslator(request_delay=0)
    segs = _segs()
    out = tr.translate_srt(segs, source_lang="en", target_lang="vi")
    assert out[0].translated == "VI:Hello world"
    assert _FakeGT.calls == 2
    # Second run fully cached -> zero network calls
    tr.translate_srt(segs, source_lang="en", target_lang="vi")
    assert _FakeGT.calls == 2


def test_google_failure_falls_back_to_source(monkeypatch):
    import deep_translator

    class _Boom:
        def __init__(self, *a, **k):
            pass

        def translate(self, text: str) -> str:
            raise RuntimeError("429 rate limited")

    monkeypatch.setattr(deep_translator, "GoogleTranslator", _Boom)
    tr = GoogleFreeTranslator(request_delay=0)
    segs = _segs()
    out = tr.translate_srt(segs)  # must not raise
    assert all(s.translated == s.text for s in out)


def test_google_empty_segment_skipped(_fake_google):
    tr = GoogleFreeTranslator(request_delay=0)
    segs = [SimpleSegment(idx=0, start=0.0, end=0.5, text="   ")]
    out = tr.translate_srt(segs)
    assert out[0].translated == ""
    assert _FakeGT.calls == 0


def test_google_progress_callback(_fake_google):
    tr = GoogleFreeTranslator(batch_size=1, request_delay=0)
    seen: list[float] = []
    tr.translate_srt(_segs(), on_progress=seen.append)
    assert seen == [50.0, 100.0]


def test_factory_defaults_to_google(monkeypatch):
    monkeypatch.setenv("TRANSLATE_PROVIDER", "google")
    from src.config import get_settings
    get_settings.cache_clear()
    client = get_translator()
    assert isinstance(client, GoogleFreeTranslator)
    assert isinstance(client, Translator)
    get_settings.cache_clear()


def test_factory_llm(monkeypatch):
    monkeypatch.setenv("TRANSLATE_PROVIDER", "llm")
    from src.config import get_settings
    get_settings.cache_clear()
    from src.translate.llm_translator import LLMTranslator
    client = get_translator()
    assert isinstance(client, LLMTranslator)
    get_settings.cache_clear()


def test_describe_translator(monkeypatch):
    from src.config import get_settings
    monkeypatch.setenv("TRANSLATE_PROVIDER", "google")
    get_settings.cache_clear()
    assert "Google" in describe_translator()
    monkeypatch.setenv("TRANSLATE_PROVIDER", "llm")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    get_settings.cache_clear()
    assert "gpt-4o-mini" in describe_translator()
    get_settings.cache_clear()
