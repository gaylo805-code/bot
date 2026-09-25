"""Tests: free Edge-TTS client (mocked network) + provider factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tts.base import TTSClient
from src.tts.edge_client import EdgeTTSClient, split_for_edge
from src.tts.factory import describe_tts, get_tts_client


def test_split_for_edge_short():
    assert split_for_edge("Xin chào. Tạm biệt.") == ["Xin chào. Tạm biệt."]


def test_split_for_edge_long():
    long_text = " ".join(f"Câu số {i}." for i in range(200))
    chunks = split_for_edge(long_text, limit=100)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)  # sentence granularity
    assert " ".join(chunks).replace("  ", " ") == long_text


def test_resolve_voice_aliases():
    assert EdgeTTSClient.resolve_voice("female") == "vi-VN-HoaiMyNeural"
    assert EdgeTTSClient.resolve_voice("male") == "vi-VN-NamMinhNeural"
    assert EdgeTTSClient.resolve_voice("vi-VN-NamMinhNeural") == "vi-VN-NamMinhNeural"
    assert EdgeTTSClient.resolve_voice("weird") == "vi-VN-HoaiMyNeural"
    assert EdgeTTSClient.resolve_voice("") == "vi-VN-HoaiMyNeural"
    assert EdgeTTSClient.resolve_voice(None) == "vi-VN-HoaiMyNeural"


class _FakeCommunicate:
    seen: list[tuple[str, str, str]] = []

    def __init__(self, text: str, voice: str, *, rate: str = "+0%", **_kwargs):
        _FakeCommunicate.seen.append((text, voice, rate))

    async def save(self, dest: str) -> None:
        Path(dest).write_bytes(b"FAKEEDGE" * 100)


@pytest.fixture()
def _fake_edge(monkeypatch):
    import edge_tts

    _FakeCommunicate.seen = []
    monkeypatch.setattr(edge_tts, "Communicate", _FakeCommunicate)
    yield


def test_edge_synthesize_single_chunk(tmp_path: Path, _fake_edge):
    client = EdgeTTSClient(default_voice="male")
    out = tmp_path / "a.mp3"
    res = client.synthesize("Xin chào", voice_id="", output_path=out)
    assert res.exists() and res.stat().st_size > 0
    text_sent, voice_sent, rate_sent = _FakeCommunicate.seen[0]
    assert text_sent == "Xin chào"
    assert "<speak" not in text_sent and "prosody" not in text_sent
    assert voice_sent == "vi-VN-NamMinhNeural"
    assert rate_sent == "+0%"


def test_edge_synthesize_voice_override(tmp_path: Path, _fake_edge):
    client = EdgeTTSClient()
    out = tmp_path / "b.mp3"
    client.synthesize("Chào", voice_id="male", output_path=out)
    text_sent, voice_sent, _rate_sent = _FakeCommunicate.seen[0]
    assert text_sent == "Chào"
    assert voice_sent == "vi-VN-NamMinhNeural"


def test_edge_synthesize_empty_raises(tmp_path: Path, _fake_edge):
    with pytest.raises(ValueError):
        EdgeTTSClient().synthesize("   ", None, tmp_path / "x.mp3")


def test_edge_synthesize_multi_chunk_concats(tmp_path: Path, monkeypatch):
    """Long text -> multiple chunks -> ffmpeg-concatenated mp3."""
    import subprocess

    import edge_tts

    sample = tmp_path / "sample.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=0.3",
         "-c:a", "libmp3lame", str(sample)], check=True)
    blob = sample.read_bytes()

    class _RealMp3:
        def __init__(self, text: str, voice: str, *, rate: str = "+0%", **_kwargs):
            pass

        async def save(self, dest: str) -> None:
            Path(dest).write_bytes(blob)

    monkeypatch.setattr(edge_tts, "Communicate", _RealMp3)
    long_text = " ".join(f"Câu thứ {i} rất hay." for i in range(60))
    assert len(long_text) > 800
    out = tmp_path / "long.mp3"
    res = EdgeTTSClient().synthesize(long_text, None, out)
    assert res.exists() and res.stat().st_size > len(blob)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(res)],
        capture_output=True, text=True)
    assert float(probe.stdout.strip()) > 0.5


def test_factory_defaults_to_edge(monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "edge")
    from src.config import get_settings
    get_settings.cache_clear()
    client = get_tts_client()
    assert isinstance(client, EdgeTTSClient)
    assert isinstance(client, TTSClient)
    get_settings.cache_clear()


def test_factory_vieneu(monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "vieneu")
    monkeypatch.setenv("VIENEU_API_KEY", "k")
    from src.config import get_settings
    get_settings.cache_clear()
    from src.tts.vieneu_client import VieNeuClient
    client = get_tts_client()
    assert isinstance(client, VieNeuClient)
    get_settings.cache_clear()


def test_describe_tts(monkeypatch):
    from src.config import get_settings
    monkeypatch.setenv("TTS_PROVIDER", "edge")
    get_settings.cache_clear()
    assert "Edge-TTS" in describe_tts() and "HoaiMy" in describe_tts()
    monkeypatch.setenv("TTS_PROVIDER", "vieneu")
    get_settings.cache_clear()
    assert "VieNeu" in describe_tts()
    get_settings.cache_clear()


def test_clean_for_tts_removes_urls():
    from src.tts.text_cleaner import clean_for_tts
    assert clean_for_tts("www.example.com Xin chào") == "Xin chào"
    assert clean_for_tts("Xem tại https://example.com/path nhé") == "Xem tại nhé"
    assert clean_for_tts("[Xem](https://example.com) nội dung") == "Xem nội dung"
    assert clean_for_tts("www.com") == ""
