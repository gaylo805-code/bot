"""Tests: mocked VieNeu API, atempo when TTS longer than slot."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.asr.srt_utils import SimpleSegment
from src.tts.audio_builder import build_dubbed_audio, stretch_to_fit
from src.tts.vieneu_client import VieNeuClient


def _tone_wav(path: Path, seconds: float = 1.0, sr: int = 44100) -> Path:
    import math
    import struct
    n = int(seconds * sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = [int(10000 * math.sin(2 * math.pi * 440 * i / sr))
                  for i in range(n)]
        w.writeframes(struct.pack("<" + "h" * n, *frames))
    return path


def test_synthesize_mocked(tmp_path: Path):
    client = VieNeuClient(api_key="test", base_url="http://x/v1")
    fake = MagicMock()
    fake.write_to_file = MagicMock(
        side_effect=lambda p: Path(p).write_bytes(b"FAKEAUDIO"))
    fake.content = None
    mock_client = MagicMock()
    mock_client.audio.speech.create.return_value = fake
    client._client = mock_client
    # bypass retry wrapper timing: call underlying once
    out = tmp_path / "a.mp3"
    # synthesize is @retry-wrapped; mock inner client only
    res = client.synthesize("Xin chào", "voice-1", out)
    assert res.exists()
    mock_client.audio.speech.create.assert_called_once()


def test_stretch_to_fit_speeds_up(tmp_path: Path):
    src = _tone_wav(tmp_path / "long.wav", seconds=2.0)
    dst = tmp_path / "fit.wav"
    out = stretch_to_fit(src, dst, target_dur=1.0, tolerance=0.15)
    assert out.exists()
    with wave.open(str(out), "rb") as w:
        dur = w.getnframes() / w.getframerate()
    assert abs(dur - 1.0) < 0.3  # atempo adjusted ~2x


def test_build_dubbed_audio_timeline(tmp_path: Path):
    segs = [SimpleSegment(idx=0, start=0.0, end=1.0, text="Hi",
                          translated="Chào"),
            SimpleSegment(idx=1, start=2.0, end=3.0, text="Bye",
                          translated="Tạm biệt")]
    client = VieNeuClient(api_key="t", base_url="http://x")

    def _fake_synth(text, voice_id, output_path):
        output_path = Path(output_path)
        _tone_wav(output_path.with_suffix(".wav"), seconds=0.8)
        # synthesize writes mp3 path; write wav then copy bytes to mp3 path
        wav = output_path.with_suffix(".wav")
        output_path.write_bytes(wav.read_bytes())
        return output_path

    with patch.object(VieNeuClient, "synthesize",
                      side_effect=_fake_synth):
        out = build_dubbed_audio(segs, "v1", tmp_path / "dub.wav",
                                 client=client, max_workers=2)
    assert out.exists()
    with wave.open(str(out), "rb") as w:
        assert w.getframerate() == 44100
        assert w.getnchannels() == 1
        dur = w.getnframes() / w.getframerate()
    assert dur >= 2.9
