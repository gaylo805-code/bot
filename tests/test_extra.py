"""Extra coverage: whisper mock, pipeline task mock, ffmpeg utils, misc."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.asr.srt_utils import SimpleSegment


def test_whisper_transcribe_mocked(tmp_path: Path):
    from src.asr.whisper_engine import WhisperEngine

    eng = WhisperEngine(model_name="tiny", device="cpu", compute_type="int8")
    fake_info = SimpleNamespace(language="en", language_probability=0.99)
    fake_segs = [SimpleNamespace(start=0.0, end=1.0, text="Hello"),
                 SimpleNamespace(start=1.1, end=2.0, text="world")]
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter(fake_segs), fake_info)
    progress = []
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake-video")
    with patch.object(WhisperEngine, "_load_model",
                      return_value=fake_model), \
        patch("src.asr.whisper_engine.extract_audio",
              return_value=tmp_path / "x.wav"):
        out = eng.transcribe(video,
                             on_progress=progress.append)
    # merge_gap=0.0 keeps Whisper's natural boundaries; merging every short
    # gap would batch minutes of speech into one TTS request.
    assert len(out) == 2
    assert out[0].text == "Hello" and out[1].text == "world"
    assert progress and progress[-1] == 100.0


def test_whisper_missing_file():
    from src.asr.whisper_engine import WhisperEngine
    with pytest.raises(FileNotFoundError):
        WhisperEngine().transcribe("/no/such/file.mp4")


def test_run_ffmpeg_ok_and_fail():
    from src.video import ffmpeg_utils
    out, _ = ffmpeg_utils.run_ffmpeg(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=0.2", "-f", "null", "-"])
    assert isinstance(out, str)
    with pytest.raises(ffmpeg_utils.FFmpegError):
        ffmpeg_utils.run_ffmpeg(["ffmpeg", "-v", "error",
                                 "-i", "/no/such/file.mp4",
                                 "-f", "null", "-"])


def test_vieneu_clone_and_empty(tmp_path: Path):
    from src.tts.vieneu_client import VieNeuClient
    c = VieNeuClient(api_key="k")
    sample = tmp_path / "s.wav"
    sample.write_bytes(b"RIFF....")
    assert c.clone_voice(sample) == c.default_voice
    with pytest.raises(FileNotFoundError):
        c.clone_voice(tmp_path / "missing.wav")
    with pytest.raises(ValueError):
        c.synthesize("   ", "v", tmp_path / "o.mp3")


def test_translator_payload_and_dict_parse():
    from src.translate.llm_translator import LLMTranslator
    tr = LLMTranslator(model="gpt-4o-mini", batch_size=5)
    segs = [SimpleSegment(idx=i, start=float(i), end=float(i + 1),
                          text=f"t{i}") for i in range(5)]
    payload = tr._batch_payload(segs)
    assert payload[0]["prev"] == [] and len(payload[2]["prev"]) == 2
    assert payload[0]["next"] == ["t1", "t2", "t3"]
    parsed = LLMTranslator._parse_response(
        '{"translations": [{"index": 0, "translated": "Chào"}]}')
    assert parsed[0]["translated"] == "Chào"
    assert LLMTranslator._parse_response("not json at all") == []


def test_stretch_within_tolerance_copies(tmp_path: Path):
    import wave
    from src.tts.audio_builder import get_wav_duration, stretch_to_fit
    p = tmp_path / "a.wav"
    sr = 44100
    n = sr  # 1.0s
    import struct
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))
    assert abs(get_wav_duration(p) - 1.0) < 0.01
    dst = tmp_path / "b.wav"
    out = stretch_to_fit(p, dst, target_dur=1.05, tolerance=0.15)
    assert Path(out).exists()


def test_celery_app_config():
    from src.worker.celery_app import make_celery
    app = make_celery()
    assert app.conf.task_track_started is True
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.task_acks_late is True


def test_pipeline_task_mocked(monkeypatch, tmp_path: Path):
    """Cover worker/tasks.py pipeline with all heavy deps mocked."""
    import src.worker.tasks as tasks_mod
    from src.asr.srt_utils import SimpleSegment as SS

    job_id, video = "job123", str(tmp_path / "in.mp4")
    Path(video).write_bytes(b"fake")

    fake_segs = [SS(idx=0, start=0.0, end=1.0, text="Hello",
                    language="en")]

    monkeypatch.setattr(tasks_mod, "_update", lambda *a, **k: None)
    monkeypatch.setattr(tasks_mod, "_save_segments", lambda *a, **k: None)
    monkeypatch.setattr(tasks_mod, "segments_to_srt", lambda *a, **k: None)
    monkeypatch.setattr(tasks_mod.WhisperEngine, "transcribe",
                        lambda self, *a, **k: fake_segs)

    class _FakeTranslator:
        def translate_srt(self, s, **k):
            for seg in s:
                seg.translated = f"VI:{seg.text}"
            return s

    monkeypatch.setattr(tasks_mod, "get_translator",
                        lambda *a, **k: _FakeTranslator())
    monkeypatch.setattr(tasks_mod, "build_dubbed_audio",
                        lambda *a, **k: tmp_path / "d.wav")
    monkeypatch.setattr(tasks_mod, "merge_dubbed_video",
                        lambda *a, **k: tmp_path / "out.mp4")

    # Fake DB job for target/voice lookup
    class FakeQ:
        def get(self, *a):
            return SimpleNamespace(target_lang="vi",
                                   voice_id="v1", keep_background=0)
    class FakeDB:
        def __enter__(self): return FakeQ()
        def __exit__(self, *a): return False
    monkeypatch.setattr(tasks_mod, "get_session_factory",
                        lambda *a, **k: FakeDB)

    # Task.run is already bound to the task instance (bind=True), so call
    # with (job_id, video). Eager backend calls are avoided: update_state
    # inside _asr_cb is guarded, and all heavy steps are mocked above.
    res = tasks_mod.process_dubbing_job.run(job_id, video)
    assert isinstance(res, str)


def test_build_format_strategies_height_cap():
    """`quality` must cap the downloaded height, not just append a no-op filter."""
    from src.api.routes_upload import _build_format_strategies

    strategies = _build_format_strategies("720")
    fmt = strategies[0][strategies[0].index("-f") + 1]
    assert "[height<=720]" in fmt, fmt
    assert "[height<=1080]" not in fmt
    # The unfiltered fallback must not sneak an uncapped best back in.
    assert "bestvideo[ext=mp4]" not in fmt


def test_build_format_strategies_best_uncapped():
    from src.api.routes_upload import _build_format_strategies

    for value in ("best", "", "nonsense", None):
        strategies = _build_format_strategies(value)  # type: ignore[arg-type]
        fmt = strategies[0][strategies[0].index("-f") + 1]
        assert "height<=" not in fmt, (value, fmt)
        assert "bestvideo[ext=mp4]" in fmt, (value, fmt)


def test_capped_quality_keeps_an_uncapped_fallback():
    """A low quality request must never be able to hard-fail the job.

    Facebook often serves one progressive stream. A strictly-capped
    selector matches nothing and yt-dlp exits "Requested format is not
    available", which surfaced as a 400 for a perfectly downloadable
    video.
    """
    from src.api.routes_upload import _build_format_strategies

    strategies = _build_format_strategies("360")
    assert len(strategies) == 4, "capped path must keep the uncapped safety net"

    def fmt(entry):
        return entry[entry.index("-f") + 1]

    assert "[height<=360]" in fmt(strategies[0])
    assert "[height<=360]" in fmt(strategies[1])
    # The tail must be able to select a format above the cap.
    for entry in strategies[2:]:
        assert "height<=" not in fmt(entry), fmt(entry)
    assert fmt(strategies[-1]) == "best"


def test_best_quality_still_uses_two_strategies():
    from src.api.routes_upload import _build_format_strategies

    strategies = _build_format_strategies("best")
    assert len(strategies) == 2
    assert strategies[-1][strategies[-1].index("-f") + 1] == "best"
