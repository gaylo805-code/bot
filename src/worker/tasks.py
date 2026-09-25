"""Celery pipeline: ASR (0-30%) -> Translate (30-50%) -> TTS (50-85%) -> Merge (85-100%)."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from celery.exceptions import Retry
from loguru import logger

from src.asr.srt_utils import segments_to_srt
from src.asr.whisper_engine import WhisperEngine
from src.config import get_settings
from src.models.database import get_session_factory, init_db
from src.models.job import Job, Segment
from src.translate.factory import get_translator
from src.tts.audio_builder import build_dubbed_audio
from src.tts.factory import get_tts_client
from src.video.merger import merge_dubbed_video
from src.worker.celery_app import celery_app


def _update(job_id: str, **fields) -> None:
    """Persist progress fields for a job (own short-lived session)."""
    init_db()
    factory = get_session_factory()
    with factory() as db:
        job = db.get(Job, job_id)
        if not job:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        db.commit()


def _save_segments(job_id: str, segs, translated: bool = False) -> None:
    init_db()
    factory = get_session_factory()
    with factory() as db:
        db.query(Segment).filter(Segment.job_id == job_id).delete()
        for s in segs:
            db.add(Segment(
                job_id=job_id, idx=s.idx, start=s.start, end=s.end,
                source_text=s.text,
                translated_text=s.translated if translated else "",
            ))
        db.commit()


@celery_app.task(bind=True, max_retries=2, name="process_dubbing_job")
def process_dubbing_job(self, job_id: str, video_path: str) -> str:
    """Run the full dubbing pipeline for one job.

    Args:
        job_id: Job primary key.
        video_path: path to uploaded source video.

    Returns:
        Output video path (or empty string on failure).
    """
    settings = get_settings()
    settings.ensure_dirs()
    logger.info(f"Starting dubbing job {job_id}: {video_path}")
    try:
        # ---- 0-30%: ASR ----
        _update(job_id, status="asr", current_stage="asr", progress=1.0)

        def _asr_cb(p: float) -> None:
            _update(job_id, progress=round(p * 0.30, 1))
            try:
                self.update_state(state="PROGRESS",
                                  meta={"stage": "asr", "percent": p})
            except Exception:
                pass

        engine = WhisperEngine()
        segments = engine.transcribe(video_path, on_progress=_asr_cb)
        if not segments:
            raise RuntimeError("ASR produced no segments")
        _save_segments(job_id, segments)
        detected = segments[0].language if segments else ""
        _update(job_id, detected_lang=detected, progress=30.0)

        # ---- 30-50%: Translate ----
        _update(job_id, status="translating",
                current_stage="translating", progress=31.0)
        factory = get_session_factory()
        with factory() as db:
            job = db.get(Job, job_id)
            target = job.target_lang if job else "vi"
            voice = (job.voice_id if job else "") or settings.edge_voice
            keep_bg = bool(job.keep_background) if job else False
            burn = bool(getattr(job, "burn_subs", 0)) if job else False
            lip = bool(getattr(job, "lipsync", 1)) if job else True

        def _tr_cb(p: float) -> None:
            _update(job_id, progress=round(30 + p * 0.20, 1))

        translator = get_translator(target_lang=target)
        translator.translate_srt(segments, source_lang=detected or "auto",
                                 target_lang=target, on_progress=_tr_cb)
        _save_segments(job_id, segments, translated=True)
        # Persist SRT artifacts for inspection
        srt_src = settings.tmp_dir / f"{job_id}_src.srt"
        segments_to_srt(
            [type(s)(idx=s.idx, start=s.start, end=s.end, text=s.text,
                     language=s.language) for s in segments], srt_src)
        _update(job_id, progress=50.0)

        # ---- 50-85%: TTS ----
        _update(job_id, status="tts", current_stage="tts", progress=51.0)
        client = get_tts_client(default_voice=voice or None)
        dubbed_wav = settings.tmp_dir / f"{job_id}_dubbed.wav"

        # Progress inside build_dubbed_audio is coarse; wrap per-segment
        # by monkey-patching client.synthesize counter.
        total = max(len(segments), 1)
        done = {"n": 0}
        orig_synth = client.synthesize

        def _counting_synth(text, voice_id, output_path):
            try:
                return orig_synth(text, voice_id, output_path)
            finally:
                done["n"] += 1
                _update(job_id,
                        progress=round(50 + done["n"] / total * 35, 1))

        client.synthesize = _counting_synth  # type: ignore[method-assign]
        build_dubbed_audio(segments, voice or settings.edge_voice,
                           dubbed_wav, client=client)
        _update(job_id, progress=85.0)

        # ---- 85-100%: Merge ----
        _update(job_id, status="merging",
                current_stage="merging", progress=86.0)
        out_path = settings.outputs_dir / f"{job_id}_dubbed.mp4"
        vi_srt = settings.tmp_dir / f"{job_id}_vi.srt"
        if burn:
            segments_to_srt(
                [type(s)(idx=s.idx, start=s.start, end=s.end,
                         text=s.translated or s.text,
                         language=s.language) for s in segments], vi_srt)
        merge_dubbed_video(video_path, dubbed_wav, out_path,
                           keep_background=keep_bg,
                           burn_subs=burn,
                           subs_srt=str(vi_srt) if burn else None)
        _update(job_id, progress=90.0)

        # ---- 90-100%: Lip-sync (Wav2Lip + GFPGAN), skips gracefully ----
        final_path = out_path
        if lip and settings.wav2lip_enabled:
            _update(job_id, status="lipsync",
                    current_stage="lipsync", progress=91.0)
            from src.video.lipsync import post_process_lipsync

            lip_out = settings.outputs_dir / f"{job_id}_lipsync.mp4"
            final_path = post_process_lipsync(
                out_path, dubbed_wav, lip_out, enable=True)
            _update(job_id, progress=99.0)

        if settings.cleanup_source_after_done:
            try:
                Path(video_path).unlink(missing_ok=True)
                logger.info(f"Cleaned up source {video_path}")
            except OSError as exc:
                logger.warning(f"Source cleanup failed: {exc}")
        _update(
            job_id, status="done", current_stage="done", progress=100.0,
            output_path=str(final_path),
            completed_at=datetime.now(timezone.utc),
        )
        logger.info(f"Job {job_id} done -> {final_path}")
        return str(final_path)
    except Exception as exc:  # noqa: BLE001 - record failure, no crash
        logger.exception(f"Job {job_id} failed: {exc}")
        _update(job_id, status="failed", current_stage="failed",
                error_message=str(exc)[:2000],
                completed_at=datetime.now(timezone.utc))
        # Only retry on transient infra errors (network/subprocess/timeout).
        # Pipeline/logic errors (bad input, no segments, bad config, etc.)
        # must NOT retry: retrying them just re-flips status back to
        # "asr"/etc. for another ~30s+ while the job is already recorded
        # as failed, which looks like a hung/oscillating progress bar to
        # the UI and Discord bot.
        transient = (
            OSError,
            TimeoutError,
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            ConnectionError,
        )
        if isinstance(exc, transient) and self.request.retries < self.max_retries:
            try:
                raise self.retry(exc=exc, countdown=30)
            except Retry:
                raise
            except Exception:
                pass
        return ""
