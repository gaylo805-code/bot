"""Job endpoints: get/list/download/delete + realtime websocket."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from src.asr.srt_utils import SimpleSegment, segments_to_srt
from src.config import get_settings
from src.models.database import get_session_factory, init_db
from src.models.job import Job, Segment
from src.schemas.job import JobListResponse, JobResponse

router = APIRouter(prefix="/api", tags=["jobs"])


def _public_urls(job_id: str) -> tuple[str, str]:
    settings = get_settings()
    base = settings.public_hostname.strip()
    if not base:
        try:
            from src.main import get_public_base_url
            base = get_public_base_url()
        except Exception:
            base = ""
    if base and not base.startswith("http"):
        base = f"https://{base}"
    base = base.rstrip("/")
    if not base:
        return "", ""
    return (
        f"{base}/api/jobs/{job_id}/download",
        f"{base}/api/watch/{job_id}",
    )


def _to_response(job: Job) -> JobResponse:
    out = job.output_path or ""
    download_url, watch_url = _public_urls(job.id)
    return JobResponse(
        job_id=job.id, filename=job.filename, status=job.status,
        progress=float(job.progress or 0.0),
        current_stage=job.current_stage or job.status,
        source_lang=job.source_lang, target_lang=job.target_lang,
        error_message=job.error_message,
        output_path=out or None,
        has_output=bool(out and Path(out).exists()),
        created_at=job.created_at, completed_at=job.completed_at,
        download_url=download_url if out and Path(out).exists() else None,
        watch_url=watch_url if out and Path(out).exists() else None,
    )


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> JobListResponse:
    """List jobs newest-first with pagination."""
    init_db()
    factory = get_session_factory()
    with factory() as db:
        total = db.query(Job).count()
        rows = (
            db.query(Job).order_by(Job.created_at.desc())
            .offset((page - 1) * page_size).limit(page_size).all()
        )
        return JobListResponse(
            total=total, page=page, page_size=page_size,
            jobs=[_to_response(j) for j in rows],
        )


@router.get("/stats")
def get_stats() -> dict:
    """Queue stats for dashboards/bot: totals by status."""
    init_db()
    factory = get_session_factory()
    with factory() as db:
        total = db.query(Job).count()
        done = db.query(Job).filter(Job.status == "done").count()
        failed = db.query(Job).filter(Job.status == "failed").count()
        active = db.query(Job).filter(
            Job.status.notin_(["done", "failed"])).count()
        return {"total": total, "active": active,
                "done": done, "failed": failed}


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    """Get one job + progress."""
    init_db()
    factory = get_session_factory()
    with factory() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return _to_response(job)


@router.get("/jobs/{job_id}/segments")
def get_segments(job_id: str) -> dict:
    """Return transcript segments for a job (for UI inspection)."""
    init_db()
    factory = get_session_factory()
    with factory() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        segs = (
            db.query(Segment).filter(Segment.job_id == job_id)
            .order_by(Segment.idx).all()
        )
        return {"job_id": job_id, "segments": [
            {"idx": s.idx, "start": s.start, "end": s.end,
             "source": s.source_text, "translated": s.translated_text}
            for s in segs
        ]}


@router.get("/jobs/{job_id}/download")
def download_job(job_id: str):
    """Download the dubbed output video."""
    init_db()
    factory = get_session_factory()
    with factory() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.output_path or not Path(job.output_path).exists():
            raise HTTPException(status_code=404,
                                detail="Output not ready yet")
        return FileResponse(
            job.output_path, media_type="video/mp4",
            filename=f"{Path(job.filename).stem}_dubbed_vi.mp4",
        )


@router.get("/jobs/{job_id}/srt")
def download_srt(
    job_id: str,
    lang: str = Query(default="vi", pattern="^(vi|src)$"),
):
    """Download subtitles: lang=vi (translated) or src (source)."""
    settings = get_settings()
    init_db()
    factory = get_session_factory()
    with factory() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        segs = (
            db.query(Segment).filter(Segment.job_id == job_id)
            .order_by(Segment.idx).all()
        )
        if not segs:
            raise HTTPException(status_code=404,
                                detail="No transcript yet")
        items = []
        for s in segs:
            text = (s.translated_text or s.source_text) if lang == "vi" \
                else s.source_text
            items.append(SimpleSegment(idx=s.idx, start=s.start, end=s.end,
                                       text=text))
        srt_path = settings.tmp_dir / f"{job_id}_{lang}.srt"
        segments_to_srt(items, srt_path)
    return FileResponse(
        srt_path, media_type="application/x-subrip",
        filename=f"{Path(job.filename).stem}_{lang}.srt",
    )


@router.get("/jobs/{job_id}/thumbnail")
def get_thumbnail(job_id: str):
    """JPEG frame (t=1s) from output (or source) for embeds/previews."""
    from src.video.ffmpeg_utils import FFmpegError, run_ffmpeg

    settings = get_settings()
    init_db()
    factory = get_session_factory()
    with factory() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        src = None
        if job.output_path and Path(job.output_path).exists():
            src = job.output_path
        elif job.source_path and Path(job.source_path).exists():
            src = job.source_path
        if not src:
            raise HTTPException(status_code=404, detail="No video yet")
    thumb = settings.tmp_dir / f"{job_id}_thumb.jpg"
    if not thumb.exists():
        try:
            run_ffmpeg(["ffmpeg", "-y", "-v", "error", "-ss", "1",
                        "-i", str(src), "-frames:v", "1",
                        "-q:v", "3", str(thumb)])
        except FFmpegError:
            # Short clips: retry from the very first frame.
            run_ffmpeg(["ffmpeg", "-y", "-v", "error",
                        "-i", str(src), "-frames:v", "1",
                        "-q:v", "3", str(thumb)])
    return FileResponse(thumb, media_type="image/jpeg",
                        filename=f"{job_id}_thumb.jpg")


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: str) -> dict:
    """Reset a failed/stuck job to queued and re-dispatch the worker."""
    from loguru import logger

    init_db()
    factory = get_session_factory()
    with factory() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status not in ("failed", "done", "queued", "asr",
                              "translating", "tts", "merging"):
            raise HTTPException(status_code=400, detail="Job is running")
        if not job.source_path or not Path(job.source_path).exists():
            raise HTTPException(status_code=410,
                                detail="Source file gone, re-upload instead")
        job.status = "queued"
        job.progress = 0.0
        job.current_stage = "queued"
        job.error_message = None
        job.completed_at = None
        source = job.source_path
        db.commit()
    try:
        from src.worker.tasks import process_dubbing_job
        process_dubbing_job.delay(job_id, source)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Retry dispatch failed for {job_id}: {exc}")
        raise HTTPException(status_code=503,
                            detail=f"Queue unreachable: {exc}")
    return {"job_id": job_id, "status": "queued"}


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """Delete job + its files (source, output, tmp artifacts)."""
    settings = get_settings()
    init_db()
    factory = get_session_factory()
    with factory() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        for p in (job.source_path, job.output_path):
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
        for tmp in settings.tmp_dir.glob(f"{job_id}*"):
            try:
                if tmp.is_dir():
                    shutil.rmtree(tmp, ignore_errors=True)
                else:
                    tmp.unlink(missing_ok=True)
            except OSError:
                pass
        db.delete(job)
        db.commit()
    return {"job_id": job_id, "deleted": True}


@router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str) -> None:
    """Push progress JSON every 1s until done/failed or disconnect."""
    await websocket.accept()
    try:
        while True:
            init_db()
            factory = get_session_factory()
            with factory() as db:
                job = db.get(Job, job_id)
                if not job:
                    await websocket.send_json({"error": "Job not found"})
                    await websocket.close()
                    return
                await websocket.send_json({
                    "job_id": job.id, "status": job.status,
                    "progress": float(job.progress or 0.0),
                    "stage": job.current_stage,
                    "error": job.error_message,
                })
                if job.status in ("done", "failed"):
                    await websocket.close()
                    return
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
