"""Upload endpoint: validate video, create Job, enqueue Celery task.

Supports file upload and URL download from Facebook, TikTok via yt-dlp.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from loguru import logger

from src.config import get_settings
from src.models.database import get_session_factory, init_db
from src.models.job import Job
from src.schemas.job import JobCreateResponse

router = APIRouter(prefix="/api", tags=["upload"])

ALLOWED_EXTS = {".mp4", ".mkv", ".mov", ".avi"}

# Platform detection patterns (YouTube removed - anti-bot too strong)
PLATFORM_PATTERNS = {
    "facebook": ["facebook.com", "fb.watch", "fb.w", "fbcdn"],
    "tiktok": ["tiktok.com", "vm.tiktok.com", "www.tiktok.com"],
}


def _check_auth(api_key_header: str | None = None) -> None:
    """Enforce API key only when configured AND public hostname is set."""
    from fastapi.security.utils import get_authorization_scheme_param

    settings = get_settings()
    if not (settings.api_key and settings.public_hostname):
        return
    scheme, param = get_authorization_scheme_param(api_key_header or "")
    token = param if scheme.lower() == "bearer" else (api_key_header or "")
    if token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def detect_platform(url: str) -> str:
    """Detect the social media platform from URL."""
    url_lower = url.lower()
    for platform, patterns in PLATFORM_PATTERNS.items():
        for pattern in patterns:
            if pattern in url_lower:
                return platform
    return "unknown"


def _get_platform_extractor_args(platform: str) -> str:
    """Get yt-dlp extractor args for the platform."""
    args = {
        "facebook": "facebook:video_url",
        "tiktok": "tiktok:web",
    }
    return args.get(platform, "default")


# Height caps for the `quality` field. "best" keeps the previous behaviour
# (highest available), which produced 145 MB for a 9-minute clip and made
# playback through the tunnel buffer constantly.
QUALITY_HEIGHTS = {"360": 360, "480": 480, "720": 720, "1080": 1080}
QUALITY_CHOICES = ["360", "480", "720", "1080", "best"]


def _build_format_strategies(quality: str) -> list[list[str]]:
    """Build yt-dlp ``-f`` strategies for the requested height cap.

    Args:
        quality: one of QUALITY_CHOICES, or "" / anything unknown for best.

    Returns:
        List of argument lists, tried in order until one succeeds.
    """
    height = QUALITY_HEIGHTS.get(str(quality or "").strip().lower())
    if not height:
        return [
            ["-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
             "--merge-output-format", "mp4",
             "--abort-on-unavailable-fragments"],
            ["-f", "best"],
        ]
    cap = f"[height<={height}]"
    return [
        ["-f", (f"bestvideo{cap}[ext=mp4]+bestaudio[ext=m4a]"
                f"/bestvideo{cap}+bestaudio/best{cap}"),
         "--merge-output-format", "mp4",
         "--abort-on-unavailable-fragments"],
        ["-f", f"best{cap}", "--merge-output-format", "mp4"],
    ]


def _download_url(url: str, quality: str = "best") -> tuple[str, str]:
    """Download a video from Facebook/TikTok using yt-dlp. Returns (filepath, filename).

    Args:
        url: Facebook/TikTok link.
        quality: height cap key from QUALITY_CHOICES, or "best".

    Returns:
        Tuple of (downloaded path, filename).
    """
    settings = get_settings()
    settings.ensure_dirs()
    platform = detect_platform(url)
    dest_dir = settings.tmp_dir / platform
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Find yt-dlp binary (prefer venv).
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        venv_yt = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "yt-dlp"
        if venv_yt.exists():
            yt_dlp = str(venv_yt)

    extractor_args = _get_platform_extractor_args(platform)

    # Try multiple format strategies.
    format_strategies = _build_format_strategies(quality)

    for strategy in format_strategies:
        cmd = [yt_dlp] + strategy + [
            "-o", str(dest_dir / "%(id)s.%(ext)s"),
            "--no-playlist",
            "--no-warnings",
            "--extractor-args", extractor_args,
            "--no-check-certificates",
            url,
        ]

        logger.info(f"[{platform.upper()}] yt-dlp attempt: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info(f"[{platform.upper()}] Downloaded: {url} ✓")
            break

        err = result.stderr.strip()[-300:]
        logger.warning(f"[{platform.upper()}] Attempt failed: {err}")

        if strategy is format_strategies[-1]:
            raise HTTPException(
                status_code=400,
                detail=f"⛔ [{platform.upper()}] yt-dlp lỗi: {err}\n"
                       f"Cookie của bạn có thể hết hạn. Cập nhật: /cookie",
            )

    # Find the downloaded file
    downloaded = list(dest_dir.glob("*"))
    if not downloaded:
        raise HTTPException(
            status_code=400, detail=f"[{platform.upper()}] Không tìm thấy file video"
        )
    filepath = downloaded[0]
    return str(filepath), filepath.name


@router.post("/upload", response_model=JobCreateResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_lang: str = Form(default="auto"),
    target_lang: str = Form(default="vi"),
    voice_id: str = Form(default=""),
    keep_background: bool = Form(default=False),
    burn_subs: bool = Form(default=False),
    lipsync: bool = Form(default=True),
) -> JobCreateResponse:
    """Receive a video (max MAX_UPLOAD_MB), create Job, dispatch Celery.

    Returns:
        JobCreateResponse with job_id for polling.
    """
    settings = get_settings()
    settings.ensure_dirs()
    init_db()

    filename = file.filename or "upload.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension {ext}. Allowed: {sorted(ALLOWED_EXTS)}",
        )
    job_id = uuid.uuid4().hex[:12]
    dest = settings.uploads_dir / f"{job_id}_{Path(filename).name}"
    size = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {settings.max_upload_mb}MB limit",
                    )
                out.write(chunk)
    finally:
        await file.close()

    factory = get_session_factory()
    with factory() as db:
        job = Job(
            id=job_id, filename=filename, status="queued", progress=0.0,
            source_lang=source_lang, target_lang=target_lang,
            voice_id=voice_id or "",
            keep_background=1 if keep_background else 0,
            burn_subs=1 if burn_subs else 0,
            lipsync=1 if lipsync else 0,
            current_stage="queued", source_path=str(dest),
        )
        db.add(job)
        db.commit()

    try:
        from src.worker.tasks import process_dubbing_job
        process_dubbing_job.delay(job_id, str(dest))
        logger.info(f"Job {job_id} queued: {filename} ({size} bytes)")
    except Exception as exc:
        logger.error(f"Celery dispatch failed for {job_id}: {exc}")
        with factory() as db:
            job = db.get(Job, job_id)
            if job:
                job.error_message = f"Queue dispatch failed: {exc}"
                db.commit()

    background_tasks.add_task(_noop_cleanup)
    return JobCreateResponse(job_id=job_id, filename=filename, status="queued")


@router.post("/upload/url", response_model=JobCreateResponse)
async def upload_from_url(
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    source_lang: str = Form(default="auto"),
    target_lang: str = Form(default="vi"),
    voice_id: str = Form(default=""),
    keep_background: bool = Form(default=False),
    burn_subs: bool = Form(default=False),
    lipsync: bool = Form(default=True),
    quality: str = Form(default="best"),
) -> JobCreateResponse:
    """Download a video from Facebook/TikTok via yt-dlp, create Job, dispatch Celery.

    Args:
        quality: one of QUALITY_CHOICES ("360".."1080") or "best".

    Returns:
        JobCreateResponse with job_id for polling.
    """
    settings = get_settings()
    settings.ensure_dirs()
    init_db()

    platform = detect_platform(url)
    if platform not in {"facebook", "tiktok"}:
        raise HTTPException(
            status_code=400,
            detail="Chỉ hỗ trợ link Facebook/TikTok. Link YouTube và link khác không được hỗ trợ.",
        )

    # Download video.
    video_path, filename = _download_url(url, quality)
    if not filename:
        filename = f"{platform}_video.mp4"

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        video_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Video format không hỗ trợ: {ext}. Allowed: {sorted(ALLOWED_EXTS)}",
        )

    job_id = uuid.uuid4().hex[:12]
    dest = settings.uploads_dir / f"{job_id}_{Path(filename).name}"
    Path(video_path).rename(dest)
    size = dest.stat().st_size

    factory = get_session_factory()
    with factory() as db:
        job = Job(
            id=job_id, filename=filename, status="queued", progress=0.0,
            source_lang=source_lang, target_lang=target_lang,
            voice_id=voice_id or "",
            keep_background=1 if keep_background else 0,
            burn_subs=1 if burn_subs else 0,
            lipsync=1 if lipsync else 0,
            current_stage="queued", source_path=str(dest),
        )
        db.add(job)
        db.commit()

    try:
        from src.worker.tasks import process_dubbing_job
        process_dubbing_job.delay(job_id, str(dest))
        logger.info(f"Job {job_id} queued from [{platform.upper()}]: {url}")
    except Exception as exc:
        logger.error(f"Celery dispatch failed for {job_id}: {exc}")
        with factory() as db:
            job = db.get(Job, job_id)
            if job:
                job.error_message = f"Queue dispatch failed: {exc}"
                db.commit()

    background_tasks.add_task(_noop_cleanup)
    return JobCreateResponse(job_id=job_id, filename=filename, status="queued")


async def _noop_cleanup() -> None:
    """Placeholder for temp-file cleanup hook."""
    return None