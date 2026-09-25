"""Web viewer endpoints: HTML5 video player for dubbed videos.

Routes:
    GET /api/watch/{job_id}  — HTML5 video player page
    GET /api/stream/{job_id} — Raw video stream (with Range support)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from src.config import get_settings
from src.models.database import get_session_factory, init_db
from src.models.job import Job

router = APIRouter(prefix="/api", tags=["watch"])


def _get_output_path(job_id: str) -> str:
    """Get the output video path for a job, raise 404 if not found/ready."""
    init_db()
    factory = get_session_factory()
    with factory() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.output_path:
            raise HTTPException(status_code=404,
                                detail="Output not ready yet")
        return job.output_path


@router.get("/watch/{job_id}")
def watch_job(job_id: str):
    """Return an HTML5 video player page for the dubbed video."""
    settings = get_settings()
    output_path = _get_output_path(job_id)
    download_url = f"/api/jobs/{job_id}/download"
    stream_url = f"/api/stream/{job_id}"
    try:
        from src.main import get_public_base_url
        public_host = get_public_base_url()
    except Exception:
        public_host = settings.public_hostname or ""

    if public_host:
        host = public_host if public_host.startswith("http") else f"https://{public_host}"
        download_url = f"{host.rstrip('/')}/api/jobs/{job_id}/download"
        stream_url = f"{host.rstrip('/')}/api/stream/{job_id}"

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 Video Lồng Tiếng</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #0a0a0a;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 960px;
            width: 100%;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 16px;
            font-size: 1.4rem;
            color: #ff4444;
        }}
        .video-wrapper {{
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 24px rgba(0,0,0,0.5);
        }}
        video {{
            width: 100%;
            display: block;
        }}
        .info {{
            margin-top: 16px;
            text-align: center;
            color: #aaa;
            font-size: 0.9rem;
        }}
        .info a {{
            color: #4fc3f7;
            text-decoration: none;
        }}
        .info a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Video Lồng Tiếng Việt</h1>
        <div class="video-wrapper">
            <video controls autoplay>
                <source src="{stream_url}" type="video/mp4">
                Trình duyệt không hỗ trợ video.
            </video>
        </div>
        <div class="info">
            <p>📄 Tải video: <a href="{download_url}" target="_blank">Tải về</a></p>
        </div>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/stream/{job_id}")
def stream_job(job_id: str):
    """Stream the dubbed video with Range header support (for seeking)."""
    output_path = _get_output_path(job_id)
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"{job_id}_dubbed_vi.mp4",
        headers={"Accept-Ranges": "bytes"},
    )