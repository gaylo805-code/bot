"""FastAPI entrypoint: CORS, logging middleware, API-key guard, /health."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.routes_job import router as job_router
from src.api.routes_upload import router as upload_router
from src.api.routes_view import router as view_router
from src.config import get_settings
from src.cloudflare_quick_tunnel import QuickTunnel
from src.models.database import get_engine, init_db
from src.schemas.job import HealthResponse

_quick_tunnel: QuickTunnel | None = None


def get_public_base_url() -> str:
    """Return configured hostname or the active Quick Tunnel URL."""
    settings = get_settings()
    if settings.public_hostname.strip():
        host = settings.public_hostname.strip()
        return host if host.startswith("http") else f"https://{host}"
    return _quick_tunnel.url if _quick_tunnel else ""


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    settings.ensure_dirs()
    init_db()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        global _quick_tunnel
        if settings.quick_tunnel_enabled and not settings.public_hostname.strip():
            _quick_tunnel = QuickTunnel(
                f"http://127.0.0.1:{settings.api_port}", enabled=True
            )
            _quick_tunnel.start()
        try:
            yield
        finally:
            if _quick_tunnel:
                _quick_tunnel.stop()
            _quick_tunnel = None

    app = FastAPI(
        title="Auto Video Dubbing", version="0.1.0", lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.public_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        req_id = uuid.uuid4().hex[:8]
        start = time.time()
        # Never log Authorization / api-key headers.
        logger.info(f"[{req_id}] {request.method} {request.url.path}")
        try:
            resp = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"[{req_id}] unhandled: {exc}")
            return JSONResponse(status_code=500,
                                content={"detail": "Internal error"})
        logger.info(f"[{req_id}] -> {resp.status_code} "
                    f"{(time.time() - start) * 1000:.0f}ms")
        return resp

    @app.middleware("http")
    async def api_key_guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        settings_ = get_settings()
        # Only enforce when public + key configured; never log the key.
        if (settings_.api_key and settings_.public_hostname
                and request.url.path.startswith("/api/")):
            provided = (request.headers.get("x-api-key")
                        or request.headers.get("authorization", ""))
            if provided.lower().startswith("bearer "):
                provided = provided[7:]
            if provided != settings_.api_key:
                return JSONResponse(status_code=401,
                                    content={"detail": "Invalid API key"})
        return await call_next(request)

    app.include_router(upload_router)
    app.include_router(job_router)
    app.include_router(view_router)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Health check for Docker + Cloudflare."""
        settings_ = get_settings()
        try:
            eng = get_engine()
            with eng.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            db = "ok"
        except Exception as exc:  # noqa: BLE001
            db = f"error: {exc}"
        redis_state = "unknown"
        try:
            import redis as redis_lib
            r = redis_lib.Redis.from_url(settings_.redis_url,
                                         socket_connect_timeout=2)
            r.ping()
            redis_state = "ok"
        except Exception:
            redis_state = "unreachable"
        return HealthResponse(status="ok", redis=redis_state, database=db,
                              whisper_model=settings_.whisper_model)

    # Alias WS at /ws/jobs/{id} too (spec path without /api prefix).
    from src.api.routes_job import job_progress_ws
    app.websocket("/ws/jobs/{job_id}")(job_progress_ws)

    return app


app = create_app()
