"""Celery app factory for the dubbing pipeline."""

from __future__ import annotations

from celery import Celery

from src.config import get_settings


def make_celery() -> Celery:
    """Create configured Celery app (broker+backend = REDIS_URL)."""
    settings = get_settings()
    app = Celery("dubbing", broker=settings.redis_url,
                 backend=settings.redis_url)
    app.conf.update(
        task_track_started=True,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_time_limit=3600,
        task_soft_time_limit=3300,
        result_expires=86400,
        timezone="UTC",
        enable_utc=True,
    )
    return app


celery_app = make_celery()

# Register task modules (import for side effects). Must come AFTER app
# creation above: tasks.py imports celery_app back from this module.
# Without this, the worker starts but reports KeyError on every dispatch.
from src.worker import tasks  # noqa: E402,F401
