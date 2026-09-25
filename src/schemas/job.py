"""Pydantic schemas for API requests/responses."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal[
    "queued", "asr", "translating", "tts", "merging", "done", "failed"
]


class SegmentSchema(BaseModel):
    """Single subtitle/dub segment."""

    idx: int = Field(ge=0)
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    text: str = Field(default="")
    translated: str = Field(default="")


class JobCreateResponse(BaseModel):
    """Response after upload + job creation."""

    job_id: str
    filename: str
    status: str
    message: str = "Job created, processing started"


class JobResponse(BaseModel):
    """Job info + progress for polling."""

    job_id: str
    filename: str
    status: str
    progress: float
    current_stage: str
    source_lang: str
    target_lang: str
    error_message: str | None = None
    output_path: str | None = None
    has_output: bool = False
    created_at: datetime | None = None
    completed_at: datetime | None = None
    download_url: str | None = None
    watch_url: str | None = None


class JobListResponse(BaseModel):
    """Paginated job list."""

    total: int
    page: int
    page_size: int
    jobs: list[JobResponse]


class HealthResponse(BaseModel):
    """Health check payload for Cloudflare / Docker."""

    status: str = "ok"
    redis: str = "unknown"
    database: str = "ok"
    whisper_model: str = ""
