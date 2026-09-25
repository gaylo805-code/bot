"""SQLAlchemy models: Job + Segment. SQLite by default, Postgres-ready."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all models."""


class Job(Base):
    """A dubbing job tracking the whole pipeline."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False)
    # status: queued | asr | translating | tts | merging | done | failed
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    source_lang: Mapped[str] = mapped_column(String(16), default="auto", nullable=False)
    target_lang: Mapped[str] = mapped_column(String(16), default="vi", nullable=False)
    voice_id: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    keep_background: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    burn_subs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    lipsync: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_stage: Mapped[str] = mapped_column(String(64), default="queued", nullable=False)
    source_path: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    output_path: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    detected_lang: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    segments: Mapped[list["Segment"]] = relationship(
        "Segment", back_populates="job", cascade="all, delete-orphan",
        order_by="Segment.idx",
    )


class Segment(Base):
    """One transcribed/translated/dubbed utterance."""

    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    start: Mapped[float] = mapped_column(Float, nullable=False)
    end: Mapped[float] = mapped_column(Float, nullable=False)
    source_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    translated_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    tts_path: Mapped[str] = mapped_column(String(1024), default="", nullable=False)

    job: Mapped[Job] = relationship("Job", back_populates="segments")
