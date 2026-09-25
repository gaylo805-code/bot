"""Discord bot package: helpers + API client + gateway."""

from src.bot.api_client import DubbingAPI
from src.bot.helpers import (
    duration_vi,
    eta_text,
    human_size,
    is_terminal,
    is_valid_rate,
    needs_link,
    progress_bar,
    public_download_url,
    queue_text,
    result_filename,
    stage_label,
    status_color,
    transcript_snippet,
    truncate,
    validate_attachment,
)

__all__ = [
    "DubbingAPI",
    "duration_vi",
    "eta_text",
    "human_size",
    "is_terminal",
    "is_valid_rate",
    "needs_link",
    "progress_bar",
    "public_download_url",
    "queue_text",
    "result_filename",
    "stage_label",
    "status_color",
    "transcript_snippet",
    "truncate",
    "validate_attachment",
]
