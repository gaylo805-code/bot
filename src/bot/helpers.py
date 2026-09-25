"""Pure helpers for the Discord bot (no discord.py dependency).

Kept dependency-free so every branch is unit-testable without a
Discord connection or gateway token.
"""

from __future__ import annotations

from pathlib import Path

ALLOWED_EXTS = {".mp4", ".mkv", ".mov", ".avi"}

STAGE_LABELS_VI: dict[str, str] = {
    "queued": "⏳ Đang xếp hàng...",
    "asr": "🎙️ Đang nhận dạng giọng nói (ASR)...",
    "translating": "🌐 Đang dịch sang tiếng Việt...",
    "tts": "🔊 Đang tạo giọng lồng tiếng (TTS)...",
    "merging": "🎬 Đang ghép video...",
    "lipsync": "👄 Đang khớp môi (Wav2Lip) + làm nét mặt (GFPGAN)...",
    "done": "✅ Hoàn thành!",
    "failed": "❌ Thất bại",
}

# Discord embed sidebar colors per status.
STATUS_COLORS: dict[str, int] = {
    "queued": 0x95A5A6,       # grey
    "asr": 0x3498DB,          # blue
    "translating": 0x9B59B6,  # purple
    "tts": 0xE67E22,          # orange
    "merging": 0xF1C40F,      # yellow
    "lipsync": 0xFF6B9D,       # pink
    "done": 0x2ECC71,         # green
    "failed": 0xE74C3C,       # red
}

TERMINAL_STATUSES = frozenset({"done", "failed"})


def progress_bar(percent: float, width: int = 12) -> str:
    """Render a text progress bar, e.g. `██████░░░░ 50.0%`.

    Args:
        percent: 0-100 (clamped).
        width: number of bar cells.

    Returns:
        Bar string safe for Discord code display.
    """
    pct = max(0.0, min(100.0, float(percent)))
    filled = int(round(pct / 100 * width))
    return f"{'█' * filled}{'░' * (width - filled)} {pct:.1f}%"


def human_size(num_bytes: int) -> str:
    """Format bytes as B/KB/MB/GB with 1 decimal (1024-based)."""
    size = max(0, float(num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}GB"  # pragma: no cover - unreachable


def validate_attachment(
    filename: str, size_bytes: int, max_bytes: int
) -> str | None:
    """Validate a user-uploaded video attachment.

    Args:
        filename: original attachment filename.
        size_bytes: attachment size reported by Discord.
        max_bytes: backend MAX_UPLOAD_MB cap in bytes.

    Returns:
        Vietnamese error message, or None when valid.
    """
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        return (
            f"❌ Định dạng `{ext or '(không có)'}` không hỗ trợ. "
            f"Chỉ nhận: {', '.join(sorted(ALLOWED_EXTS))}"
        )
    if size_bytes <= 0:
        return "❌ File rỗng, hãy gửi lại video khác."
    if size_bytes > max_bytes:
        return (
            f"❌ File quá lớn ({human_size(size_bytes)} > "
            f"{human_size(max_bytes)}). Hãy nén/cắt nhỏ video rồi thử lại."
        )
    return None


def stage_label(stage: str) -> str:
    """Vietnamese label for a pipeline stage (fallback: raw stage)."""
    return STAGE_LABELS_VI.get((stage or "").lower(), stage or "...")


def status_color(status: str) -> int:
    """Discord embed color int for a job status (fallback: blurple)."""
    return STATUS_COLORS.get((status or "").lower(), 0x5865F2)


def is_terminal(status: str) -> bool:
    """True when the job reached done/failed."""
    return (status or "").lower() in TERMINAL_STATUSES


def needs_link(size_bytes: int, limit_bytes: int) -> bool:
    """True when the result must be delivered as link, not attachment."""
    return size_bytes <= 0 or size_bytes > limit_bytes


def result_filename(source_filename: str, job_id: str) -> str:
    """Build a clean mp4 filename for the dubbed result."""
    stem = Path(source_filename or "video").stem or "video"
    safe = "".join(c for c in stem if c.isalnum() or c in ("-", "_", " "))[:60]
    return f"{safe.strip() or job_id}_dubbed_vi.mp4"


def public_download_url(
    api_base_url: str, public_hostname: str, job_id: str
) -> str:
    """Public download URL for a finished job.

    Prefers https://PUBLIC_HOSTNAME when configured (works for users
    outside Docker network), else falls back to API_BASE_URL.
    """
    host = (public_hostname or "").strip()
    if host:
        base = host if host.startswith("http") else f"https://{host}"
    else:
        base = api_base_url.rstrip("/")
    return f"{base}/api/jobs/{job_id}/download"


def public_view_url(
    api_base_url: str, public_hostname: str, job_id: str
) -> str:
    """Public URL to watch the dubbed video in a browser.

    Uses PUBLIC_HOSTNAME when configured, else falls back to API_BASE_URL.
    """
    host = (public_hostname or "").strip()
    if host:
        base = host if host.startswith("http") else f"https://{host}"
    else:
        base = api_base_url.rstrip("/")
    return f"{base}/api/watch/{job_id}"


def truncate(text: str, limit: int = 200) -> str:
    """Shorten long error text for embed fields."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def duration_vi(seconds: float) -> str:
    """Format seconds as Vietnamese duration ('2 phút 5 giây')."""
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} giờ {m} phút" if m else f"{h} giờ"
    if m:
        return f"{m} phút {s} giây" if s else f"{m} phút"
    return f"{s} giây"


def eta_text(progress: float, elapsed_s: float) -> str:
    """Estimate remaining time from progress velocity.

    Args:
        progress: 0-100 job progress.
        elapsed_s: seconds since tracking started.

    Returns:
        Vietnamese ETA ('còn ~3 phút') or 'đang ước tính...' when unknown.
    """
    pct = max(0.0, min(100.0, float(progress)))
    if pct <= 1.0 or elapsed_s <= 0:
        return "đang ước tính..."
    remaining = elapsed_s / pct * (100.0 - pct)
    return f"còn ~{duration_vi(remaining)}"


def is_valid_rate(rate: str) -> bool:
    """Validate an Edge-TTS SSML rate like '+20%' / '-10%' / '+0%'."""
    import re

    return bool(re.fullmatch(r"[+-]\d{1,3}%", (rate or "").strip()))


def transcript_snippet(segments: list[dict], n: int = 2) -> str:
    """First n translated lines for embed preview (fallback: source)."""
    lines = []
    for s in (segments or [])[:max(1, n)]:
        text = (s.get("translated") or s.get("source") or "").strip()
        if text:
            lines.append(f"• {truncate(text, 80)}")
    return "\n".join(lines) if lines else "_(chưa có transcript)_"


def queue_text(active_count: int) -> str:
    """Queue position line for the /dub acknowledgement."""
    if active_count <= 1:
        return "⚡ Vào làm ngay, không phải chờ!"
    return f"🚶 Trước bạn có {active_count - 1} job đang chạy."
