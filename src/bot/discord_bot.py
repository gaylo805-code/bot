"""Discord gateway: VIP dubbing bot with live progress embeds.

Run:
    uv run python -m src.bot.discord_bot        # needs DISCORD_BOT_TOKEN

Slash commands:
    /dub <video> [url] [source_lang] [voice_id] [keep_background] [phu_de] - lồng tiếng
    /status <job_id>  - xem tiến trình 1 job (+ transcript)
    /jobs             - 5 job mới nhất
    /srt <job_id>     - tải phụ đề .srt
    /cancel <job_id>  - xóa job
    /retry <job_id>   - chạy lại job lỗi
    /stats            - thống kê hàng đợi
    /ping             - đo độ trễ bot + API
    /voice-test <text> - nghe thử giọng TTS
    /voices           - xem voice đang dùng
    /help             - hướng dẫn

Bot chỉ gọi FastAPI backend qua HTTP (không chạm DB trực tiếp).
"""

from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands
from loguru import logger

from src.bot.api_client import DubbingAPI, DubbingAPIError
from src.bot.helpers import (
    eta_text,
    human_size,
    is_terminal,
    is_valid_rate,
    needs_link,
    progress_bar,
    public_download_url,
    public_view_url,
    queue_text,
    result_filename,
    stage_label,
    status_color,
    transcript_snippet,
    truncate,
    validate_attachment,
)
from src.config import get_settings
from src.api.routes_upload import detect_platform

SOURCE_CHOICES = [
    app_commands.Choice(name="🌐 Tự động nhận diện", value="auto"),
    app_commands.Choice(name="🇬🇧 Tiếng Anh", value="en"),
    app_commands.Choice(name="🇨🇳 Tiếng Trung", value="zh"),
    app_commands.Choice(name="🇯🇵 Tiếng Nhật", value="ja"),
    app_commands.Choice(name="🇰🇷 Tiếng Hàn", value="ko"),
    app_commands.Choice(name="🇫🇷 Tiếng Pháp", value="fr"),
    app_commands.Choice(name="🇩🇪 Tiếng Đức", value="de"),
    app_commands.Choice(name="🇪🇸 Tiếng Tây Ban Nha", value="es"),
]


def build_progress_embed(info: dict, extra_line: str = "") -> discord.Embed:
    """Build a status embed from a GET /api/jobs/{id} payload.

    Args:
        info: job payload.
        extra_line: optional second line (e.g. ETA) under the stage label.
    """
    status = str(info.get("status", "queued"))
    stage = str(info.get("current_stage") or status)
    pct = float(info.get("progress", 0.0) or 0.0)
    filename = str(info.get("filename", ""))
    description = f"**{stage_label(stage)}**\n`{progress_bar(pct)}`"
    if extra_line:
        description += f"\n{extra_line}"
    embed = discord.Embed(
        title=f"🎬 {filename or info.get('job_id', '')}",
        description=description,
        color=status_color(status),
    )
    embed.add_field(name="🆔 Job", value=f"`{info.get('job_id')}`",
                    inline=True)
    embed.add_field(name="📊 Trạng thái", value=f"`{status}`", inline=True)
    embed.add_field(
        name="🗣️ Ngôn ngữ",
        value=f"{info.get('source_lang', '?')} → {info.get('target_lang', '?')}",
        inline=True,
    )
    if info.get("error_message"):
        embed.add_field(name="⚠️ Lỗi",
                        value=truncate(str(info["error_message"]), 300),
                        inline=False)
    return embed


class DubbingBot(commands.Bot):
    """Slash-command bot with per-user concurrency guard + cooldown."""

    def __init__(self) -> None:
        """Init bot (no privileged intents needed for slash commands)."""
        super().__init__(command_prefix="!",
                         intents=discord.Intents.default())
        settings = get_settings()
        self.api = DubbingAPI(
            base_url=settings.api_base_url, api_key=settings.api_key
        )
        self.active: dict[int, set[str]] = {}  # user_id -> job_ids
        self.last_dub: dict[int, float] = {}   # user_id -> epoch seconds

    # -- guards ---------------------------------------------------------
    def check_dub_allowed(self, user_id: int) -> str | None:
        """Return Vietnamese error or None when user may submit."""
        settings = get_settings()
        now = time.time()
        if now - self.last_dub.get(user_id, 0.0) < settings.discord_dub_cooldown_s:
            wait = int(settings.discord_dub_cooldown_s - (now - self.last_dub[user_id]))
            return f"⏳ Chờ {wait}s nữa rồi gửi tiếp nhé (chống spam)."
        running = {j for j in self.active.get(user_id, set())}
        if len(running) >= settings.discord_max_concurrent_per_user:
            return (
                f"🚦 Bạn đang có {len(running)} job chạy "
                f"(tối đa {settings.discord_max_concurrent_per_user}). "
                "Đợi xong hoặc /cancel bớt nhé."
            )
        return None

    def track_job(self, user_id: int, job_id: str) -> None:
        """Register a running job for the concurrency guard."""
        self.active.setdefault(user_id, set()).add(job_id)
        self.last_dub[user_id] = time.time()

    def untrack_job(self, user_id: int, job_id: str) -> None:
        """Release a finished job from the concurrency guard."""
        self.active.get(user_id, set()).discard(job_id)

    # -- lifecycle ------------------------------------------------------
    async def setup_hook(self) -> None:
        """Sync slash commands on startup (guild = instant, global = ~1h)."""
        settings = get_settings()
        self.tree.add_command(dub_cmd)
        self.tree.add_command(status_cmd)
        self.tree.add_command(jobs_cmd)
        self.tree.add_command(srt_cmd)
        self.tree.add_command(cancel_cmd)
        self.tree.add_command(retry_cmd)
        self.tree.add_command(stats_cmd)
        self.tree.add_command(ping_cmd)
        self.tree.add_command(voice_test_cmd)
        self.tree.add_command(voices_cmd)
        self.tree.add_command(help_cmd)
        if settings.discord_guild_id.strip():
            guild = discord.Object(id=int(settings.discord_guild_id))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info(f"Synced commands to guild {settings.discord_guild_id}")
        else:
            await self.tree.sync()
            logger.info("Synced global commands (may take up to 1h to appear)")

    async def on_ready(self) -> None:
        """Set presence after login."""
        logger.info(f"Logged in as {self.user} (id={self.user.id if self.user else '?'})")
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening, name="/dub lồng tiếng VI"
            )
        )


# --- progress tracker --------------------------------------------------
async def track_progress(
    bot: DubbingBot,
    message: discord.WebhookMessage,
    user_id: int,
    job_id: str,
    source_filename: str,
) -> None:
    """Poll backend and edit the embed until done/failed/timeout."""
    settings = get_settings()
    started = time.time()
    deadline = started + settings.discord_track_timeout_s
    try:
        while time.time() < deadline:
            try:
                info = await asyncio.to_thread(bot.api.get_job, job_id)
            except DubbingAPIError as exc:
                logger.warning(f"Poll {job_id} failed: {exc}")
                await asyncio.sleep(settings.discord_poll_interval)
                continue
            pct = float(info.get("progress", 0.0) or 0.0)
            eta = "" if is_terminal(str(info.get("status"))) else (
                f"⏱️ {eta_text(pct, time.time() - started)}"
            )
            await message.edit(embed=build_progress_embed(info, eta))
            if is_terminal(str(info.get("status"))):
                await deliver_result(bot, message, user_id, info,
                                     source_filename)
                return
            await asyncio.sleep(settings.discord_poll_interval)
        await message.edit(
            content="⌛ Hết thời gian theo dõi (job vẫn chạy). Dùng `/status` để xem tiếp.",
            embed=None,
        )
    except discord.NotFound:
        logger.warning(f"Tracking message for {job_id} was deleted")
    except Exception as exc:  # noqa: BLE001 - never crash the bot loop
        logger.exception(f"Tracker {job_id} crashed: {exc}")
    finally:
        bot.untrack_job(user_id, job_id)


async def deliver_result(
    bot: DubbingBot,
    message: discord.WebhookMessage,
    user_id: int,
    info: dict,
    source_filename: str,
) -> None:
    """Send the finished video (attachment) or a download link button."""
    settings = get_settings()
    job_id = str(info.get("job_id", ""))
    if str(info.get("status")) == "failed":
        await message.edit(
            content=f"❌ Job `{job_id}` thất bại.",
            embed=build_progress_embed(info),
        )
        return
    try:
        data, fname = await asyncio.to_thread(bot.api.download, job_id)
    except DubbingAPIError as exc:
        await message.edit(
            content=f"⚠️ Xong job nhưng tải file lỗi: {truncate(str(exc))}. "
                    f"Dùng `/status {job_id}` thử lại.",
            embed=build_progress_embed(info),
        )
        return
    mention = f"<@{user_id}>"
    # Best-effort extras for the final message (never block delivery).
    snippet = ""
    try:
        segs = await asyncio.to_thread(bot.api.get_segments, job_id)
        snippet = transcript_snippet(segs)
    except DubbingAPIError:
        pass
    thumb_file = None
    try:
        thumb_bytes = await asyncio.to_thread(bot.api.get_thumbnail, job_id)
        thumb_file = discord.File(fp=io.BytesIO(thumb_bytes),
                                  filename=f"{job_id}_thumb.jpg")
    except DubbingAPIError:
        pass
    download_url = str(info.get("download_url") or "")
    view_url = str(info.get("watch_url") or "")
    if needs_link(len(data), settings.discord_max_file_bytes):
        if not download_url:
            download_url = public_download_url(
                settings.api_base_url, settings.public_hostname, job_id
            )
        if not view_url:
            view_url = public_view_url(
                settings.api_base_url, settings.public_hostname, job_id
            )
        view = discord.ui.View()
        view.add_item(discord.ui.Button(
            label=f"⬇️ Tải video ({human_size(len(data))})", url=download_url
        ))
        view.add_item(discord.ui.Button(
            label="🎬 Xem trực tiếp trên web", url=view_url
        ))
        await message.edit(
            content=f"✅ Xong! {mention} File {human_size(len(data))} vượt "
                    f"giới hạn đính kèm Discord ({settings.discord_max_file_mb}MB).\n"
                    "⬇️ Tải về hoặc 🎬 Xem ngay trên trình duyệt:",
            embed=_final_embed(info, snippet),
            view=view,
        )
        return
    file = discord.File(
        fp=io.BytesIO(data),
        filename=result_filename(source_filename or fname, job_id),
    )
    attachments = [file] + ([thumb_file] if thumb_file else [])
    final = _final_embed(info, snippet)
    if thumb_file:
        final.set_thumbnail(url=f"attachment://{job_id}_thumb.jpg")
    view = discord.ui.View()
    if download_url:
        view.add_item(discord.ui.Button(label="⬇️ Link tải video", url=download_url))
    if view_url:
        view.add_item(discord.ui.Button(label="🎬 Xem trên web", url=view_url))
    await message.edit(
        content=f"✅ Xong! {mention} Video lồng tiếng của bạn đây:",
        embed=final,
        attachments=attachments,
        view=view if view.children else None,
    )


def _final_embed(info: dict, snippet: str) -> discord.Embed:
    """Done embed with transcript snippet."""
    embed = build_progress_embed({**info, "progress": 100.0})
    if snippet:
        embed.add_field(name="💬 Nội dung", value=snippet, inline=False)
    return embed


# --- slash commands (module-level for tree registration) ----------------
@app_commands.command(name="dub", description="🎬 Lồng tiếng video sang tiếng Việt")
@app_commands.describe(
    video="File video (mp4/mkv/mov/avi)",
    url="Link Facebook / TikTok (bỏ trống nếu dùng file)",
    source_lang="Ngôn ngữ nguồn",
    voice_id="Voice (bỏ trống = mặc định)",
    keep_background="Giữ nhạc nền gốc ở volume nhỏ",
    phu_de="Burn phụ đề Việt vào video",
    khop_moi="Khớp môi Wav2Lip + làm nét mặt GFPGAN",
)
@app_commands.choices(source_lang=SOURCE_CHOICES)
async def dub_cmd(
    interaction: discord.Interaction,
    video: discord.Attachment = None,
    url: str = "",
    source_lang: str = "auto",
    voice_id: str = "",
    keep_background: bool = False,
    phu_de: bool = False,
    khop_moi: bool = True,
) -> None:
    """Upload a video file or supported Facebook/TikTok URL."""
    bot = interaction.client  # type: ignore[assignment]
    assert isinstance(bot, DubbingBot)
    settings = get_settings()
    src = str(source_lang or "auto")

    err = bot.check_dub_allowed(interaction.user.id)
    if err:
        await interaction.response.send_message(err, ephemeral=True)
        return

    await interaction.response.defer(thinking=True)

    try:
        if url.strip() and video:
            await interaction.followup.send(
                "❌ Gửi file hoặc link Facebook/TikTok, không gửi cả hai cùng lúc.",
                ephemeral=True,
            )
            return
        if url.strip():
            platform_name = detect_platform(url)
            if platform_name not in {"facebook", "tiktok"}:
                await interaction.followup.send(
                    "❌ Chỉ hỗ trợ link Facebook/TikTok.",
                    ephemeral=True,
                )
                return
            created = await asyncio.to_thread(
                bot.api.upload_from_url,
                url.strip(), src, "vi", voice_id.strip(),
                keep_background, phu_de, khop_moi,
            )
        else:
            if not video:
                await interaction.followup.send(
                    "❌ Gửi file video hoặc link Facebook/TikTok.",
                    ephemeral=True,
                )
                return
            err = validate_attachment(
                video.filename, video.size, settings.max_upload_bytes
            )
            if err:
                await interaction.followup.send(err, ephemeral=True)
                return
            raw = await video.read()
            created = await asyncio.to_thread(
                bot.api.upload_video, raw, video.filename,
                src, "vi", voice_id.strip(), keep_background, phu_de, khop_moi,
            )
    except DubbingAPIError as exc:
        await interaction.followup.send(f"❌ {truncate(str(exc))}",
                                        ephemeral=True)
        return
    except Exception as exc:
        await interaction.followup.send(
            f"❌ Lỗi: {truncate(str(exc), 300)}", ephemeral=True)
        return

    job_id = str(created.get("job_id", ""))
    bot.track_job(interaction.user.id, job_id)
    try:
        listing = await asyncio.to_thread(bot.api.list_jobs, 1, 100)
        active = sum(1 for j in listing.get("jobs", [])
                     if not is_terminal(str(j.get("status", ""))))
        queue_line = f"\n{queue_text(active)}"
    except DubbingAPIError:
        queue_line = ""

    source_label = f"🎥 {url}" if url.strip() else f"🎥 {video.filename}" if video else "🎥"
    platform_name = detect_platform(url) if url.strip() else "file"
    platform_emoji = {"facebook": "📘", "tiktok": "🎵"}.get(platform_name, "🎥")
    msg = await interaction.followup.send(
        content=f"{platform_emoji} Đã nhận {'link ' + platform_name if url.strip() else 'video'}, job `{job_id}` bắt đầu chạy!{queue_line}",
        embed=build_progress_embed({
            "job_id": job_id, "filename": source_label,
            "status": "queued", "current_stage": "queued", "progress": 0.0,
            "source_lang": src, "target_lang": "vi",
        }),
    )
    asyncio.create_task(
        track_progress(bot, msg, interaction.user.id, job_id, source_label)
    )


@app_commands.command(name="status", description="📊 Xem tiến trình 1 job")
@app_commands.describe(job_id="ID job (copy từ tin nhắn /dub)")
async def status_cmd(interaction: discord.Interaction, job_id: str) -> None:
    """Show one job snapshot."""
    bot = interaction.client
    assert isinstance(bot, DubbingBot)
    await interaction.response.defer(ephemeral=True)
    try:
        info = await asyncio.to_thread(bot.api.get_job, job_id.strip())
    except DubbingAPIError as exc:
        await interaction.followup.send(f"❌ {truncate(str(exc))}",
                                        ephemeral=True)
        return
    embed = build_progress_embed(info)
    if is_terminal(str(info.get("status"))):
        try:
            segs = await asyncio.to_thread(bot.api.get_segments,
                                           job_id.strip())
            snippet = transcript_snippet(segs)
            if snippet:
                embed.add_field(name="💬 Nội dung", value=snippet,
                                inline=False)
        except DubbingAPIError:
            pass
    await interaction.followup.send(embed=embed, ephemeral=True)


@app_commands.command(name="jobs", description="📋 Xem 5 job mới nhất")
async def jobs_cmd(interaction: discord.Interaction) -> None:
    """List recent jobs."""
    bot = interaction.client
    assert isinstance(bot, DubbingBot)
    await interaction.response.defer(ephemeral=True)
    try:
        data = await asyncio.to_thread(bot.api.list_jobs, 1, 5)
    except DubbingAPIError as exc:
        await interaction.followup.send(f"❌ {truncate(str(exc))}",
                                        ephemeral=True)
        return
    embed = discord.Embed(title="📋 Job mới nhất",
                          color=status_color("asr"))
    jobs = data.get("jobs", [])
    if not jobs:
        embed.description = "Chưa có job nào. Gõ `/dub` để bắt đầu!"
    for j in jobs:
        embed.add_field(
            name=f"`{j.get('job_id')}` — {j.get('filename', '')}",
            value=f"{stage_label(str(j.get('current_stage') or j.get('status')))} "
                  f"`{progress_bar(float(j.get('progress', 0) or 0), 8)}`",
            inline=False,
        )
    embed.set_footer(text=f"Tổng {data.get('total', 0)} job")
    await interaction.followup.send(embed=embed, ephemeral=True)


@app_commands.command(name="cancel", description="🗑️ Xóa 1 job + file của nó")
@app_commands.describe(job_id="ID job cần xóa")
async def cancel_cmd(interaction: discord.Interaction, job_id: str) -> None:
    """Delete a job and its files."""
    bot = interaction.client
    assert isinstance(bot, DubbingBot)
    await interaction.response.defer(ephemeral=True)
    try:
        await asyncio.to_thread(bot.api.delete_job, job_id.strip())
    except DubbingAPIError as exc:
        await interaction.followup.send(f"❌ {truncate(str(exc))}",
                                        ephemeral=True)
        return
    bot.untrack_job(interaction.user.id, job_id.strip())
    await interaction.followup.send(f"🗑️ Đã xóa job `{job_id}`.",
                                    ephemeral=True)


@app_commands.command(name="srt", description="📝 Tải phụ đề .srt của job")
@app_commands.describe(job_id="ID job", lang="vi = đã dịch, src = gốc")
@app_commands.choices(lang=[
    app_commands.Choice(name="🇻🇳 Tiếng Việt (đã dịch)", value="vi"),
    app_commands.Choice(name="🎙️ Ngôn ngữ gốc", value="src"),
])
async def srt_cmd(
    interaction: discord.Interaction,
    job_id: str,
    lang: str = "vi",
) -> None:
    """Send the subtitle file as attachment."""
    bot = interaction.client
    assert isinstance(bot, DubbingBot)
    await interaction.response.defer(ephemeral=True)
    try:
        data, fname = await asyncio.to_thread(
            bot.api.get_srt, job_id.strip(), str(lang or "vi"))
    except DubbingAPIError as exc:
        await interaction.followup.send(f"❌ {truncate(str(exc))}",
                                        ephemeral=True)
        return
    file = discord.File(fp=io.BytesIO(data), filename=fname)
    await interaction.followup.send(
        content=f"📝 Phụ đề `{fname}`:", file=file, ephemeral=True)


@app_commands.command(name="stats", description="📈 Thống kê hàng đợi")
async def stats_cmd(interaction: discord.Interaction) -> None:
    """Show queue counters."""
    bot = interaction.client
    assert isinstance(bot, DubbingBot)
    await interaction.response.defer(ephemeral=True)
    try:
        stats = await asyncio.to_thread(bot.api.get_stats)
    except DubbingAPIError as exc:
        await interaction.followup.send(f"❌ {truncate(str(exc))}",
                                        ephemeral=True)
        return
    embed = discord.Embed(title="📈 Thống kê", color=status_color("done"))
    embed.add_field(name="📦 Tổng job", value=f"`{stats.get('total', 0)}`",
                    inline=True)
    embed.add_field(name="⚙️ Đang chạy", value=f"`{stats.get('active', 0)}`",
                    inline=True)
    embed.add_field(name="✅ Xong", value=f"`{stats.get('done', 0)}`",
                    inline=True)
    embed.add_field(name="❌ Lỗi", value=f"`{stats.get('failed', 0)}`",
                    inline=True)
    await interaction.followup.send(embed=embed, ephemeral=True)


@app_commands.command(name="retry", description="🔁 Chạy lại job bị lỗi/kẹt")
@app_commands.describe(job_id="ID job cần chạy lại")
async def retry_cmd(interaction: discord.Interaction, job_id: str) -> None:
    """Reset a job to queued and track it again."""
    bot = interaction.client
    assert isinstance(bot, DubbingBot)
    await interaction.response.defer()
    jid = job_id.strip()
    try:
        await asyncio.to_thread(bot.api.retry_job, jid)
        info = await asyncio.to_thread(bot.api.get_job, jid)
    except DubbingAPIError as exc:
        await interaction.followup.send(f"❌ {truncate(str(exc))}",
                                        ephemeral=True)
        return
    bot.track_job(interaction.user.id, jid)
    msg = await interaction.followup.send(
        content=f"🔁 Job `{jid}` chạy lại từ đầu!",
        embed=build_progress_embed(info),
    )
    filename = str(info.get("filename", ""))
    asyncio.create_task(
        track_progress(bot, msg, interaction.user.id, jid, filename)
    )


@app_commands.command(name="ping", description="🏓 Đo độ trễ bot + backend")
async def ping_cmd(interaction: discord.Interaction) -> None:
    """Show gateway + API latency."""
    bot = interaction.client
    assert isinstance(bot, DubbingBot)
    await interaction.response.defer(ephemeral=True)
    ws_ms = round(bot.latency * 1000) if bot.latency else -1
    try:
        payload, api_ms = await asyncio.to_thread(bot.api.health)
        api_line = (f"`{api_ms:.0f}ms` (db: {payload.get('database')}, "
                    f"redis: {payload.get('redis')})")
    except DubbingAPIError as exc:
        api_line = f"❌ {truncate(str(exc), 100)}"
    embed = discord.Embed(title="🏓 Pong!", color=status_color("asr"))
    embed.add_field(name="Discord gateway",
                    value=f"`{ws_ms}ms`" if ws_ms >= 0 else "`?`",
                    inline=True)
    embed.add_field(name="Backend API", value=api_line, inline=True)
    await interaction.followup.send(embed=embed, ephemeral=True)


@app_commands.command(name="voice-test", description="🎤 Nghe thử giọng TTS")
@app_commands.describe(
    text="Câu cần đọc thử (tối đa 300 ký tự)",
    voice="Giọng đọc",
    rate="Tốc độ (+20% nhanh hơn, -10% chậm hơn)",
)
@app_commands.choices(voice=[
    app_commands.Choice(name="👩 Nữ (Hoài My)", value="female"),
    app_commands.Choice(name="👨 Nam (Nam Minh)", value="male"),
])
async def voice_test_cmd(
    interaction: discord.Interaction,
    text: str,
    voice: str = "female",
    rate: str = "+0%",
) -> None:
    """Synthesize a sample directly (no backend job)."""
    from src.tts.edge_client import EdgeTTSClient

    await interaction.response.defer(ephemeral=True)
    text = (text or "").strip()
    if not text:
        await interaction.followup.send("❌ Nhập câu cần đọc thử.",
                                        ephemeral=True)
        return
    if len(text) > 300:
        await interaction.followup.send(
            "❌ Tối đa 300 ký tự cho bản nghe thử.", ephemeral=True)
        return
    if not is_valid_rate(rate or "+0%"):
        await interaction.followup.send(
            "❌ Tốc độ phải dạng `+20%` / `-10%` / `+0%`.", ephemeral=True)
        return
    try:
        import tempfile

        client = EdgeTTSClient(default_voice=str(voice or "female"),
                               rate=(rate or "+0%").strip())
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "voice_test.mp3")
            await asyncio.to_thread(client.synthesize, text,
                                    str(voice or "female"), out)
            data = Path(out).read_bytes()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"voice-test failed: {exc}")
        await interaction.followup.send(
            f"❌ Đọc thử lỗi: {truncate(str(exc))}", ephemeral=True)
        return
    file = discord.File(fp=io.BytesIO(data), filename="voice_test.mp3")
    await interaction.followup.send(
        content=f"🎤 `{text}`", file=file, ephemeral=True)


@app_commands.command(name="voices", description="🎙️ Xem voice TTS đang dùng")
async def voices_cmd(interaction: discord.Interaction) -> None:
    """Show configured TTS provider + voice."""
    from src.tts.factory import describe_tts

    settings = get_settings()
    embed = discord.Embed(title="🎙️ Voice lồng tiếng", color=0x9B59B6)
    embed.add_field(name="Provider hiện tại", value=describe_tts(),
                    inline=False)
    embed.add_field(
        name="Đổi voice",
        value="Thêm tham số `voice_id` trong `/dub` (edge: `female`/`male` "
              "hoặc tên voice `vi-VN-...`; vieneu: ID từ dashboard). "
              "Đổi provider bằng `TTS_PROVIDER` trong `.env`.",
        inline=False,
    )
    _ = settings
    await interaction.response.send_message(embed=embed, ephemeral=True)


@app_commands.command(name="help", description="❓ Hướng dẫn dùng bot lồng tiếng")
async def help_cmd(interaction: discord.Interaction) -> None:
    """Usage guide."""
    settings = get_settings()
    embed = discord.Embed(
        title="🎬 Auto Video Dubbing — Bot VIP",
        description=(
            "Lồng tiếng video sang tiếng Việt tự động:\n"
            "`/dub` → bot báo tiến trình live → trả video ngay trong chat."
        ),
        color=0x2ECC71,
    )
    embed.add_field(
        name="📌 Lệnh",
        value=(
            "`/dub <video>` — lồng tiếng từ file đính kèm\n"
            "`/dub url=<link>` — lồng tiếng từ Facebook / TikTok\n"
            "`/status <job_id>` — tiến trình + transcript\n"
            "`/jobs` — 5 job mới nhất\n"
            "`/srt <job_id>` — tải phụ đề .srt\n"
            "`/retry <job_id>` — chạy lại job lỗi\n"
            "`/cancel <job_id>` — xóa job\n"
            "`/stats` — thống kê hàng đợi\n"
            "`/ping` — đo độ trễ\n"
            "`/voice-test <text>` — nghe thử giọng\n"
            "`/voices` — xem voice TTS\n"
            "`/help` — hướng dẫn này"
        ),
        inline=False,
    )
    embed.add_field(
        name="⚠️ Giới hạn",
        value=(
            f"• Video vào: {', '.join(sorted(['.mp4', '.mkv', '.mov', '.avi']))}, "
            f"tối đa {settings.max_upload_mb}MB\n"
            f"• File trả về đính kèm tối đa {settings.discord_max_file_mb}MB "
            "(lớn hơn → bot gửi link tải)\n"
            f"• Tối đa {settings.discord_max_concurrent_per_user} job chạy/user"
        ),
        inline=False,
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)


def main() -> None:
    """Bot entrypoint (called by compose service `bot`)."""
    settings = get_settings()
    if not settings.discord_bot_token:
        raise SystemExit(
            "DISCORD_BOT_TOKEN is empty. Tạo bot tại "
            "https://discord.com/developers/applications → Bot → Reset Token, "
            "rồi paste vào .env"
        )
    bot = DubbingBot()

    @bot.tree.error
    async def _on_app_error(
        interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        logger.error(f"Slash command error: {error}")
        msg = f"❌ Lỗi lệnh: {truncate(str(error), 200)}"
        try:
            if interaction.response.is_done():
                await interaction.followup.send(msg, ephemeral=True)
            else:
                await interaction.response.send_message(msg, ephemeral=True)
        except discord.HTTPException:
            pass

    bot.run(settings.discord_bot_token, log_handler=None)


if __name__ == "__main__":
    main()
