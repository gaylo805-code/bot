"""Central application configuration via pydantic-settings.

All secrets come from environment / .env. Never log secret values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    llm_model: str = Field(default="gemini/gemini-2.0-flash", alias="LLM_MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )

    # --- VieNeu TTS ---
    vieneu_api_key: str = Field(default="", alias="VIENEU_API_KEY")
    vieneu_base_url: str = Field(
        default="https://api.vieneu.io/api/v1", alias="VIENEU_BASE_URL"
    )
    vieneu_voice_id: str = Field(
        default="vieneu-female-01", alias="VIENEU_VOICE_ID"
    )
    vieneu_model: str = Field(default="vieneu-v4", alias="VIENEU_MODEL")

    # --- TTS provider switch: edge (free) | vieneu (paid) ---
    tts_provider: str = Field(default="edge", alias="TTS_PROVIDER")
    edge_voice: str = Field(
        default="vi-VN-HoaiMyNeural", alias="EDGE_VOICE"
    )

    # --- Translate provider switch: google (free) | llm (paid) ---
    translate_provider: str = Field(default="google", alias="TRANSLATE_PROVIDER")

    # --- Dubbing extras ---
    edge_rate: str = Field(default="+0%", alias="EDGE_RATE")
    cleanup_source_after_done: bool = Field(
        default=False, alias="CLEANUP_SOURCE_AFTER_DONE"
    )

    # --- Lip-sync post-processing: Wav2Lip (mouth) + GFPGAN (face) ---
    wav2lip_enabled: bool = Field(default=True, alias="WAV2LIP_ENABLED")
    wav2lip_dir: Path = Field(
        default=Path("./third_party/Wav2Lip"), alias="WAV2LIP_DIR"
    )
    wav2lip_checkpoint: Path = Field(
        default=Path("./models/wav2lip_gan.pth"), alias="WAV2LIP_CHECKPOINT"
    )
    wav2lip_static: bool = Field(default=False, alias="WAV2LIP_STATIC")
    gfpgan_enabled: bool = Field(default=True, alias="GFPGAN_ENABLED")
    gfpgan_path: Path = Field(
        default=Path("./models/GFPGANv1.4.pth"), alias="GFPGAN_PATH"
    )

    # --- Whisper ---
    whisper_model: str = Field(default="large-v3", alias="WHISPER_MODEL")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(
        default="int8", alias="WHISPER_COMPUTE_TYPE"
    )  # CUDA driver 580 incompatible with ctranslate2, use CPU
    whisper_beam_size: int = Field(default=5, alias="WHISPER_BEAM_SIZE")

    # --- Storage / infra ---
    storage_path: Path = Field(default=Path("./storage"), alias="STORAGE_PATH")
    database_url: str = Field(
        default="sqlite:///./storage/dubbing.db", alias="DATABASE_URL"
    )
    redis_url: str = Field(
        default="redis://redis:6379/0", alias="REDIS_URL"
    )
    max_upload_mb: int = Field(default=500, alias="MAX_UPLOAD_MB")

    # --- API ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_base_url: str = Field(default="http://api:8000", alias="API_BASE_URL")
    api_key: str = Field(default="", alias="API_KEY")

    # --- Cloudflare ---
    cloudflare_tunnel_token: str = Field(
        default="", alias="CLOUDFLARE_TUNNEL_TOKEN"
    )
    public_hostname: str = Field(default="", alias="PUBLIC_HOSTNAME")
    quick_tunnel_enabled: bool = Field(
        default=True, alias="QUICK_TUNNEL_ENABLED"
    )

    # --- Discord bot ---
    discord_bot_token: str = Field(default="", alias="DISCORD_BOT_TOKEN")
    discord_guild_id: str = Field(default="", alias="DISCORD_GUILD_ID")
    discord_max_file_mb: int = Field(default=25, alias="DISCORD_MAX_FILE_MB")
    discord_max_concurrent_per_user: int = Field(
        default=2, alias="DISCORD_MAX_CONCURRENT_PER_USER"
    )
    discord_poll_interval: float = Field(
        default=5.0, alias="DISCORD_POLL_INTERVAL"
    )
    discord_dub_cooldown_s: int = Field(
        default=10, alias="DISCORD_DUB_COOLDOWN_S"
    )
    discord_track_timeout_s: int = Field(
        default=7200, alias="DISCORD_TRACK_TIMEOUT_S"
    )

    @field_validator("storage_path", mode="before")
    @classmethod
    def _coerce_storage_path(cls, v: object) -> Path:
        if isinstance(v, Path):
            return v
        return Path(str(v))

    @field_validator("wav2lip_dir", "wav2lip_checkpoint", "gfpgan_path",
                     mode="before")
    @classmethod
    def _coerce_model_paths(cls, v: object) -> Path:
        if isinstance(v, Path):
            return v
        return Path(str(v))

    @property
    def max_upload_bytes(self) -> int:
        """Max upload size in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def discord_max_file_bytes(self) -> int:
        """Max file size (bytes) the bot may attach back to Discord."""
        return self.discord_max_file_mb * 1024 * 1024

    @property
    def uploads_dir(self) -> Path:
        """Directory for uploaded source videos."""
        return self.storage_path / "uploads"

    @property
    def outputs_dir(self) -> Path:
        """Directory for dubbed output videos."""
        return self.storage_path / "outputs"

    @property
    def tmp_dir(self) -> Path:
        """Directory for temporary pipeline files."""
        return self.storage_path / "tmp"

    def ensure_dirs(self) -> None:
        """Create storage directories if missing."""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def public_cors_origins(self) -> list[str]:
        """CORS origins: local Streamlit + optional public hostname."""
        origins = [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://ui:8501",
        ]
        if self.public_hostname:
            host = self.public_hostname.strip()
            if host and not host.startswith("http"):
                origins.append(f"https://{host}")
                origins.append(f"http://{host}")
            elif host:
                origins.append(host)
        return origins


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    s = Settings()
    return s
