FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

# System deps: ffmpeg + audio libs + espeak-ng + DejaVu (subtitle burn-in)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg libsndfile1 espeak-ng curl ca-certificates fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Cloudflare Quick Tunnel (account-less trycloudflare.com URL)
RUN curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb \
    -o /tmp/cloudflared.deb \
    && dpkg -i /tmp/cloudflared.deb \
    && rm -f /tmp/cloudflared.deb

# uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY ui ./ui
COPY tests ./tests

RUN uv sync --frozen --no-dev || uv sync --no-dev

VOLUME ["/app/storage", "/app/models"]

EXPOSE 8000 8501

# Default: API. Override command per service in docker-compose.
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
