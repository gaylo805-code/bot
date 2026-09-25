#!/usr/bin/env bash
# Deploy production stack: checks GPU, docker, env, then compose up.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== [1/5] GPU check ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
else
  echo "WARN: nvidia-smi not found - worker will fall back to CPU (slow)."
fi

echo "=== [2/5] Docker check ==="
if ! command -v docker >/dev/null 2>&1; then
  echo "Installing docker..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER" || true
fi
docker compose version >/dev/null 2>&1 || {
  echo "ERROR: docker compose plugin missing. Install docker-compose-plugin."
  exit 1
}

echo "=== [3/5] Env check ==="
if [ ! -f .env ]; then
  echo "Creating .env from .env.example - EDIT IT before going public!"
  cp .env.example .env
fi
if grep -q "^CLOUDFLARE_TUNNEL_TOKEN=$" .env 2>/dev/null; then
  echo "WARN: CLOUDFLARE_TUNNEL_TOKEN is empty - cloudflared will fail."
  echo "  Get a token at one.dash.cloudflare.com -> Networks -> Tunnels,"
  echo "  then set CLOUDFLARE_TUNNEL_TOKEN in .env"
fi
if grep -q "^DISCORD_BOT_TOKEN=$" .env 2>/dev/null; then
  echo "INFO: DISCORD_BOT_TOKEN is empty - bot stays off."
  echo "  Create bot at discord.com/developers/applications, set token,"
  echo "  then run: docker compose --profile bot up -d bot"
fi
mkdir -p storage/uploads storage/outputs storage/tmp models

echo "=== [4/5] Compose up ==="
docker compose up -d --build
sleep 5
docker compose ps

echo "=== [5/5] Health ==="
curl -fsS http://localhost:8000/health || echo "WARN: API not healthy yet, check 'docker compose logs api'"
echo ""
echo "API : http://localhost:8000  (docs: /docs)"
echo "UI  : http://localhost:8501"
echo "Bot : docker compose --profile bot up -d bot  (can DISCORD_BOT_TOKEN)"
if grep -q "^PUBLIC_HOSTNAME=" .env && [ -n "$(grep '^PUBLIC_HOSTNAME=' .env | cut -d= -f2)" ]; then
  echo "Public: https://$(grep '^PUBLIC_HOSTNAME=' .env | cut -d= -f2)"
fi
