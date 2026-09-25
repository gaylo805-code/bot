#!/usr/bin/env bash
# Local-dev Cloudflare Tunnel (Cách B): expose http://localhost:8000 via hostname.
# Usage: PUBLIC_HOSTNAME=dubbing.example.com ./scripts/tunnel.sh
set -euo pipefail

HOSTNAME="${PUBLIC_HOSTNAME:-${1:-}}"
if [ -z "$HOSTNAME" ]; then
  echo "Usage: PUBLIC_HOSTNAME=dubbing.example.com $0"
  exit 1
fi
if [ ! -f .env ]; then cp .env.example .env; fi

echo "=== [1/4] Install cloudflared ==="
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb \
    -o /tmp/cloudflared.deb
  sudo dpkg -i /tmp/cloudflared.deb
fi
cloudflared --version

echo "=== [2/4] Login (browser step, once) ==="
if [ ! -f ~/.cloudflared/cert.pem ]; then
  cloudflared tunnel login
fi

echo "=== [3/4] Create + route tunnel ==="
if ! cloudflared tunnel list 2>/dev/null | grep -q dubbing; then
  cloudflared tunnel create dubbing
fi
cloudflared tunnel route dns dubbing "$HOSTNAME" || true
TUNNEL_ID="$(cloudflared tunnel list 2>/dev/null | awk '/dubbing/ {print $1; exit}')"
echo "Tunnel ID: $TUNNEL_ID"

echo "=== [4/4] Write config + run ==="
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml <<EOF
tunnel: $TUNNEL_ID
credentials-file: /home/$USER/.cloudflared/$TUNNEL_ID.json
ingress:
  - hostname: $HOSTNAME
    service: http://localhost:8000
  - service: http_status:404
EOF
echo "Config written to ~/.cloudflared/config.yml"
echo "Run: cloudflared tunnel run dubbing"
exec cloudflared tunnel run dubbing
