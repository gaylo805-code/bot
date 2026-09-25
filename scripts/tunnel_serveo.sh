#!/usr/bin/env bash
# Account-less public tunnel via serveo (SSH remote forward).
#
# Why not Cloudflare quick tunnels: those measure ~114 KB/s and the edge
# cancels long-lived responses mid-stream, which shows up as a player that
# spins forever. Measured on the same video, serveo sustained ~965 KB/s
# with 206/accept-ranges intact.
#
# Trade-off: serveo has no SLA and the hostname changes on every reconnect,
# so keep PUBLIC_HOSTNAME in .env in sync (see the tail of this script).
#
# Usage:
#   ./scripts/tunnel_serveo.sh            # foreground, auto-reconnect
#   ./scripts/tunnel_serveo.sh --once     # single connection, no retry
set -euo pipefail

PORT="${PORT:-8000}"
URL_FILE="${URL_FILE:-storage/tunnel_url.txt}"
ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

command -v ssh >/dev/null 2>&1 || {
  echo "ssh chua co san. Cai: apt-get install -y openssh-client"; exit 1; }
command -v curl >/dev/null 2>&1 || {
  echo "curl chua co san. Cai: apt-get install -y curl"; exit 1; }
echo "ssh: $(ssh -V 2>&1)"

# serveo prints "Forwarding HTTP traffic from https://<host>" on stdout.
# grep exits 1 on no-match, which `set -e` would treat as fatal.
grab_url() {
  grep -oE 'https://[a-z0-9.-]*serveousercontent\.com' "$1" 2>/dev/null | head -1 || true
}

run_once() {
  local log; log="$(mktemp)"
  ssh -o StrictHostKeyChecking=no \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes \
      -R "80:localhost:${PORT}" serveo.net >"$log" 2>&1 &
  local ssh_pid=$!

  # Wait for the forwarding banner instead of a fixed sleep.
  local url="" i
  for i in $(seq 1 30); do
    url="$(grab_url "$log")"
    [ -n "$url" ] && break
    kill -0 "$ssh_pid" 2>/dev/null || break
    sleep 1
  done

  if [ -z "$url" ]; then
    echo "serveo khong len duoc:"; tail -5 "$log" || true; rm -f "$log"; return 1
  fi

  mkdir -p "$(dirname "$URL_FILE")"
  echo "$url" > "$URL_FILE"
  echo
  echo "  URL: $url"
  echo "  ghi vao: $URL_FILE"
  echo
  echo "  Cap nhat .env de bot sinh dung link:"
  echo "    PUBLIC_HOSTNAME=${url#https://}"
  echo
  echo "  Dang chay. Ctrl-C de dung."
  echo

  if [ "$ONCE" = "1" ]; then
    wait "$ssh_pid"
  else
    tail -f "$log" &
  fi
  rm -f "$log"
  return 0
}

if [ "$ONCE" = "1" ]; then
  run_once
  exit $?
fi

BACKOFF=5
while true; do
  echo "--- connecting (reconnect sau ${BACKOFF}s neu rot) ---"
  run_once || echo "ket noi that bai, thu lai sau ${BACKOFF}s"
  sleep "$BACKOFF"
  # Reset backoff so a long-lived session is not penalised on its next drop.
  [ "$BACKOFF" -lt 60 ] && BACKOFF=$((BACKOFF * 2))
done
