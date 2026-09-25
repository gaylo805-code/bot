"""Manage an account-less Cloudflare Quick Tunnel for public result links."""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from loguru import logger

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)


class QuickTunnel:
    """Start one Cloudflare Quick Tunnel and reuse it for every finished job."""

    def __init__(self, local_url: str, enabled: bool = True) -> None:
        self.local_url = local_url
        self.enabled = enabled
        self._process: subprocess.Popen[str] | None = None
        self._url = ""
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None

    @property
    def url(self) -> str:
        with self._lock:
            return self._url

    def start(self) -> str:
        if not self.enabled or self.url:
            return self.url
        if shutil.which("cloudflared") is None:
            logger.warning(
                "Quick Tunnel disabled: cloudflared was not found. "
                "Install cloudflared or set PUBLIC_HOSTNAME."
            )
            return ""

        try:
            self._process = subprocess.Popen(
                [
                    "cloudflared", "tunnel", "--no-autoupdate",
                    "--url", self.local_url,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            logger.warning(f"Could not start Cloudflare Quick Tunnel: {exc}")
            return ""

        self._reader = threading.Thread(
            target=self._read_output, name="cloudflare-quick-tunnel", daemon=True
        )
        self._reader.start()

        # Give cloudflared a short head start, but never block app startup for long.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not self.url:
            if self._process.poll() is not None:
                break
            time.sleep(0.1)
        if self.url:
            logger.info(f"Cloudflare Quick Tunnel ready: {self.url}")
        else:
            logger.warning("Cloudflare Quick Tunnel did not publish a URL yet")
        return self.url

    def _read_output(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                match = _URL_RE.search(line)
                if match:
                    with self._lock:
                        self._url = match.group(0).rstrip("/")
                logger.debug(f"[cloudflared] {line.rstrip()}")
        except Exception as exc:  # pragma: no cover - defensive logging thread
            logger.debug(f"Cloudflare tunnel reader stopped: {exc}")

    def stop(self) -> None:
        proc = self._process
        self._process = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        with self._lock:
            self._url = ""
