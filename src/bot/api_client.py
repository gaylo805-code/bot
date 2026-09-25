"""HTTP client for the dubbing backend (used by bot + UI-style flows).

Thin wrapper over the FastAPI endpoints so the Discord gateway never
touches the DB directly. Uses plain httpx (already a project dep).
"""

from __future__ import annotations

import httpx
from loguru import logger


class DubbingAPIError(RuntimeError):
    """Raised for transport errors or non-2xx backend responses."""


class DubbingAPI:
    """Sync HTTP client for the Auto-Dubbing FastAPI backend."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_upload: float = 300.0,
        timeout: float = 15.0,
    ) -> None:
        """Init client.

        Args:
            base_url: e.g. http://api:8000 (compose) or http://localhost:8000.
            api_key: sent as X-API-Key when set (public deployments).
            timeout_upload: seconds for video upload.
            timeout: seconds for normal JSON calls.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_upload = timeout_upload
        self.timeout = timeout

    # -- internals ------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        # Never log the key value.
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def _raise(self, resp: httpx.Response, action: str) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:  # noqa: BLE001
                detail = resp.text
            raise DubbingAPIError(
                f"{action} thất bại (HTTP {resp.status_code}): {detail}"
            ) from exc

    # -- API ------------------------------------------------------------
    def upload_video(
        self,
        data: bytes,
        filename: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        voice_id: str = "",
        keep_background: bool = False,
        burn_subs: bool = False,
        lipsync: bool = True,
    ) -> dict:
        """POST /api/upload -> response dict with job_id."""
        logger.info(f"Bot uploading {filename} ({len(data)} bytes)")
        files = {"file": (filename, data, "video/mp4")}
        form = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "voice_id": voice_id,
            "keep_background": str(bool(keep_background)).lower(),
            "burn_subs": str(bool(burn_subs)).lower(),
            "lipsync": str(bool(lipsync)).lower(),
        }
        with httpx.Client(timeout=self.timeout_upload) as client:
            resp = client.post(
                f"{self.base_url}/api/upload",
                files=files, data=form, headers=self._headers(),
            )
        self._raise(resp, "Upload video")
        return resp.json()

    def get_job(self, job_id: str) -> dict:
        """GET /api/jobs/{id} -> status/progress dict."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                f"{self.base_url}/api/jobs/{job_id}", headers=self._headers()
            )
        self._raise(resp, "Lấy trạng thái job")
        return resp.json()

    def list_jobs(self, page: int = 1, page_size: int = 5) -> dict:
        """GET /api/jobs -> paginated job list."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                f"{self.base_url}/api/jobs",
                params={"page": page, "page_size": page_size},
                headers=self._headers(),
            )
        self._raise(resp, "Liệt kê job")
        return resp.json()

    def delete_job(self, job_id: str) -> dict:
        """DELETE /api/jobs/{id} -> deletion receipt."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.delete(
                f"{self.base_url}/api/jobs/{job_id}", headers=self._headers()
            )
        self._raise(resp, "Xóa job")
        return resp.json()

    def download(self, job_id: str) -> tuple[bytes, str]:
        """GET /api/jobs/{id}/download -> (content, filename)."""
        with httpx.Client(timeout=self.timeout_upload) as client:
            resp = client.get(
                f"{self.base_url}/api/jobs/{job_id}/download",
                headers=self._headers(),
            )
        self._raise(resp, "Tải video kết quả")
        filename = f"{job_id}_dubbed_vi.mp4"
        ctype = resp.headers.get("content-disposition", "")
        if "filename=" in ctype:
            filename = ctype.split("filename=")[-1].strip().strip('"')
        return resp.content, filename

    def health(self) -> tuple[dict, float]:
        """GET /health -> (payload, latency_ms)."""
        import time as _time

        start = _time.time()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/health",
                              headers=self._headers())
        self._raise(resp, "Kiểm tra backend")
        return resp.json(), ( _time.time() - start) * 1000

    def get_segments(self, job_id: str) -> list[dict]:
        """GET /api/jobs/{id}/segments -> segment list."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                f"{self.base_url}/api/jobs/{job_id}/segments",
                headers=self._headers(),
            )
        self._raise(resp, "Lấy transcript")
        return resp.json().get("segments", [])

    def get_srt(self, job_id: str, lang: str = "vi") -> tuple[bytes, str]:
        """GET /api/jobs/{id}/srt -> (content, filename)."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                f"{self.base_url}/api/jobs/{job_id}/srt",
                params={"lang": lang}, headers=self._headers(),
            )
        self._raise(resp, "Tải phụ đề")
        return resp.content, f"{job_id}_{lang}.srt"

    def get_thumbnail(self, job_id: str) -> bytes:
        """GET /api/jobs/{id}/thumbnail -> JPEG bytes."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                f"{self.base_url}/api/jobs/{job_id}/thumbnail",
                headers=self._headers(),
            )
        self._raise(resp, "Lấy thumbnail")
        return resp.content

    def get_stats(self) -> dict:
        """GET /api/stats -> queue counters."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/api/stats",
                              headers=self._headers())
        self._raise(resp, "Lấy thống kê")
        return resp.json()

    def retry_job(self, job_id: str) -> dict:
        """POST /api/jobs/{id}/retry -> reset + re-dispatch."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/api/jobs/{job_id}/retry",
                headers=self._headers(),
            )
        self._raise(resp, "Chạy lại job")
        return resp.json()

    def upload_from_url(
        self,
        url: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        voice_id: str = "",
        keep_background: bool = False,
        burn_subs: bool = False,
        lipsync: bool = True,
        quality: str = "best",
    ) -> dict:
        """POST /api/upload/url -> download a supported social video, create job.

        Args:
            quality: height cap for yt-dlp ("360", "480", "720", "1080", "best").

        Returns:
            JobCreateResponse dict.
        """
        logger.info(f"Bot uploading from social URL: {url} (quality={quality})")
        with httpx.Client(timeout=self.timeout_upload) as client:
            resp = client.post(
                f"{self.base_url}/api/upload/url",
                data={
                    "url": url,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "voice_id": voice_id,
                    "keep_background": str(bool(keep_background)).lower(),
                    "burn_subs": str(bool(burn_subs)).lower(),
                    "lipsync": str(bool(lipsync)).lower(),
                    "quality": quality or "best",
                },
                headers=self._headers(),
            )
        self._raise(resp, "Upload từ link mạng xã hội")
        return resp.json()
