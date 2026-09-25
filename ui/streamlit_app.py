"""Streamlit UI: upload -> dub -> realtime progress -> preview/download."""

from __future__ import annotations

import os
import time

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")
POLL_INTERVAL = 2.0

STAGES_VI = {
    "queued": "⏳ Đang xếp hàng...",
    "asr": "🎙️ Đang nhận dạng giọng nói (ASR)...",
    "translating": "🌐 Đang dịch sang tiếng Việt...",
    "tts": "🔊 Đang tạo giọng lồng tiếng (TTS)...",
    "merging": "🎬 Đang ghép video...",
    "lipsync": "👄 Đang khớp môi (Wav2Lip) + làm nét mặt (GFPGAN)...",
    "done": "✅ Hoàn thành!",
    "failed": "❌ Thất bại",
}

st.set_page_config(page_title="Auto Video Dubbing", page_icon="🎬", layout="wide")
st.title("🎬 Auto Video Dubbing — Lồng tiếng Việt tự động")

with st.sidebar:
    st.header("⚙️ Cấu hình")
    source_lang = st.selectbox("Ngôn ngữ nguồn", ["auto", "en", "zh", "ja", "ko",
                                                  "fr", "de", "es"],
                               index=0)
    target_lang = st.selectbox("Ngôn ngữ đích", ["vi"], index=0)
    voice_id = st.text_input("Voice ID (VieNeu)",
                             value=os.getenv("VIENEU_VOICE_ID",
                                             "vieneu-female-01"))
    keep_background = st.checkbox("Giữ nhạc nền (mix nhỏ)", value=False)
    burn_subs = st.checkbox("Burn phụ đề Việt vào video", value=False)
    lipsync = st.checkbox("Khớp môi Wav2Lip + nét mặt GFPGAN", value=True)
    st.divider()
    st.caption(f"API: {API_BASE_URL}")

if "job_id" not in st.session_state:
    st.session_state.job_id = None

uploaded = st.file_uploader("1️⃣ Chọn video nguồn",
                            type=["mp4", "mkv", "mov", "avi"])
col1, col2 = st.columns([1, 3])
with col1:
    start_btn = st.button("🚀 Bắt đầu lồng tiếng", type="primary",
                          disabled=uploaded is None)


def api_post_upload(data: bytes, filename: str) -> dict:
    """POST video to backend; raises on error."""
    files = {"file": (filename, data, "video/mp4")}
    payload = {"source_lang": source_lang, "target_lang": target_lang,
               "voice_id": voice_id,
               "keep_background": str(keep_background).lower(),
               "burn_subs": str(burn_subs).lower(),
               "lipsync": str(lipsync).lower()}
    headers = {}
    if os.getenv("API_KEY"):
        headers["X-API-Key"] = os.environ["API_KEY"]
    with httpx.Client(timeout=300.0) as client:
        r = client.post(f"{API_BASE_URL}/api/upload", files=files,
                        data=payload, headers=headers)
        r.raise_for_status()
        return r.json()


def api_get_job(job_id: str) -> dict:
    """Poll job status."""
    with httpx.Client(timeout=15.0) as client:
        r = client.get(f"{API_BASE_URL}/api/jobs/{job_id}")
        r.raise_for_status()
        return r.json()


if start_btn and uploaded is not None:
    with st.spinner("Đang tải video lên server..."):
        try:
            res = api_post_upload(uploaded.getvalue(), uploaded.name)
            st.session_state.job_id = res["job_id"]
            st.success(f"Đã tạo job: `{res['job_id']}` — bắt đầu xử lý!")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Upload thất bại: {exc}")

job_id = st.session_state.job_id
if job_id:
    st.subheader(f"2️⃣ Tiến trình job `{job_id}`")
    bar = st.progress(0)
    stage_box = st.empty()
    log_box = st.empty()
    done = False
    while not done:
        try:
            info = api_get_job(job_id)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Không poll được job (thử lại): {exc}")
            time.sleep(POLL_INTERVAL)
            continue
        pct = float(info.get("progress", 0))
        stage = info.get("current_stage") or info.get("status", "")
        bar.progress(min(100, int(pct)), text=f"{pct:.1f}%")
        stage_box.info(STAGES_VI.get(stage, stage))
        if info.get("status") == "done":
            done = True
            bar.progress(100, text="100% — Xong!")
            st.success("🎉 Lồng tiếng hoàn thành!")
            video_url = info.get("download_url") or f"{API_BASE_URL}/api/jobs/{job_id}/download"
            st.subheader("3️⃣ Kết quả")
            st.video(video_url)
            if info.get("watch_url"):
                st.link_button("🎬 Mở trang xem", info["watch_url"])
            # Download button fetches bytes server-side (works behind tunnel).
            try:
                with httpx.Client(timeout=300.0) as client:
                    dl = client.get(video_url)
                    dl.raise_for_status()
                    st.download_button("⬇️ Tải video lồng tiếng",
                                       data=dl.content,
                                       file_name=f"{job_id}_dubbed_vi.mp4",
                                       mime="video/mp4")
            except Exception as exc:  # noqa: BLE001
                st.link_button("⬇️ Mở link download", video_url)
                st.caption(f"(Tải trực tiếp thất bại: {exc})")
        elif info.get("status") == "failed":
            done = True
            st.error(f"Job thất bại: {info.get('error_message')}")
        else:
            time.sleep(POLL_INTERVAL)
else:
    st.info("Tải video lên và bấm **Bắt đầu lồng tiếng** để chạy pipeline "
            "ASR → Dịch → TTS → Ghép video → Khớp môi.")
