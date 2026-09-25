# Auto Video Dubbing — Lồng tiếng video sang tiếng Việt

Pipeline tự động: **Video → ASR (faster-whisper large-v3) → Dịch (LiteLLM) → TTS (VieNeu) → Ghép video (FFmpeg) → Khớp môi (Wav2Lip + GFPGAN)**.
Backend **FastAPI**, UI **Streamlit**, queue **Celery + Redis**, public internet qua **Cloudflare Tunnel**,
điều khiển VIP qua **Discord Bot** (`/dub` + embed tiến trình live).

```
video.mp4 ──▶ [ASR 0-30%] ──▶ [Translate 30-50%] ──▶ [TTS 50-85%] ──▶ [Merge 85-90%] ──▶ [Lip-sync 90-100%] ──▶ video_dubbed_vi.mp4
```

## Yêu cầu hệ thống

| Thành phần | Tối thiểu |
|---|---|
| OS | Ubuntu 22.04+ |
| Python | 3.11+ (quản lý bằng `uv`) |
| GPU | NVIDIA + CUDA 12.x (khuyến nghị; CPU vẫn chạy được, chậm) |
| RAM | 8GB (16GB+ với large-v3) |
| Docker + Compose | 24+ / v2 |
| FFmpeg | 6.x (`apt install ffmpeg`) |

## Cài đặt Ubuntu (từ đầu)

```bash
# 1. System deps
sudo apt update && sudo apt install -y ffmpeg libsndfile1 espeak-ng curl git

# 2. NVIDIA driver + CUDA 12.x (nếu có GPU)
sudo apt install -y nvidia-driver-550
# reboot, kiểm tra: nvidia-smi

# 3. uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. Clone + env
git clone <repo> auto-dubbing && cd auto-dubbing
cp .env.example .env
# Sửa .env: GEMINI_API_KEY / OPENAI_API_KEY, VIENEU_API_KEY, PUBLIC_HOSTNAME...

# 5a. Chạy DEV (không Docker)
uv sync
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 &
uv run celery -A src.worker.celery_app.celery_app worker -c 1 --loglevel=info &
API_BASE_URL=http://localhost:8000 uv run streamlit run ui/streamlit_app.py
# Bot Discord (cần DISCORD_BOT_TOKEN trong .env):
API_BASE_URL=http://localhost:8000 uv run python -m src.bot.discord_bot

# 5b. Chạy PRODUCTION (Docker, khuyến nghị)
chmod +x deploy.sh && ./deploy.sh
```

## Cấu hình `.env`

| Biến | Ví dụ | Ghi chú |
|---|---|---|
| `LLM_MODEL` | `gemini/gemini-2.0-flash` | Đổi provider tại đây: `gpt-4o-mini`, `deepseek/deepseek-chat`, `ollama/llama3.1` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` | `AIza...` | Key theo provider |
| `VIENEU_API_KEY` / `VIENEU_BASE_URL` / `VIENEU_VOICE_ID` | `https://api.vieneu.io/api/v1` | Chỉ cần khi `TTS_PROVIDER=vieneu` |
| `TTS_PROVIDER` | `edge` | `edge` (free, mặc định) \| `vieneu` (trả phí, có voice-clone) |
| `EDGE_VOICE` | `vi-VN-HoaiMyNeural` | Voice mặc định; `male` = `vi-VN-NamMinhNeural` |
| `TRANSLATE_PROVIDER` | `google` | `google` (free, mặc định) \| `llm` (trả phí, dịch slang/idiom hay hơn) |
| `EDGE_RATE` | `+0%` | Tốc độ đọc Edge-TTS (`+20%` nhanh, `-10%` chậm) |
| `CLEANUP_SOURCE_AFTER_DONE` | `false` | `true` = xóa video gốc sau khi xong (tiết kiệm disk) |
| `WAV2LIP_ENABLED` / `GFPGAN_ENABLED` | `true` | Tắt stage khớp môi / làm nét mặt |
| `WAV2LIP_CHECKPOINT` / `GFPGAN_PATH` | `./models/...` | Đường dẫn checkpoint (xem mục Lip-sync) |
| `WHISPER_MODEL/DEVICE/COMPUTE_TYPE` | `large-v3/cuda/float16` | CPU: `tiny/cpu/int8` để test |
| `MAX_UPLOAD_MB` | `500` | Giới hạn upload |
| `CLOUDFLARE_TUNNEL_TOKEN` | `eyJh...` | Lấy từ Dashboard (mục dưới) |
| `PUBLIC_HOSTNAME` | `dubbing.tenmien.com` | Dùng cho CORS + log |
| `DISCORD_BOT_TOKEN` | `MTIz...` | Token bot (mục Discord bên dưới) |
| `DISCORD_GUILD_ID` | `123456789` | (Tùy chọn) sync lệnh tức thì khi dev |
| `DISCORD_MAX_FILE_MB` | `25` | File lớn hơn → bot gửi link tải |
| `DISCORD_MAX_CONCURRENT_PER_USER` | `2` | Giới hạn job chạy/user |

Đổi provider dịch chỉ cần đổi `TRANSLATE_PROVIDER` (`google` free mặc định;
`llm` thì đổi tiếp `LLM_MODEL`: `gemini/gemini-2.0-flash`, `gpt-4o-mini`,
`deepseek/deepseek-chat`, `ollama/llama3.1` — LiteLLM tự route theo prefix).

## API

| Method | Endpoint | Mô tả |
|---|---|---|
| POST | `/api/upload` | Upload video (multipart `file` + form `source_lang/target_lang/voice_id/keep_background`) → `{job_id}` |
| GET | `/api/jobs/{id}` | Poll progress `{status, progress, current_stage}` |
| GET | `/api/jobs` | List phân trang `?page=&page_size=` |
| GET | `/api/jobs/{id}/segments` | Xem transcript/dịch |
| GET | `/api/jobs/{id}/srt?lang=vi\|src` | Tải phụ đề .srt (đã dịch / gốc) |
| GET | `/api/jobs/{id}/thumbnail` | Frame JPEG cho embed/preview |
| GET | `/api/stats` | Thống kê hàng đợi `{total,active,done,failed}` |
| POST | `/api/jobs/{id}/retry` | Chạy lại job lỗi/kẹt |
| GET | `/api/jobs/{id}/download` | Tải mp4 kết quả |
| DELETE | `/api/jobs/{id}` | Xóa job + files |
| WS | `/ws/jobs/{id}` | Realtime progress (JSON mỗi 1s) |
| GET | `/health` | Health cho Docker/Cloudflare |

```bash
curl -F "file=@demo.mp4" http://localhost:8000/api/upload
curl http://localhost:8000/api/jobs/<job_id>
```

## Cloudflare Tunnel — mở ra internet

### Cách A — Docker Compose (khuyên dùng)

1. Vào `one.dash.cloudflare.com` → **Networks → Tunnels** → **Create Tunnel** → chọn **Cloudflared**.
2. Copy **Tunnel Token** → paste vào `.env` (`CLOUDFLARE_TUNNEL_TOKEN`).
3. Tab **Public Hostnames → Add**:
   - Subdomain: `dubbing`, Domain: `tenmien.com`
   - Service Type: `HTTP`, URL: `api:8000` (tên service trong compose).
4. Set `PUBLIC_HOSTNAME=dubbing.tenmien.com` trong `.env`, chạy `./deploy.sh`.
5. (Tùy chọn) **Access → Applications** → bảo vệ hostname bằng email login / OTP.

Service `cloudflared` trong `docker-compose.yml` tự forward `PUBLIC_HOSTNAME → http://api:8000`.

### Cách B — CLI local (dev)

```bash
PUBLIC_HOSTNAME=dubbing.tenmien.com ./scripts/tunnel.sh
# Script: cài cloudflared → login → create/route → ghi ~/.cloudflared/config.yml → run
```

Config mẫu `~/.cloudflared/config.yml`:

```yaml
tunnel: <TUNNEL_ID>
credentials-file: /home/<user>/.cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: dubbing.tenmien.com
    service: http://localhost:8000
  - service: http_status:404
```

### Bảo mật

- Khi `PUBLIC_HOSTNAME` + `API_KEY` đều được set, mọi `/api/*` yêu cầu header `X-API-Key` (hoặc `Authorization: Bearer`).
- **Không expose Streamlit (8501) ra public** trừ khi bật Cloudflare Access.
- Tunnel **không thay thế authentication** — luôn bật `API_KEY` + Access cho production.

## Discord Bot (VIP) — `/dub` ngay trong chat

Bot gọi backend qua HTTP (như Streamlit UI), báo tiến trình bằng embed live,
xong việc gửi video ngay trong channel. File > `DISCORD_MAX_FILE_MB` (mặc định
25MB) thì bot gửi nút **link tải** thay vì đính kèm.

| Lệnh | Mô tả |
|---|---|
| `/dub <video> [ngôn ngữ] [voice] [nhạc nền] [phụ đề]` | Lồng tiếng, embed live có ETA + vị trí hàng đợi |
| `/status <job_id>` | Tiến trình + transcript khi xong |
| `/jobs` | 5 job mới nhất |
| `/srt <job_id>` | Tải phụ đề .srt (Việt/gốc) |
| `/retry <job_id>` | Chạy lại job lỗi/kẹt |
| `/cancel <job_id>` | Xóa job + file |
| `/stats` | Thống kê hàng đợi |
| `/ping` | Đo độ trễ bot + API |
| `/voice-test <text>` | Nghe thử giọng (chỉnh tốc độ) |
| `/voices` | Xem provider + voice TTS đang dùng |
| `/help` | Hướng dẫn + giới hạn |

TTS free mặc định là **Edge-TTS** (`vi-VN-HoaiMyNeural` nữ / `NamMinhNeural` nam,
không cần key, đổi voice qua tham số `voice_id` của `/dub`). Muốn chất lượng
VIP + voice-clone thì set `TTS_PROVIDER=vieneu` + `VIENEU_API_KEY` — pipeline
tự đổi provider, không sửa code.

Setup (5 phút):

1. Vào `discord.com/developers/applications` → **New Application** → tab **Bot** → **Reset Token** → paste vào `.env` (`DISCORD_BOT_TOKEN`). **Không cần bật privileged intents** (chỉ dùng slash command).
2. Tab **OAuth2 → URL Generator**: tick scope `bot` + `applications.commands`, quyền `Attach Files` + `Send Messages` → mở URL mời bot vào server.
3. Dev: điền `DISCORD_GUILD_ID` (Server Settings → Widget → copy ID, bật Developer Mode) để lệnh hiện ngay; production bỏ trống (global, hiện sau ~1h).
4. Chạy:
   ```bash
   # DEV
   API_BASE_URL=http://localhost:8000 uv run python -m src.bot.discord_bot
   # PRODUCTION (service riêng, chỉ chạy khi có profile bot)
   docker compose --profile bot up -d bot
   docker compose logs -f bot
   ```

Giới hạn VIP mặc định: video vào ≤ `MAX_UPLOAD_MB`, ≤ 2 job chạy/user, cooldown
10s/lệnh `/dub` (chống spam), tune qua `DISCORD_*` trong `.env`.

## Lip-sync (Wav2Lip + GFPGAN) — khớp môi sau khi ghép

Stage cuối pipeline: **Wav2Lip** lái miệng theo audio lồng tiếng, **GFPGAN**
làm nét lại mặt (khử mờ do Wav2Lip). Chạy CUDA (L4), Python 3.11+.

```bash
chmod +x scripts/setup_lipsync.sh && ./scripts/setup_lipsync.sh
# - Clone Wav2Lip (KHÔNG cài requirements.txt của nó: pin torch 1.1!)
# - Tải wav2lip_gan.pth (~146MB) + s3fd detector (~90MB)
# - Cài gfpgan + facexlib (+ patch basicsr setup.py cho py3.13)
# - Patch tương thích librosa>=0.10 cho Wav2Lip audio.py
```

Hành vi (không bao giờ gãy job):
- Không phát hiện mặt → bỏ qua, giữ video dubbed nguyên
- Thiếu checkpoint → bỏ qua + log cảnh báo
- Wav2Lip lỗi → giữ bản dubbed; GFPGAN lỗi → giữ bản Wav2Lip
- Tắt hẳn: `WAV2LIP_ENABLED=false`; tắt từng job: `/dub [khop_moi=false]`

## Test

```bash
uv sync --group dev 2>/dev/null || uv sync
uv run pytest --cov=src tests/ -v
```

Bao phủ: ASR (extract/clean/SRT), translate (mock LiteLLM + cache), TTS (mock VieNeu + atempo), merger (codec/audio track), API (upload/get/download/limit/health), bot (helpers 100% + API client 96%; `discord_bot.py` là gateway glue cần token thật nên nằm trong omit).

## Troubleshooting

| Lỗi | Cách fix |
|---|---|
| `CUDA OOM` / Whisper chậm | `.env`: `WHISPER_MODEL=medium`, `WHISPER_COMPUTE_TYPE=int8` |
| `ffmpeg not found` | `sudo apt install ffmpeg`; check `ffmpeg -version` |
| API timeout khi upload lớn | Tăng `MAX_UPLOAD_MB`, dùng multipart resumable; check nginx `client_max_body_size` nếu có proxy |
| `502` từ Cloudflare | `docker compose logs api/cloudflared`; check token, hostname route, `curl localhost:8000/health` |
| Redis unreachable | `docker compose up redis`; check `REDIS_URL` |
| VieNeu 401 | Kiểm tra `VIENEU_API_KEY` + `VIENEU_BASE_URL` (chỉ khi `TTS_PROVIDER=vieneu`) |
| Edge-TTS bị throttle/timeout | Endpoint free không SLA — pipeline tự retry 3 lần; thử lại sau hoặc chuyển `TTS_PROVIDER=vieneu` |
| Google dịch 429 rate-limit | Free endpoint giới hạn ~5 req/s — pipeline tự retry + giữ câu gốc; chạy lại sau hoặc chuyển `TRANSLATE_PROVIDER=llm` |
| Celery job kẹt `queued` | `docker compose logs worker`; Redis phải reachable từ worker |
| Worker `KeyError: 'process_dubbing_job'` | Worker cũ thiếu import task — restart worker với code mới nhất |
| Wav2Lip `mel() takes 0 positional` | Chạy `./scripts/setup_lipsync.sh` để patch librosa>=0.10 |
| GFPGAN/basicsr cài lỗi metadata | Script tự patch `setup.py` basicsr cho py3.13 rồi cài `--no-build-isolation` |
| Lệnh `/dub` không hiện | Đợi ~1h (global sync) hoặc set `DISCORD_GUILD_ID` để sync tức thì; check `docker compose logs bot` |
| Bot báo `DISCORD_BOT_TOKEN is empty` | Chưa set token trong `.env`; service bot chạy với `--profile bot` |

## Kiến trúc

```
              ┌─────────────┐   poll/WS    ┌──────────┐
UI (8501) ───▶│ FastAPI :8000│◀──────────▶ │ SQLite   │
              │ /api/*      │   enqueue    │ jobs     │
              └──────┬──────┘─────────────▶│ segments │
                     │ Celery delay        └──────────┘
                     ▼
              ┌─────────────┐    ┌───────┐
              │ worker (GPU)│───▶│ Redis │
              │ ASR→TR→TTS  │    │ :6379 │
              │ →Merge      │    └───────┘
              └──────┬──────┘
                     ▼ storage/{uploads,tmp,outputs}
Internet ◀── cloudflared ──▶ api:8000 (PUBLIC_HOSTNAME)
Discord ◀── bot (slash /dub) ──▶ api:8000 (poll 5s, embed live)
```
