# Multi-Person Face Mask Detection (ESP32-CAM + YOLO11n + AWS Cloud)

Hệ thống phát hiện khẩu trang nhiều người theo thời gian thực sử dụng **YOLO11n** (Ultralytics), thu thập ảnh từ **ESP32-CAM** và hỗ trợ linh hoạt 2 chế độ suy luận: **Local GPU** (Laptop) và **Cloud Inference** (AWS EC2 + Amazon S3), tích hợp **Web Dashboard (FastAPI)**, **Telegram Bot**, **Hiệu chỉnh ngưỡng `unknown`**, **Đo độ trễ 3 tầng** và **Kiểm thử độ tin cậy**.

> **Lưu ý kiến trúc:** ESP32-CAM **không** chạy YOLO trực tiếp trên board. ESP32-CAM chỉ cung cấp ảnh JPEG qua HTTP (`/capture`, `/flash`); toàn bộ xử lý AI chạy trên Laptop Gateway hoặc AWS EC2.

---

## 1. Kiến Trúc Hệ Thống

```text
                       ┌─────────────────────────────────────────────────┐
                       │                Laptop Gateway                   │
┌───────────────┐      │  FastAPI Dashboard (scripts/esp32_dashboard.py) │      ┌──────────────────────────┐
│   ESP32-CAM   │─────►│  - Single Pipeline Worker (Zero-Queue)          │─────►│   AWS EC2 Cloud API      │
│  /capture     │ WiFi │  - Mode 1: Local GPU (YOLO11n)                  │ SSM/ │  - FastAPI + Uvicorn     │
│  /flash       │      │  - Mode 2: Cloud Client (POST /v1/infer)        │ SSH  │  - YOLO11n (CPU)         │
└───────────────┘      │  - Latency & Reliability Tracker                │Tunnel│  - Timing Breakdown      │
                       └──────────────────┬──────────────────────────────┘      └────────────┬─────────────┘
                                          │                                                  │
                              ┌───────────┴───────────┐                                      ▼
                              ▼                       ▼                          ┌──────────────────────────┐
                       Web Dashboard           Telegram Bot                      │       Amazon S3          │
                     (MJPEG + Browser ACK)   (/status, /photo, /flash)           │  Model weights & Logs    │
```

---

## 2. Phân Loại Nhãn (2 Class Huấn Luyện + Runtime `unknown`)

Mô hình **YOLO11n** chỉ huấn luyện **2 class** (`mask`, `no_mask`). Nhãn `unknown` được áp dụng tại thời điểm suy luận (**runtime abstention**) theo cấu hình hiệu chỉnh trong `configs/thresholds.json`:

| Nhãn | Nguồn | Điều kiện phân loại |
|---|---|---|
| `mask` | Class ID `0` | `conf >= accept_thresh` (`0.75`) |
| `no_mask` | Class ID `1` | `conf >= accept_thresh` (`0.75`) |
| `unknown` | Runtime | `conf_floor (0.20) <= conf < accept_thresh (0.75)` |

- **Không phát hiện box nào (`zero box`)** là bỏ sót (**Missed Detection / FN**), **không** đồng nghĩa với `unknown`.
- Kết quả hiệu chỉnh ngưỡng trên tập Validation (`709` GT faces): đạt **Accepted Precision = 95.83%**, **Coverage = 69.11%**, **False-confident Rate = 4.17%**; đánh giá độc lập trên 60 ảnh ESP32-CAM thực tế đạt **Accepted Precision = 100.00%**, **False-confident Rate = 0.00%**.

---

## 3. Cấu Trúc Project

```text
mask_detection_yolo/
├── cloud_api/                     # FastAPI Cloud Inference Server (AWS EC2)
│   ├── main.py                    # Endpoints GET /health, POST /v1/infer
│   ├── model_service.py           # Nạp YOLO11n, warm-up, đo timing phía Cloud
│   ├── s3_logger.py               # Ghi lịch sử sự kiện lên S3 (non-blocking)
│   ├── schemas.py                 # Pydantic schemas request/response
│   ├── security.py                # Xác thực X-API-Key & kiểm tra magic bytes ảnh
│   ├── config.py                  # Cấu hình tập trung qua biến môi trường
│   ├── scripts/                   # Upload/download weights S3 & smoke test client
│   └── systemd/                   # Service unit mẫu cho Ubuntu EC2
├── configs/
│   └── thresholds.json            # Ngưỡng đã hiệu chỉnh (conf_floor=0.20, accept_thresh=0.75)
├── scripts/
│   ├── esp32_dashboard.py         # Web Dashboard FastAPI + Single Pipeline Worker
│   ├── benchmark_performance.py   # CLI chạy benchmark (S0–S4), thống kê Local/Cloud/E2E
│   ├── test_reliability.py        # CLI kiểm thử độ tin cậy & phục hồi kết nối (R1–R6)
│   ├── calibrate_thresholds.py    # Quét ngưỡng & hiệu chỉnh unknown
│   ├── capture_esp32_dataset.py   # Thu thập ảnh từ ESP32-CAM
│   ├── live_esp32_yolo.py         # Nhận diện trực tiếp bằng cửa sổ OpenCV
│   ├── train_yolo.py              # Huấn luyện YOLO11n
│   └── infer_yolo.py              # Suy luận hàng loạt ảnh/video
├── src/
│   ├── cloud_client.py            # HTTP client gửi ảnh lên AWS EC2 & đo RTT
│   ├── metrics.py                 # Đo độ trễ, tính Avg/P50/P95, xuất CSV & Markdown
│   ├── reliability.py             # Exponential backoff, stale detection, recovery time
│   ├── calibration.py             # Tính Accepted Precision, Coverage, FCR, MDR
│   └── telegram_service.py        # Tích hợp Telegram Bot dùng chung pipeline
├── tests/                         # Bộ Unit Tests (unittest)
├── results/                       # Kết quả calibration, benchmark performance, reliability
├── .env.example                   # Mẫu biến môi trường
├── dataset.yaml                   # Cấu hình dataset YOLO
└── requirements.txt               # Thư viện Python cho Laptop Gateway
```

---

## 4. Cài Đặt Môi Trường (Windows PowerShell)

```powershell
# Tạo và kích hoạt virtual environment
python -m venv venv
.\venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt

# Tạo file cấu hình môi trường từ file mẫu
Copy-Item .env.example .env
```

Các biến môi trường chính trong `.env`:

```ini
# ESP32-CAM & Chế độ suy luận
ESP32_IP=10.62.83.32
INFERENCE_MODE=local                  # local | cloud
CLOUD_API_URL=http://127.0.0.1:9000/v1/infer
CLOUD_SOCKET_TIMEOUT=4.0

# Telegram Bot (tùy chọn)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ALLOWED_CHAT_IDS=123456789
TELEGRAM_STALE_THRESHOLD=5.0
```

---

## 5. Hướng Dẫn Chạy Hệ Thống

### 5.1. Chạy Web Dashboard ở chế độ Local GPU

```powershell
.\venv\Scripts\python.exe scripts/esp32_dashboard.py --ip 10.62.83.32 --device 0 --port 8000
```

Hoặc chạy **Replay Mode** từ thư mục ảnh có sẵn khi không kết nối camera:

```powershell
.\venv\Scripts\python.exe scripts/esp32_dashboard.py --replay-dir dataset_raw/own_photos/session01_mask --device 0
```

### 5.2. Chạy Web Dashboard ở chế độ Cloud (AWS EC2)

1. Mở đường hầm SSM / SSH Tunnel từ Laptop (`127.0.0.1:9000`) tới Cloud API trên EC2 (`127.0.0.1:8001`).
2. Khởi động Dashboard ở chế độ Cloud:

```powershell
.\venv\Scripts\python.exe scripts/esp32_dashboard.py --mode cloud --cloud-url http://127.0.0.1:9000/v1/infer
```

3. Truy cập trình duyệt:
   - Giao diện Dashboard: `http://127.0.0.1:8000/`
   - Luồng MJPEG: `http://127.0.0.1:8000/video`
   - Trạng thái hệ thống: `http://127.0.0.1:8000/api/status`
   - Thống kê độ trễ & FPS: `http://127.0.0.1:8000/api/metrics`

*(Xem chi tiết triển khai server trên EC2 và S3 tại [`cloud_api/README.md`](cloud_api/README.md)).*

---

## 6. Đo Độ Trễ (Local & Cloud) và Benchmark Thực Nghiệm

Toàn bộ các mốc thời gian được đo bằng `time.perf_counter()` (và đồng bộ `torch.cuda.synchronize()` khi chạy GPU), tổng hợp thống kê **Average, P50 (Median), P95, Min, Max**.

### 6.1. Ba Chỉ Số Độ Trễ Chính (Local → Cloud)

| Chỉ số | Biến / Trường dữ liệu | Cách đo & Ý nghĩa |
|---|---|---|
| **a) Thời gian suy luận trên Cloud** | `cloud_inference_ms` | Đo trực tiếp trên EC2 trước/sau `model.predict()`, tách riêng `cloud_decode_ms`, `cloud_inference_ms`, `cloud_postprocess_ms`, `cloud_total_ms`. Đánh giá tốc độ mô hình trên Cloud. |
| **b) Thời gian phản hồi Local ↔ Cloud** | `cloud_request_response_ms` | Đo tại Laptop Gateway từ lúc gửi `POST /v1/infer` đến khi nhận xong HTTP response. Kèm ước lượng `cloud_transport_overhead_est_ms = max(0, RTT - cloud_server_ms)`. |
| **c) Độ trễ toàn trình** | `e2e_display_ms` & `total_loop_ms` | • `total_loop_ms`: từ lúc Gateway `GET /capture` từ ESP32-CAM đến khi suy luận, vẽ box và encode JPEG xong.<br>• `e2e_display_ms`: từ lúc `GET /capture` đến khi trình duyệt render xong frame qua `requestAnimationFrame` và gửi `POST /api/frame_ack` về Gateway (phản ánh thời gian người dùng thực sự phải chờ). |

### 6.2. Chạy Benchmark Theo Kịch Bản (`S0` – `S4`)

| Mã | Kịch bản |
|---|---|
| `S0` | Khung cảnh trống (không có người) |
| `S1` | 1 người không đeo khẩu trang |
| `S2` | 1 người đeo khẩu trang đúng cách |
| `S3` | Nhiều người với trạng thái khẩu trang khác nhau |
| `S4` | Khuôn mặt nhỏ, quay nghiêng hoặc thiếu sáng |

```powershell
# Mở trình duyệt tại http://127.0.0.1:8000/ (để ghi nhận Browser ACK cho e2e_display_ms)
# Sau đó chạy benchmark từ terminal thứ 2:
.\venv\Scripts\python.exe scripts/benchmark_performance.py --scenario S1 --duration 60

# Lặp lại 3 lần để lấy trung bình ± độ lệch chuẩn:
.\venv\Scripts\python.exe scripts/benchmark_performance.py --scenario S1 --duration 60 --repeat 3
```

Báo cáo tự động lưu tại `results/performance/`:
- Dữ liệu từng frame: `benchmark_<scenario>_<timestamp>.csv`
- Báo cáo tổng hợp Markdown (gồm bảng Breakdown, bảng Cloud Inference & RTT, bảng End-to-End Display Latency): `benchmark_<scenario>_<timestamp>_report.md`

---

## 7. Các Tính Năng Khác

### Telegram Bot
Tích hợp trong `src/telegram_service.py`, dùng chung pipeline worker duy nhất, hỗ trợ:
- `/start`: Hướng dẫn & trạng thái kết nối
- `/status`: Thống kê số lượng `mask`, `no_mask`, `unknown`, FPS và độ trễ
- `/photo`: Gửi ảnh mới nhất đã vẽ bounding box (từ chối trả ảnh cũ nếu vượt `TELEGRAM_STALE_THRESHOLD`)
- `/flash`: Bật/tắt đèn Flash trên ESP32-CAM

> **Lưu ý:** Mỗi Bot Token chỉ được chạy **1 poller** duy nhất. Nếu ESP32-CAM đang chạy `bot.getUpdates()` với cùng token sẽ gây lỗi `HTTP 409 Conflict`.

### Kiểm Thử Độ Tin Cậy & Phục Hồi (`R1` – `R6`)
Hỗ trợ Exponential Backoff có trần (`0.5s → 5.0s`), tự động khôi phục kết nối, cảnh báo dữ liệu cũ (`Stale Data Detection`), và Graceful Shutdown (`< 3s`):

```powershell
.\venv\Scripts\python.exe scripts/test_reliability.py --scenario R1 --duration 30
.\venv\Scripts\python.exe scripts/test_reliability.py --scenario R5 --duration 30 --clients 8
.\venv\Scripts\python.exe scripts/test_reliability.py --scenario R6
```

### Hiệu Chỉnh Ngưỡng `unknown`

```powershell
.\venv\Scripts\python.exe scripts/calibrate_thresholds.py --device 0 --conf-floor 0.20 --iou-thresh 0.50 --min-precision 0.95
```

---

## 8. Kiểm Thử Tự Động (Unit Tests)

```powershell
# Chạy toàn bộ unit tests
.\venv\Scripts\python.exe -m unittest discover tests -v

# Hoặc chạy từng nhóm riêng lẻ
.\venv\Scripts\python.exe -m unittest tests/test_metrics.py -v
.\venv\Scripts\python.exe -m unittest tests/test_calibration.py -v
.\venv\Scripts\python.exe -m unittest tests/test_cloud_api.py -v
.\venv\Scripts\python.exe -m unittest tests/test_telegram_service.py -v
```
