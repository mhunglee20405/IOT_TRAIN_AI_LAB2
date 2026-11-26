# Multi-Person Face Mask Detection — ESP32-CAM, YOLO11n & AWS Cloud

Hệ thống phát hiện khẩu trang nhiều người theo thời gian thực sử dụng **ESP32-CAM**, **YOLO11n (Ultralytics)** và kiến trúc suy luận lai **Local GPU / AWS Cloud (EC2 + SSM Tunnel + S3)**, tích hợp **Web Dashboard FastAPI**, **Telegram Bot**, **Hiệu chỉnh ngưỡng Unknown**, **Kiểm thử độ tin cậy (R1–R6)** và **Đo lường độ trễ toàn trình (Local ↔ Cloud & Browser ACK)**.

> **Lưu ý kiến trúc:** ESP32-CAM **không** chạy mô hình AI trực tiếp. Thiết bị chỉ chụp/phát ảnh JPEG qua HTTP (`/capture`, `/flash`) để **Laptop Gateway** điều phối suy luận trên **Local GPU** hoặc đẩy lên **AWS EC2 Cloud API**.

---

## 1. Kiến Trúc Hệ Thống

```text
[ ESP32-CAM ] ── HTTP GET /capture (JPEG over Wi-Fi) ──► [ Laptop Gateway (FastAPI Dashboard) ]
                                                                │
                   ┌────────────────────────────────────────────┴────────────────────────────────────────────┐
                   ▼ (Local Mode: GPU)                                                                       ▼ (Cloud Mode: AWS)
        [ YOLO11n trên Local GPU ]                                                           [ AWS SSM / SSH Tunnel (127.0.0.1:9000) ]
                   │                                                                                         │
                   │                                                                                         ▼
                   │                                                                     [ AWS EC2 Cloud API (Uvicorn :8001) ]
                   │                                                                       • YOLO11n CPU Inference + Timing Breakdown
                   │                                                                       • Non-blocking Result Logging ──► [ Amazon S3 ]
                   │                                                                                         │
                   └────────────────────────────────────────────┬────────────────────────────────────────────┘
                                                                ▼
                                                [ Render Bounding Boxes + Metrics ]
                                                     ├── Web Dashboard (MJPEG + Browser ACK E2E Latency)
                                                     ├── REST API (/api/status, /api/metrics, /api/benchmark/*)
                                                     └── Telegram Bot (/status, /photo, /flash)
```

---

## 2. Phân Loại Nhãn & Cơ Chế Ngưỡng `unknown`

Mô hình YOLO11n được huấn luyện chuẩn **2 lớp** (`mask`, `no_mask`). Nhãn `unknown` được gán tại thời điểm **runtime** (không thêm vào tập huấn luyện) nhằm hạn chế các quyết định sai lầm tự tin cao (*False-confident decisions*):

| Nhãn hiển thị | Nguồn gốc | Quy tắc áp dụng (`configs/thresholds.json`) |
|---|---|---|
| `mask` | Class ID `0` | `confidence >= 0.75` (`accept_thresh`) |
| `no_mask` | Class ID `1` | `confidence >= 0.75` (`accept_thresh`) |
| `unknown` | Runtime abstention | `0.20 <= confidence < 0.75` (`conf_floor <= conf < accept_thresh`) |

- **Kết quả hiệu chỉnh ngưỡng trên tập Validation** (`709` GT faces): chọn `conf_floor = 0.20`, `accept_thresh = 0.75` → **Accepted Precision: 95.83%**, **Coverage: 69.11%**, **Unknown Rate: 38.58%**, **False-confident Rate: 4.17%**.
- **Đánh giá độc lập trên 60 ảnh ESP32-CAM thực tế**: **Accepted Precision: 100.00%**, **Coverage: 68.18%**, **Unknown Rate: 31.82%**, **False-confident Rate: 0.00%**.

---

## 3. Cấu Trúc Thư Mục

```text
mask_detection_yolo/
├── cloud_api/                       # FastAPI Cloud Inference Server (AWS EC2)
│   ├── main.py                      # Endpoints GET /health, POST /v1/infer
│   ├── model_service.py             # Nạp YOLO11n, đo cloud_decode/inference/postprocess/total_ms
│   ├── s3_logger.py                 # Ghi lịch sử JSON lên Amazon S3 (background thread, non-blocking)
│   ├── config.py / schemas.py / security.py
│   ├── scripts/                     # test_client.py, upload_to_s3.py, download_model_s3.py
│   ├── systemd/yolo_cloud_api.service
│   └── README.md
├── configs/
│   └── thresholds.json              # Cấu hình ngưỡng hiệu chuẩn tập trung (conf_floor, accept_thresh)
├── scripts/
│   ├── esp32_dashboard.py           # Laptop Gateway: FastAPI Web Dashboard + Single Pipeline Worker
│   ├── benchmark_performance.py     # CLI chạy Benchmark (S0–S4), xuất CSV/Markdown & in 3 nhóm độ trễ
│   ├── test_reliability.py          # CLI kiểm thử độ tin cậy & phục hồi kết nối (R1–R6)
│   ├── calibrate_thresholds.py      # Hiệu chỉnh ngưỡng unknown & đánh giá ảnh ESP32-CAM
│   ├── capture_esp32_dataset.py     # Thu thập ảnh thô từ ESP32-CAM (/capture)
│   ├── live_esp32_yolo.py           # Nhận diện trực tiếp qua cửa sổ OpenCV
│   ├── train_yolo.py / infer_yolo.py / check_dataset.py / label_assistant.py / ...
├── src/
│   ├── cloud_client.py              # HTTP Client gọi EC2 /v1/infer & bấm giờ Local ↔ Cloud RTT
│   ├── metrics.py                   # PerformanceTracker, BenchmarkSession, thống kê Avg/P50/P95 & E2E
│   ├── reliability.py               # Exponential backoff, recovery time, stale data detection
│   ├── calibration.py               # Thuật toán tính Precision/Coverage/Cost sweep
│   └── telegram_service.py          # Telegram Bot tích hợp chung worker (có stale check)
├── tests/                           # Unit tests (metrics, cloud_api, cloud_gateway, calibration, reliability, telegram)
├── results/                         # Báo cáo thực nghiệm (calibration/, performance/, reliability/)
├── dataset.yaml
├── requirements.txt
└── .env.example
```

---

## 4. Cài Đặt Môi Trường (Windows PowerShell)

```powershell
# 1. Tạo và kích hoạt môi trường ảo
python -m venv venv
.\venv\Scripts\activate

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Tạo file cấu hình .env từ mẫu
Copy-Item .env.example .env
```

Các biến môi trường chính trong `.env`:

```ini
# ESP32-CAM
ESP32_IP=10.62.83.32

# Chế độ suy luận mặc định: local hoặc cloud
INFERENCE_MODE=local
CLOUD_API_URL=http://127.0.0.1:9000/v1/infer
CLOUD_SOCKET_TIMEOUT=4.0

# Telegram Bot (tùy chọn)
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_CHAT_IDS=
TELEGRAM_STALE_THRESHOLD=5.0
```

---

## 5. Hướng Dẫn Chạy Hệ Thống

### 5.1. Chạy Web Dashboard (Laptop Gateway)

```powershell
# Chế độ Local GPU (mặc định, đọc ngưỡng từ configs/thresholds.json)
.\venv\Scripts\python.exe scripts/esp32_dashboard.py --ip 10.62.83.32 --device 0 --port 8000

# Chế độ AWS Cloud (qua SSM/SSH Tunnel tại 127.0.0.1:9000)
.\venv\Scripts\python.exe scripts/esp32_dashboard.py --mode cloud --cloud-url http://127.0.0.1:9000/v1/infer --ip 10.62.83.32

# Chế độ Replay ảnh mẫu (khi ESP32-CAM offline)
.\venv\Scripts\python.exe scripts/esp32_dashboard.py --replay-dir dataset_raw/own_photos/session01_mask --device 0
```

Truy cập trên trình duyệt:
- **Web Dashboard:** `http://127.0.0.1:8000/`
- **Luồng MJPEG:** `http://127.0.0.1:8000/video`
- **Trạng thái & độ trễ tức thời:** `GET http://127.0.0.1:8000/api/status`
- **Thống kê hiệu năng (Avg, P50, P95):** `GET http://127.0.0.1:8000/api/metrics`

### 5.2. Chạy Cloud Inference API & Smoke Test

Xem chi tiết triển khai AWS EC2, IAM Role, S3 và Systemd tại [cloud_api/README.md](cloud_api/README.md).

```powershell
# Chạy thử Cloud API cục bộ
python -m uvicorn cloud_api.main:app --host 127.0.0.1 --port 8001

# Smoke test đo Cloud Inference vs Client Round-Trip Time
.\venv\Scripts\python.exe cloud_api/scripts/test_client.py --url http://127.0.0.1:8001 --iterations 5
```

---

## 6. Đo Lường Độ Trễ (Local ↔ Cloud & Toàn Trình) & Benchmark

Toàn bộ các mốc thời gian được đo bằng **`time.perf_counter()`** (đổi sang mili-giây bằng `* 1000.0`, có `torch.cuda.synchronize()` khi chạy GPU) và tổng hợp thống kê **Average, P50 (Median), P95, Min, Max**.

### 6.1. Ba nhóm độ trễ cốt lõi (Local → Cloud)

| Nhóm chỉ số | Biến đo trong code | Cách đo thực tế |
|---|---|---|
| **a) Thời gian suy luận trên Cloud** | `cloud_inference_ms` | Đo trực tiếp trên EC2 trước và sau `model.predict()` (`cloud_api/model_service.py`), trả về trong JSON `timing_ms` của `POST /v1/infer` (kèm `cloud_decode_ms`, `cloud_postprocess_ms`, `cloud_total_ms`). |
| **b) Thời gian phản hồi Local ↔ Cloud** | `cloud_request_response_ms` | Đo tại Laptop Gateway (`src/cloud_client.py`) từ lúc gửi `POST /v1/infer` đến khi nhận xong response. Ước lượng overhead truyền tải: `max(0, cloud_request_response_ms - cloud_server_ms)`. |
| **c) Độ trễ toàn trình** | `total_loop_ms` & `e2e_display_ms` | • `total_loop_ms`: từ lúc Gateway `GET /capture` từ ESP32-CAM → suy luận (Local/Cloud) → vẽ bbox → encode JPEG xong.<br>• `e2e_display_ms` (**Browser ACK**): từ lúc Gateway bắt đầu `GET /capture` đến khi trình duyệt render frame trong `requestAnimationFrame` và gửi `POST /api/frame_ack` về Gateway (dùng chung đồng hồ `perf_counter()` trên Gateway nên không lệch clock). |

### 6.2. Các kịch bản Benchmark (`S0`–`S4`)

| Mã | Kịch bản | Mô tả |
|---|---|---|
| **S0** | Khung cảnh trống | Không có người trước camera |
| **S1** | 1 người không khẩu trang | 1 người đối diện camera, không đeo khẩu trang |
| **S2** | 1 người đeo khẩu trang | 1 người đeo khẩu trang đúng cách |
| **S3** | Nhiều người hỗn hợp | Nhiều người cùng khung hình với trạng thái khác nhau |
| **S4** | Góc nghiêng / thiếu sáng | Khuôn mặt nhỏ, quay nghiêng hoặc ánh sáng yếu |

### 6.3. Lệnh chạy Benchmark

```powershell
# 1. Mở Dashboard trước (nên mở trang http://127.0.0.1:8000/ trên trình duyệt để ghi nhận Browser ACK cho e2e_display_ms)
.\venv\Scripts\python.exe scripts/esp32_dashboard.py --mode cloud

# 2. Chạy benchmark kịch bản S1 trong 60 giây (tự động in bảng 3 nhóm độ trễ và xuất CSV + Markdown)
.\venv\Scripts\python.exe scripts/benchmark_performance.py --scenario S1 --duration 60

# 3. Lặp lại 3 lần để lấy trung bình ± độ lệch chuẩn
.\venv\Scripts\python.exe scripts/benchmark_performance.py --scenario S0 --duration 60 --repeat 3
```

Kết quả được xuất tự động tại `results/performance/`:
- `benchmark_<scenario>_<timestamp>.csv` — dữ liệu chi tiết từng frame.
- `benchmark_<scenario>_<timestamp>_report.md` — báo cáo Markdown gồm bảng Local/Pipeline, bảng AWS Cloud Inference (Avg/P50/P95/Min/Max) và bảng End-to-End Display Latency.

---

## 7. Độ Tin Cậy, Hiệu Chỉnh Ngưỡng & Telegram Bot

### 7.1. Kiểm thử Độ tin cậy & Phục hồi kết nối (`R1`–`R6`)
- **Exponential Backoff có trần** (`0.5s × 1.5^n`, tối đa `5.0s`), tự động phục hồi và đo `recovery_time`.
- **Stale Data Detection**: tự động cảnh báo khi dữ liệu cũ hơn ngưỡng (`4.0s` trên Dashboard, `5.0s` trên Telegram).
- **Graceful Shutdown**: dùng `threading.Event` giúp dừng an toàn trong `< 1.0s`.

```powershell
.\venv\Scripts\python.exe scripts/test_reliability.py --scenario R1 --duration 30
.\venv\Scripts\python.exe scripts/test_reliability.py --scenario R5 --duration 30 --clients 8
.\venv\Scripts\python.exe scripts/test_reliability.py --scenario R6
```

### 7.2. Hiệu chỉnh ngưỡng Unknown (`configs/thresholds.json`)

```powershell
.\venv\Scripts\python.exe scripts/calibrate_thresholds.py --device 0 --conf-floor 0.20 --iou-thresh 0.50 --min-precision 0.95
```

### 7.3. Lệnh Telegram Bot hỗ trợ

| Lệnh | Chức năng |
|---|---|
| `/start` | Xem hướng dẫn và thông tin hệ thống |
| `/status` | Xem bộ đếm `mask`, `no_mask`, `unknown`, FPS và trạng thái camera |
| `/photo` | Gửi ảnh mới nhất đã vẽ bounding box từ worker dùng chung |
| `/flash` | Bật/tắt đèn Flash trên ESP32-CAM |

> **Lưu ý HTTP 409 Conflict:** Telegram API chỉ cho phép 1 poller duy nhất trên mỗi Bot Token. Không chạy song song polling Telegram trên cả firmware ESP32-CAM và Python server với cùng một token.

---

## 8. Kiểm Thử Tự Động (Unit Tests)

```powershell
# Chạy từng bộ test
.\venv\Scripts\python.exe -m unittest tests/test_metrics.py -v
.\venv\Scripts\python.exe -m unittest tests/test_calibration.py -v
.\venv\Scripts\python.exe -m unittest tests/test_reliability.py -v
.\venv\Scripts\python.exe -m unittest tests/test_telegram_service.py -v
.\venv\Scripts\python.exe -m unittest tests/test_s3_logger.py -v
.\venv\Scripts\python.exe -m unittest tests/test_performance_optimization.py -v
```
