# Multi-Person Face Mask Detection with YOLO

Dự án phát hiện khẩu trang nhiều người trong cùng một ảnh sử dụng YOLO Object Detection.
Ảnh được thu thập từ ESP32-CAM và inference chạy trên Laptop/Server.

---

## Classes

| ID | Class | Ý nghĩa |
|---:|-------|---------|
| 0 | `mask` | Khuôn mặt đeo khẩu trang đúng cách |
| 1 | `no_mask` | Khuôn mặt không đeo khẩu trang |
| 2 | `unknown` | Có khuôn mặt nhưng không đủ thông tin để kết luận |

---

## Current Phase

```
Project setup / Dataset preparation
```

---

## AI Framework

```
Ultralytics YOLO (object detection)
```

> **Lưu ý:** ESP32-CAM **KHÔNG** chạy YOLO trực tiếp.  
> ESP32-CAM chỉ chụp/stream ảnh → gửi đến Laptop/Server để inference.

---

## Kiến trúc hệ thống

```
ESP32-CAM
    │
    │ JPEG image/frame
    ▼
AI Inference trên Laptop/Server
    │
    ▼
YOLO Object Detector
    │
    ├── mask
    ├── no_mask
    └── unknown
    │
    ▼
Bounding Boxes + Labels + Confidence
    │
    ├── Web UI
    └── Telegram
```

---

## Cấu trúc project

```
mask_detection_yolo/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
├── results/
├── scripts/
│   └── check_project.py
├── src/
│
├── dataset.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Cài đặt môi trường

```powershell
# Tạo virtual environment
python -m venv venv

# Kích hoạt
venv\Scripts\activate

# Cài dependencies
pip install -r requirements.txt

# Kiểm tra Ultralytics
yolo checks
```

---

## Kiểm tra cấu trúc project

```powershell
python scripts/check_project.py
```

---

## Thu thập ảnh từ ESP32-CAM (Windows)

Script [capture_esp32_dataset.py](file:///d:/PTIT%20D23%20N23-091/YEAR%204/IOT_TRAIN%20AI/mask_detection_yolo/scripts/capture_esp32_dataset.py) gọi HTTP GET `/capture` để tải ảnh JPEG từ ESP32-CAM và tự động lưu vào `dataset_raw/own_photos/<session>/`.

> **Lưu ý:**
> - Ảnh được lưu vào thư mục thô (`dataset_raw`), **không** tự động đưa vào tập `dataset/images/(train|val|test)`.
> - Không sửa đổi firmware ESP32-CAM và không thay đổi weights model YOLO.

### Lệnh chạy trên Windows PowerShell / Command Prompt:

```powershell
# 1. Kích hoạt môi trường ảo (nếu chưa kích hoạt)
.\venv\Scripts\activate

# 2. Chụp mặc định (20 ảnh, mỗi ảnh cách 2s, tự sinh session theo thời gian)
python scripts/capture_esp32_dataset.py --ip 192.168.1.50

# 3. Tùy chỉnh session, số lượng ảnh, chu kỳ và timeout
python scripts/capture_esp32_dataset.py `
    --ip 192.168.1.50 `
    --count 50 `
    --interval 1.5 `
    --session session_mask_office `
    --timeout 10

# 4. Xem trợ giúp chi tiết
python scripts/capture_esp32_dataset.py --help
```

---

## Nhận diện trực tiếp từ ESP32-CAM (Desktop GUI)

Chạy nhận diện trực tiếp bằng cửa sổ OpenCV qua script [live_esp32_yolo.py](file:///d:/PTIT%20D23%20N23-091/YEAR%204/IOT_TRAIN%20AI/mask_detection_yolo/scripts/live_esp32_yolo.py):

```powershell
python scripts/live_esp32_yolo.py --ip 192.168.1.153
```

---

## Web Dashboard Giám Sát Khẩu Trang (FastAPI)

Script [esp32_dashboard.py](file:///d:/PTIT%20D23%20N23-091/YEAR%204/IOT_TRAIN%20AI/mask_detection_yolo/scripts/esp32_dashboard.py) khởi chạy giao diện Web Dashboard theo thời gian thực:
- **Streaming MJPEG**: Hiển thị luồng video thời gian thực từ ESP32-CAM với bounding box, nhãn và confidence.
- **Phân loại**: `mask` (xanh), `no_mask` (đỏ), `unknown` (dành cho detection có confidence dưới ngưỡng chấp nhận).
- **Thống kê thời gian thực**: Bộ đếm số lượng khuôn mặt theo từng nhóm, tổng số khuôn mặt và FPS pipeline thực tế.
- **Kiến trúc Single Pipeline Worker**: Chỉ duy nhất 1 vòng lặp capture + inference GPU dùng chung cho tất cả tab trình duyệt, tránh nghẽn camera hoặc quá tải model.
- **Tự động khôi phục**: Xử lý timeout và tự động thử kết nối lại khi ESP32-CAM rớt mạng.

### Lệnh chạy trên Windows PowerShell:

```powershell
# 1. Kích hoạt môi trường ảo
.\venv\Scripts\activate

# 2. Chạy Web Dashboard với tham số mặc định (IP 192.168.1.153, GPU RTX 5060, port 8000)
python scripts/esp32_dashboard.py

# 3. Chạy tùy chỉnh IP hoặc ngưỡng confidence
python scripts/esp32_dashboard.py `
    --ip 192.168.1.153 `
    --conf 0.20 `
    --accept 0.60 `
    --device 0 `
    --port 8000

# 4. Mở trình duyệt để giám sát
# URL: http://127.0.0.1:8000/
# Luồng stream MJPEG trực tiếp: http://127.0.0.1:8000/video
# API trạng thái JSON: http://127.0.0.1:8000/api/status
# API kích hoạt Flash: http://127.0.0.1:8000/api/flash
```

---

## Tích hợp Telegram Bot (YOLO11n + ESP32-CAM)

Dịch vụ [telegram_service.py](file:///d:/PTIT%20D23%20N23-091/YEAR%204/IOT_TRAIN%20AI/mask_detection_yolo/src/telegram_service.py) tích hợp trực tiếp vào vòng đời của Web Dashboard FastAPI [esp32_dashboard.py](file:///d:/PTIT%20D23%20N23-091/YEAR%204/IOT_TRAIN%20AI/mask_detection_yolo/scripts/esp32_dashboard.py):
- **Tái sử dụng pipeline**: Đọc trực tiếp frame và kết quả từ worker duy nhất trên GPU RTX 5060, **không** load model thứ hai.
- **Xác thực thời gian thực (Stale check)**: Nếu camera mất kết nối hoặc dữ liệu cũ hơn ngưỡng `TELEGRAM_STALE_THRESHOLD` (mặc định 5.0 giây), bot từ chối trả kết quả cũ và gửi cảnh báo rõ ràng.
- **Bảo mật**: Chỉ các Telegram Chat ID có trong danh sách cấu hình mới được phép điều khiển.

### 1. Cấu hình file `.env`

Sao chép file mẫu [.env.example](file:///d:/PTIT%20D23%20N23-091/YEAR%204/IOT_TRAIN%20AI/mask_detection_yolo/.env.example) thành `.env`:

```powershell
Copy-Item .env.example .env
```

Nội dung cấu hình trong `.env`:

```ini
# Token do @BotFather cấp
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

# Chat ID được phép gửi lệnh (phân tách bằng dấu phẩy)
TELEGRAM_ALLOWED_CHAT_IDS=123456789,987654321

# Địa chỉ IP ESP32-CAM
ESP32_IP=192.168.1.153

# Ngưỡng thời gian tối đa coi dữ liệu là thời gian thực (giây)
TELEGRAM_STALE_THRESHOLD=5.0
```

### 2. Các lệnh Telegram Bot hỗ trợ

| Lệnh | Chức năng | Phản hồi |
|---|---|---|
| `/start` | Hướng dẫn sử dụng | Danh sách lệnh, IP camera, thông tin hệ thống |
| `/status` | Thống kê số lượng | Số lượng `mask`, `no_mask`, `unknown`, `total`, FPS pipeline và độ trễ |
| `/photo` | Chụp ảnh kiểm tra | Gửi ảnh chụp mới nhất kèm bounding box và nhãn YOLO11n |
| `/flash` | Điều khiển đèn Flash | Gọi HTTP GET `/flash` trên ESP32-CAM và thông báo kết quả |

### 3. Lưu ý về Xung đột Telegram Polling (HTTP 409 Conflict)

> [!WARNING]
> **Xung đột khi dùng chung Bot Token:**
> Telegram Bot API chỉ cho phép **duy nhất 1 poller** chạy trên một Bot Token tại một thời điểm.
> Nếu firmware Arduino trên ESP32-CAM đang chạy vòng lặp `bot.getUpdates()` với cùng Token, server Python sẽ gặp lỗi `HTTP 409 Conflict: terminated by other getUpdates request`.
>
> **Hai giải pháp được khuyến nghị:**
> 1. **Thử nghiệm an toàn (không sửa firmware):** Tạo một Bot mới qua `@BotFather` (ví dụ `@MaskDetectorServerBot`) và gán Token này cho file `.env` của server Python. Khi đó cả hai bot hoạt động song song độc lập.
> 2. **Chuyển đổi hoàn toàn sang Python (khuyến nghị cho bản hoàn chỉnh):** Xóa bỏ hoặc comment vòng lặp Telegram trong firmware Arduino (để ESP32-CAM chỉ làm nhiệm vụ cung cấp camera `/capture` và `/flash`), giao toàn bộ việc xử lý Telegram cho Python server vì Python nắm giữ model AI và GPU.

### 4. Lệnh chạy và kiểm thử trên Windows PowerShell

```powershell
# Kích hoạt venv
.\venv\Scripts\activate

# Chạy Unit Tests (kiểm thử độc lập bằng Mock, không cần camera hay Telegram thật)
python -m unittest tests/test_telegram_service.py

# Chạy Web Dashboard kèm Telegram Bot (tự động nhận cấu hình từ .env)
python scripts/esp32_dashboard.py

# Hoặc truyền trực tiếp Token qua CLI
python scripts/esp32_dashboard.py `
    --telegram-token "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz" `
    --telegram-chat-id "123456789"
```

---

## Giám Sát Hiệu Năng & Đánh Giá Thực Nghiệm (Performance & Benchmark)

Hệ thống tích hợp module [src/metrics.py](file:///d:/PTIT%20D23%20N23-091/YEAR%204/IOT_TRAIN%20AI/mask_detection_yolo/src/metrics.py) đo lường chính xác từng phân đoạn xử lý bằng `time.perf_counter()` và đồng bộ `torch.cuda.synchronize()` trên GPU RTX 5060:

### 1. Phân định các thành phần độ trễ (Latency Breakdown)

- **HTTP Capture Latency (`capture_http_ms`):** Thời gian mạng request/response để truyền tải ảnh JPEG từ ESP32-CAM về máy chủ qua Wi-Fi/LAN (*không phải thời gian phơi sáng/màn trập phần cứng của cảm biến OV2640*).
- **JPEG Decoding Latency (`decode_ms`):** Thời gian CPU giải mã mảng byte JPEG thành ma trận ảnh BGR (`cv2.imdecode`).
- **YOLO11n Inference Latency (`inference_ms`):** Thời gian mạng nơ-ron YOLO suy luận trên GPU RTX 5060 (`device=0`, có đồng bộ CUDA).
- **Box Rendering Latency (`render_ms`):** Thời gian vẽ bounding box, nhãn phân loại và thanh trạng thái lên ảnh.
- **JPEG Encoding Latency (`encode_ms`):** Thời gian nén khung hình đã annotate thành JPEG phục vụ luồng stream MJPEG.
- **Server Processing Latency (`server_processing_ms`):** Tổng thời gian máy tính xử lý (`decode + inference + render + encode`).
- **Total Loop Latency (`total_loop_ms`):** Tổng chu kỳ thực thi của một khung hình (`capture_http + server_processing`).
- **Pipeline FPS:** Tốc độ khung hình thực tế (`1000 / total_loop_ms`).

### 2. Các Kịch Bản Đánh Giá Thực Nghiệm (Benchmark Scenarios)

| Mã | Tên kịch bản | Mô tả điều kiện kiểm thử |
|---|---|---|
| **S0** | Khung cảnh trống | Không có người trong vùng quan sát của ESP32-CAM |
| **S1** | 1 người không khẩu trang | 1 người đối diện camera, không đeo khẩu trang |
| **S2** | 1 người đeo khẩu trang | 1 người đeo khẩu trang đúng quy cách |
| **S3** | Nhiều người hỗn hợp | 2-3 người cùng xuất hiện với trạng thái khẩu trang khác nhau |
| **S4** | Góc nghiêng / Thiếu sáng | Khuôn mặt nhỏ ở xa, quay nghiêng góc hoặc điều kiện ánh sáng yếu |

### 3. Cách chạy Benchmark trên Windows PowerShell

Tất cả các phương thức đều **tái sử dụng pipeline worker duy nhất** đang chạy trong Web Dashboard, tránh tạo 2 tiến trình YOLO chạy đồng thời hoặc gây nghẽn kết nối camera:

#### Cách 1: Chạy qua dòng lệnh CLI (Khuyến nghị cho kiểm thử tự động)

```powershell
# 1. Khởi động Web Dashboard trước trong terminal thứ nhất
python scripts/esp32_dashboard.py

# 2. Mở terminal thứ hai, chạy benchmark kịch bản S1 trong 60 giây
python scripts/benchmark_performance.py --scenario S1 --duration 60

# Chạy benchmark kịch bản S3 trong 30 giây
python scripts/benchmark_performance.py --scenario S3 --duration 30
```

Kết quả chi tiết được xuất ra:
- File CSV từng frame: `results/performance/benchmark_<scenario>_<timestamp>.csv`
- Báo cáo Markdown tổng hợp: `results/performance/benchmark_<scenario>_<timestamp>_report.md`

#### Cách 2: Chạy trực tiếp từ giao diện Web Dashboard
- Truy cập `http://127.0.0.1:8000/`.
- Tại góc dưới giao diện, chọn kịch bản `S0` - `S4`, điền thời lượng (mặc định 60 giây) và nhấn **▶ Chạy Benchmark**.
- Hệ thống hiển thị thanh tiến độ, đếm ngược thời gian và cung cấp đường dẫn file kết quả ngay khi hoàn thành.

#### Cách 3: Kiểm tra API số liệu thời gian thực
- Thống kê độ trễ và FPS (Avg, P50, P95): `GET http://127.0.0.1:8000/api/metrics`
- Trạng thái phiên benchmark: `GET http://127.0.0.1:8000/api/benchmark/status`

### 4. Chạy kiểm thử tự động (Unit Tests)

```powershell
.\venv\Scripts\python.exe -m unittest discover tests
```

---

## BƯỚC 12: ĐỘ TIN CẬY & PHỤC HỒI KẾT NỐI (RELIABILITY & RECOVERY TESTING)

Hệ thống được trang bị cơ chế tự phục hồi, chống nghẽn mạng và phát hiện dữ liệu cũ (Stale Data Detection) nhằm đảm bảo hoạt động liên tục 24/7 trong môi trường Wi-Fi chập chờn:

### 1. Cơ chế Kỹ thuật

- **Exponential Backoff có trần:** Khi mất kết nối hoặc HTTP timeout, worker tự động lùi thời gian thử lại theo cấp số nhân ($0.5s \times 1.5^n$, tối đa $5.0s$) để tránh bão request làm treo ESP32-CAM khi board khởi động lại.
- **Tự động phục hồi & Đo lường Recovery Time:** Ngay khi nhận được khung hình JPEG hợp lệ đầu tiên sau sự cố, worker lập tức ghi nhận thời gian phục hồi ($t_{recovery}$), đưa camera về trạng thái `connected`, tính toán Recovery Time trung bình/P95 và tiếp tục suy luận YOLO mà không cần khởi động lại server.
- **Phát hiện Dữ liệu Cũ (Stale Data Detection):** Cả Web Dashboard và Telegram Bot kiểm tra tính tươi mới của frame ($t_{age} = \text{now} - t_{last\_capture}$). Nếu camera rớt mạng hoặc dữ liệu vượt ngưỡng $4.0s$, hệ thống hiển thị badge `⚠️ DỮ LIỆU CŨ`, bot từ chối cung cấp dữ liệu sai lệch.
- **Ngủ ngắt được (Interruptible Sleeps) & Dừng an toàn (Graceful Shutdown):** Sử dụng `threading.Event` thay cho `time.sleep()`, giải phóng `requests.Session` tức thì giúp server FastAPI, worker YOLO và Telegram poller dừng hoàn toàn trong $< 1.0$ giây khi nhận tín hiệu Ctrl+C.

### 2. Bảng Kịch bản Kiểm thử Độ tin cậy (R1–R6)

| Mã | Tên kịch bản | Mô tả điều kiện kiểm thử | Mục tiêu kỹ thuật |
|:---|:---|:---|:---|
| **R1** | Mất kết nối thoáng qua (1–2s) | Mất 1–2 gói tin hoặc rớt 1–2 frame rồi có lại ngay | Worker không sập, chuyển `reconnecting` và tự phục hồi tức thì |
| **R2** | Mất kết nối Wi-Fi tạm thời (10–15s) | Tắt Wi-Fi hoặc ESP32 khởi động lại | Tăng khoảng cách retry (backoff), tự phục hồi khi có sóng |
| **R3** | Mất nguồn / mất mạng kéo dài (>30s) | Rút nguồn camera hoặc đứt mạng lâu | Chạm trần backoff 5.0s, cảnh báo Stale trên Web và Telegram |
| **R4** | Mạng trễ cao / HTTP Timeout (>3s) | ESP32-CAM phản hồi chậm vượt socket timeout | Ngắt timeout an toàn, tăng `timeout_count`, không treo server |
| **R5** | Đồng thời nhiều Web Client (5–10 tabs) | Mở nhiều tab xem stream `/video` và gọi API liên tục | Duy trì duy nhất 1 worker YOLO, FPS không bị sụt giảm |
| **R6** | Graceful Shutdown khi đang có tải | Gửi tín hiệu dừng (Ctrl+C / SIGINT) khi đang stream | Dừng an toàn toàn bộ tiến trình trong thời gian $< 3.0$ giây |

### 3. Cách chạy Kiểm thử Độ tin cậy trên Windows PowerShell

#### Cách 1: Chạy qua dòng lệnh CLI (`scripts/test_reliability.py`)

```powershell
# Kiểm thử kịch bản R1 (Mất kết nối thoáng qua) trong 30 giây:
python scripts/test_reliability.py --scenario R1 --duration 30

# Kiểm thử kịch bản R5 (Đồng thời 8 client stream video và poll API):
python scripts/test_reliability.py --scenario R5 --duration 30 --clients 8

# Kiểm thử kịch bản R6 (Đo đạc tốc độ dừng an toàn Graceful Shutdown):
python scripts/test_reliability.py --scenario R6
```

Kết quả được xuất tự động:
- File CSV chi tiết: `results/reliability/reliability_<scenario>_<timestamp>.csv`
- Báo cáo Markdown: `results/reliability/reliability_<scenario>_<timestamp>_report.md`

#### Cách 2: Chạy trực tiếp từ Web Dashboard
- Truy cập `http://127.0.0.1:8000/`.
- Tại thẻ **🛡️ Độ Tin Cậy & Phục Hồi Kết Nối**, chọn kịch bản `R1`–`R6`, nhập thời lượng và nhấn **🛡️ Chạy Kiểm Thử R**.

#### Cách 3: Hướng dẫn Kiểm thử Thủ công trên phần cứng ESP32-CAM thật
1. **Kiểm thử R1 (Rớt frame):** Che nhẹ ăng-ten Wi-Fi trên ESP32-CAM trong 1-2 giây rồi thả ra; quan sát Web Dashboard hiển thị ảnh cảnh báo và hồi phục ngay lập tức.
2. **Kiểm thử R2/R3 (Mất nguồn & Wi-Fi):** Rút cáp nguồn micro-USB của ESP32-CAM; quan sát Web Dashboard chuyển sang trạng thái `RECONNECTING` kèm badge `⚠️ DỮ LIỆU CŨ`. Cắm nguồn lại; xác minh hệ thống tự kết nối lại trong 5–10 giây mà không cần bấm F5 trình duyệt.
3. **Kiểm thử Telegram từ chối dữ liệu cũ:** Khi camera đang rút nguồn, mở Telegram gửi lệnh `/status` hoặc `/photo`; bot sẽ trả về thông báo từ chối gửi dữ liệu cũ để tránh sai lệch thông tin giám sát.

---

---

## BƯỚC 13: HIỆU CHỈNH NGƯỠNG UNKNOWN & ĐÁNH GIÁ ĐỘ CHÍNH XÁC (UNKNOWN THRESHOLD CALIBRATION & REAL-CAMERA EVALUATION)

Hệ thống triển khai cơ chế từ chối kết luận dứt khoát (abstention mechanism) khi độ tin cậy của khuôn mặt thấp hơn ngưỡng chấp nhận, nhằm ngăn ngừa triệt để các quyết định sai lầm tự tin cao (False-confident decisions):

### 1. Nguyên Tắc & Quy Tắc Phân Loại

1. **Bảo tồn 2 Class Huấn Luyện:** YOLO11n giữ nguyên 2 class: `mask` (0) và `no_mask` (1). **Tuyệt đối không bổ sung class thứ 3 `unknown` vào quá trình huấn luyện model.**
2. **Cơ Chế Phân Loại Runtime:**
   - **Candidate Bounding Boxes:** Thu thập các ứng viên khuôn mặt tại ngưỡng sàn thấp `conf_floor` (mặc định $0.20$).
   - **Chấp Nhận (Accepted):** Nếu $conf \ge T_{\text{accept}}$, gán nhãn `mask` hoặc `no_mask`.
   - **Nghi Ngờ (Unknown):** Nếu $conf_{\text{floor}} \le conf < T_{\text{accept}}$, gán nhãn `unknown` (yêu cầu kiểm tra thủ công).
   - **Bỏ Sót (Missed Detection - FN):** Nếu YOLO hoàn toàn không phát hiện được bounding box nào (zero box), đây là $FN$. **Không phát hiện bounding box tuyệt đối KHÔNG đồng nghĩa với `unknown`.**
3. **Phân Cách Dữ Liệu Nghiêm Ngặt:**
   - Hiệu chỉnh ngưỡng trên tập **Validation** có Ground Truth (`dataset/images/val`).
   - Tập **TEST** (`dataset/images/test`) được giữ nguyên độc lập, không tham gia vào quá trình chọn ngưỡng.
   - Các session ảnh ESP32-CAM (`session01_mask`, `session01_nomask`) được tách biệt hoàn toàn để tránh data leakage.

### 2. Định Nghĩa & Mẫu Số Của Các Chỉ Số Đánh Giá

| Chỉ số (Metric) | Ký hiệu | Công thức toán học | Mẫu số (Denominator) | Định nghĩa & Ý nghĩa |
| :--- | :---: | :---: | :---: | :--- |
| **Total GT** | $N_{\text{gt}}$ | $\sum GT$ | N/A | Tổng số khuôn mặt thực tế trong tập dữ liệu. |
| **Total Candidates** | $N_{\text{cand}}$ | $\sum (conf \ge conf\_floor)$ | N/A | Tổng số bounding box do YOLO đề xuất. |
| **Total Accepted** | $N_{\text{acc}}$ | $TP_{\text{acc}} + WrongClass_{\text{acc}} + FP_{\text{acc}}$ | N/A | Tổng số dự đoán được chấp nhận. |
| **Accepted Precision** | $P_{\text{acc}}$ | $\frac{TP_{\text{acc}}}{N_{\text{acc}}}$ | $N_{\text{acc}}$ | Tỷ lệ nhận diện đúng trong các dự đoán chấp nhận. |
| **Coverage** | $Cov$ | $\frac{TP_{\text{acc}} + WrongClass_{\text{acc}}}{N_{\text{gt}}}$ | $N_{\text{gt}}$ | Tỷ lệ khuôn mặt thực tế nhận được quyết định dứt khoát. |
| **Correct Coverage** | $Cov_{\text{corr}}$ | $\frac{TP_{\text{acc}}}{N_{\text{gt}}}$ | $N_{\text{gt}}$ | Tỷ lệ khuôn mặt thực tế được nhận diện **đúng** và chấp nhận. |
| **Unknown Rate** | $UR$ | $\frac{N_{\text{cand}}(conf < T_{\text{accept}})}{N_{\text{cand}}}$ | $N_{\text{cand}}$ | Tỷ lệ ứng viên chuyển sang trạng thái nghi ngờ (`unknown`). |
| **False-confident Rate** | $FCR$ | $\frac{WrongClass_{\text{acc}} + FP_{\text{acc}}}{N_{\text{acc}}}$ | $N_{\text{acc}}$ | Tỷ lệ quyết định tự tin nhưng sai lầm ($1 - P_{\text{acc}}$). |
| **Missed Detection Rate** | $MDR$ | $\frac{FN}{N_{\text{gt}}}$ | $N_{\text{gt}}$ | Tỷ lệ khuôn mặt thực tế bị bỏ sót hoàn toàn (không có box). |

### 3. Kết Quả Khảo Sát & Lựa Chọn Ngưỡng Tối Ưu

Chạy quét dải ngưỡng từ $0.30$ đến $0.90$ trên tập Validation ($709$ GT faces, $819$ candidates):

| Ngưỡng ($T_{\text{accept}}$) | Accepted | Precision | Coverage | Unknown Rate | False-Conf | Cost |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.30` | 751 | 83.75% | 91.26% | 8.30% | 16.25% | 972.0 |
| `0.50` | 670 | 89.25% | 86.60% | 18.19% | 10.75% | 793.0 |
| `0.60` | 627 | 91.71% | 82.51% | 23.44% | 8.29% | 706.0 |
| `0.70` | 557 | 94.08% | 75.32% | 31.99% | 5.92% | 681.0 |
| **`0.75` (CHỌN)** | **503** | **95.83%** | **69.11%** | **38.58%** | **4.17%** | **665.0** |
| `0.80` | 438 | 96.58% | 60.79% | 46.52% | 3.42% | 700.0 |
| `0.85` | 266 | 98.12% | 37.24% | 67.52% | 1.88% | 797.0 |
| `0.90` | 50 | 100.00% | 7.05% | 93.89% | 0.00% | 973.0 |

- **Ngưỡng tối ưu được chọn:** $T_{\text{accept}} = 0.75$ với $conf_{\text{floor}} = 0.20$.
- **Lý do lựa chọn:** Đạt ràng buộc an toàn khắt khe (Accepted Precision $\ge 95\%$, thực tế đạt **95.83%**), đồng thời tối thiểu hóa chi phí rủi ro sai sót (Cost Function tối thiểu $665.0$), duy trì Coverage $69.11\%$ và tỷ lệ Unknown $38.58\%$.
- **Đánh giá độc lập trên ảnh ESP32-CAM có nhãn (60 ảnh thực tế):**
  - Accepted Precision: **100.00%**
  - Coverage: **68.18%**
  - Unknown Rate: **31.82%**
  - False-confident Rate: **0.00%**

### 4. Hướng Dẫn Thực Hiện Trên Windows PowerShell

#### Bước 4.1: Sử dụng Công cụ Gán Nhãn & Kiểm Tra Session ESP32-CAM
```powershell
# Tiền gán nhãn tự động cho session đeo khẩu trang (class 0 - mask):
.\venv\Scripts\python.exe scripts/label_assistant.py auto --session-dir dataset_raw/own_photos/session01_mask --default-class 0 --device 0

# Tiền gán nhãn tự động cho session không đeo khẩu trang (class 1 - no_mask):
.\venv\Scripts\python.exe scripts/label_assistant.py auto --session-dir dataset_raw/own_photos/session01_nomask --default-class 1 --device 0

# Kiểm tra tính hợp lệ và thống kê tọa độ bounding box:
.\venv\Scripts\python.exe scripts/label_assistant.py verify --session-dir dataset_raw/own_photos/session01_mask
.\venv\Scripts\python.exe scripts/label_assistant.py verify --session-dir dataset_raw/own_photos/session01_nomask

# Xuất ảnh mẫu trực quan để kiểm tra bằng mắt:
.\venv\Scripts\python.exe scripts/label_assistant.py visualize --session-dir dataset_raw/own_photos/session01_mask --limit 5
```

#### Bước 4.2: Chạy Quét & Hiệu Chỉnh Ngưỡng Calibration
```powershell
# Chạy hiệu chỉnh ngưỡng trên tập Validation và đánh giá độc lập ảnh ESP32-CAM:
.\venv\Scripts\python.exe scripts/calibrate_thresholds.py --device 0 --conf-floor 0.20 --iou-thresh 0.50 --min-precision 0.95
```

Kết quả sinh tự động:
- File cấu hình tập trung: `configs/thresholds.json`
- Bảng số liệu chi tiết: `results/calibration/threshold_sweep.csv`
- Biểu đồ hiệu chỉnh: `results/calibration/threshold_curves.png`
- Báo cáo Markdown chi tiết: `results/calibration/threshold_calibration_report.md`

#### Bước 4.3: Chạy Unit Tests Kiểm Thử Thuật Toán & Metrics
```powershell
.\venv\Scripts\python.exe -m unittest tests/test_calibration.py
```

---

### BƯỚC 14: TỐI ƯU HIỆU NĂNG & SO SÁNH BEFORE/AFTER BENCHMARK (GIAI ĐOẠN 9)

Giai đoạn 9 tập trung vào phân tích sâu nguyên nhân gốc rễ gây chậm, triển khai các kỹ thuật tối ưu hóa không xâm lấn tầng Python, tích hợp so sánh Before/After tự động và phân tích trade-off khi nâng cấp Firmware ESP32-CAM.

#### 1. Các Tối Ưu Đã Thực Hiện
- **HTTP Connection Pooling & Keep-Alive:** Tái sử dụng socket TCP thông qua `requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=4, max_retries=0)` kèm header `Connection: keep-alive`, loại bỏ chi phí 3-way handshake giữa các frame liên tiếp.
- **Tối ưu JPEG Encode phía Server:** Giảm nhẹ chất lượng nén từ 80 xuống 75 kèm tắt tối ưu hóa nén Huffman lặp (`IMWRITE_JPEG_OPTIMIZE=0`), tiết kiệm 1–2 ms CPU mỗi khung hình.
- **Zero-Queue Streaming:** Loại bỏ hàng đợi frame trung gian; dashboard và MJPEG stream luôn phản ánh trực tiếp khung hình vừa suy luận tức thì.
- **Phân Biệt Rõ Ràng Hai Chỉ Số FPS:**
  - **Throughput FPS:** $N_{\text{completed}} / T_{\text{elapsed}}$ (phản ánh thông lượng thực tế tới người dùng).
  - **Rolling Loop FPS:** $1000 / \bar{t}_{\text{loop}}$ (tốc độ chu kỳ trượt).
- **Hỗ trợ Replay Mode (`--replay-dir`):** Cho phép kiểm thử hiệu năng suy luận độc lập không phụ thuộc biến động sóng Wi-Fi.

#### 2. Hướng Dẫn Chạy Benchmark Đánh Giá Before / After Trên Windows PowerShell

```powershell
# 1. Khởi động Web Dashboard:
.\venv\Scripts\python.exe scripts/esp32_dashboard.py --ip 192.168.1.153 --device 0

# Hoặc khởi động ở chế độ Replay giả lập ảnh nếu camera đang offline:
.\venv\Scripts\python.exe scripts/esp32_dashboard.py --replay-dir dataset_raw/own_photos/session01_mask --device 0

# 2. Chạy benchmark kịch bản S1 (60 giây), tự động đối chiếu với Baseline gốc:
.\venv\Scripts\python.exe scripts/benchmark_performance.py --scenario S1 --duration 60

# 3. Chạy benchmark kịch bản S0 lặp lại 3 lần để tính trung bình và độ lệch chuẩn:
.\venv\Scripts\python.exe scripts/benchmark_performance.py --scenario S0 --duration 60 --repeat 3

# 4. Chạy Unit Tests kiểm thử tối ưu hiệu năng:
.\venv\Scripts\python.exe -m unittest tests/test_performance_optimization.py
```

#### 3. Báo Cáo Phân Tích & Đối Chiếu Số Liệu
- **Báo cáo phân tích nút thắt chi tiết:** `results/performance/bottleneck_analysis.md`
- **Báo cáo so sánh Before/After:** `results/performance/before_after_comparison.md`
- **File số liệu CSV:** `results/performance/before_after_comparison.csv`

---

## Quy trình phát triển

```
BƯỚC 1: Project setup
    ↓
BƯỚC 2: Thu thập & chuẩn hóa dataset (scripts/capture_esp32_dataset.py)
    ↓
BƯỚC 3: Annotate bounding box
    ↓
BƯỚC 4: Kiểm tra & chia Train/Val/Test (scripts/check_dataset.py)
    ↓
BƯỚC 5: Train YOLO (scripts/train_yolo.py)
    ↓
BƯỚC 6: Evaluate (antigravity_gpu_v1)
    ↓
BƯỚC 7: Inference nhiều người (scripts/infer_yolo.py)
    ↓
BƯỚC 8: Tích hợp ESP32-CAM (scripts/live_esp32_yolo.py)
    ↓
BƯỚC 9: Web UI + bounding boxes (scripts/esp32_dashboard.py)
    ↓
BƯỚC 10: Telegram Bot Integration (src/telegram_service.py)
    ↓
BƯỚC 11: Performance Monitoring & Experimental Benchmark (src/metrics.py)
    ↓
BƯỚC 12: Reliability & Recovery Testing (src/reliability.py, scripts/test_reliability.py)
    ↓
BƯỚC 13: Unknown Threshold Calibration & Real-Camera Evaluation (src/calibration.py, scripts/calibrate_thresholds.py)
    ↓
BƯỚC 14: Performance Optimization & Before/After Benchmark (src/metrics.py, scripts/benchmark_performance.py)
```







