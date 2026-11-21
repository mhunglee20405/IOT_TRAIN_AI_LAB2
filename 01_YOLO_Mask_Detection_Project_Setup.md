# BƯỚC 1 — KHỞI TẠO PROJECT YOLO MASK DETECTION

## 1. Mục tiêu

Thiết lập project AI riêng cho bài LAB ESP32-CAM, **không sử dụng Teachable Machine**.

Bài toán cần giải quyết là **object detection nhiều người trong cùng một ảnh**, trong đó mỗi khuôn mặt/người được phát hiện phải có:

- Bounding box riêng.
- Nhãn riêng.
- Confidence riêng.

Ba class cố định:

```text
0: mask
1: no_mask
2: unknown
```

Ví dụ kết quả mong muốn:

```text
Ảnh có 4 người
├── Person 1 → mask      96%
├── Person 2 → no_mask   91%
├── Person 3 → unknown   78%
└── Person 4 → mask      94%
```

> Lưu ý: Đây không còn là bài toán image classification một ảnh → một nhãn.  
> Project phải được xây dựng theo hướng **YOLO Object Detection**.

---

## 2. Yêu cầu cho Antigravity Code

Hãy đóng vai **Senior Computer Vision Engineer + Python Engineer** và thiết lập project theo đúng đặc tả trong tài liệu này.

### Nguyên tắc bắt buộc

1. Không dùng Teachable Machine.
2. Không xây classification model toàn ảnh.
3. Dùng **Ultralytics YOLO** cho object detection.
4. Giữ nguyên đúng ba class:
   - `mask`
   - `no_mask`
   - `unknown`
5. Không tự ý đổi tên class.
6. Không tự sinh dataset giả.
7. Không bắt đầu training khi dataset chưa được chuẩn bị và kiểm tra.
8. Không tích hợp ESP32-CAM ở bước này.
9. Không tích hợp Telegram ở bước này.
10. Mọi path phải dùng đường dẫn tương đối trong project nếu có thể.
11. Code Python phải dễ đọc, có comment ngắn gọn ở những phần quan trọng.
12. Project phải chạy được trên Windows.
13. Sử dụng UTF-8.
14. Không hard-code thông tin nhạy cảm như Wi-Fi password, API key hoặc Telegram token.

---

# 3. Kiến trúc tổng thể của hệ thống

Kiến trúc cuối cùng dự kiến:

```text
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

Ở giai đoạn hiện tại chỉ thực hiện:

```text
Khởi tạo project
        ↓
Chuẩn bị môi trường Python
        ↓
Chuẩn hóa cấu trúc dataset
        ↓
Chuẩn bị cấu hình YOLO
```

Chưa thực hiện:

```text
Thu thập ảnh
Annotate
Train
Evaluate
ESP32 integration
Telegram integration
```

---

# 4. Công nghệ sử dụng

## Ngôn ngữ

```text
Python 3.11
```

Khuyến nghị Python 3.11 để có độ tương thích tốt với các thư viện AI/CV.

## Framework chính

```text
Ultralytics YOLO
```

## Thư viện hỗ trợ

```text
opencv-python
matplotlib
PyYAML
```

Có thể bổ sung thư viện khác ở các bước sau nếu thực sự cần thiết.

---

# 5. Cấu trúc project bắt buộc

Tạo project theo cấu trúc:

```text
mask_detection_yolo/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│
├── results/
│
├── scripts/
│
├── src/
│
├── dataset.yaml
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

### Ý nghĩa

```text
dataset/
→ chứa dataset YOLO.

dataset/images/
→ chứa ảnh.

dataset/labels/
→ chứa annotation dạng YOLO .txt.

models/
→ lưu model/checkpoint sau này.

results/
→ lưu kết quả train/evaluation/inference.

scripts/
→ script hỗ trợ xử lý dataset.

src/
→ source code chính.

dataset.yaml
→ cấu hình dataset cho YOLO.

requirements.txt
→ dependency Python.

.gitignore
→ loại bỏ venv, cache, model lớn, dữ liệu tạm.

README.md
→ hướng dẫn project.
```

---

# 6. Tạo virtual environment

Trên Windows, trong thư mục project:

```powershell
python -m venv venv
```

Kích hoạt:

```powershell
venv\Scripts\activate
```

Sau khi kích hoạt thành công terminal phải xuất hiện dạng:

```text
(venv)
```

Nâng cấp pip:

```powershell
python -m pip install --upgrade pip
```

---

# 7. requirements.txt

Tạo:

```text
requirements.txt
```

với nội dung:

```text
ultralytics
opencv-python
matplotlib
PyYAML
```

Cài:

```powershell
pip install -r requirements.txt
```

---

# 8. Kiểm tra Ultralytics

Sau khi cài xong:

```powershell
yolo checks
```

Nếu command `yolo` không được nhận diện thì kiểm tra bằng Python:

```powershell
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

Kết quả mong muốn:

```text
Ultralytics OK
```

---

# 9. Định nghĩa ba class

Class ID phải cố định:

| ID | Class | Ý nghĩa |
|---:|---|---|
| 0 | `mask` | Khuôn mặt/người đeo khẩu trang đúng |
| 1 | `no_mask` | Khuôn mặt/người không đeo khẩu trang |
| 2 | `unknown` | Có khuôn mặt/người nhưng không đủ thông tin đáng tin cậy để kết luận mask/no_mask |

## Quy tắc `mask`

Dùng khi:

- Thấy được khuôn mặt đủ rõ.
- Khẩu trang che đúng vùng mũi và miệng.
- Có thể xác định chắc chắn người đang đeo khẩu trang.

## Quy tắc `no_mask`

Dùng khi:

- Thấy được khuôn mặt đủ rõ.
- Không có khẩu trang che mũi và miệng.
- Có thể xác định chắc chắn người không đeo khẩu trang.

## Quy tắc `unknown`

Dùng khi có đối tượng người/khuôn mặt nhưng không đủ rõ để phân loại đáng tin cậy, ví dụ:

- Mặt quá xa.
- Mặt quá mờ.
- Mặt bị che nhiều.
- Mặt chỉ xuất hiện một phần.
- Góc mặt quá nghiêng.
- Ánh sáng quá kém.
- Không đủ thông tin để kết luận `mask` hay `no_mask`.

### Không được gán `unknown` cho:

```text
bàn
ghế
điện thoại
tường
cửa
đồ vật
background
```

`unknown` vẫn phải là một đối tượng liên quan đến người/khuôn mặt.

---

# 10. Tạo dataset.yaml

Tạo file:

```text
dataset.yaml
```

với nội dung chính xác:

```yaml
path: dataset

train: images/train
val: images/val
test: images/test

names:
  0: mask
  1: no_mask
  2: unknown
```

Không thay đổi thứ tự ID.

---

# 11. Format annotation YOLO

Mỗi ảnh:

```text
abc.jpg
```

phải có file label cùng tên:

```text
abc.txt
```

Ví dụ:

```text
dataset/images/train/group001.jpg
dataset/labels/train/group001.txt
```

Mỗi object là một dòng:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Ví dụ:

```text
0 0.230000 0.410000 0.180000 0.300000
1 0.540000 0.430000 0.200000 0.320000
2 0.810000 0.450000 0.160000 0.280000
```

Các tọa độ YOLO phải được normalize:

```text
0.0 <= x_center <= 1.0
0.0 <= y_center <= 1.0
0.0 < width <= 1.0
0.0 < height <= 1.0
```

Một ảnh có nhiều người thì phải có nhiều dòng annotation.

Ví dụ:

```text
group001.jpg
```

có:

```text
Person 1 → mask
Person 2 → no_mask
Person 3 → unknown
```

thì:

```text
group001.txt
```

phải có 3 dòng.

---

# 12. Yêu cầu dataset dự kiến

Không cần có dataset ngay ở bước 1, nhưng cấu trúc phải sẵn sàng cho mục tiêu:

```text
Tổng ảnh dự kiến: 300–500+ ảnh
```

Ưu tiên ảnh có nhiều người.

Số object/instance tối thiểu mong muốn:

```text
mask:
300–500+ instances

no_mask:
300–500+ instances

unknown:
200–400+ instances
```

> Instance khác số ảnh.  
> Một ảnh có 5 người = có thể tạo 5 instances.

Dataset phải đa dạng:

```text
1 người
2 người
3 người
4–5 người
người đứng gần
người đứng xa
góc mặt khác nhau
nhiều background
nhiều điều kiện ánh sáng
nhiều màu/kiểu khẩu trang
nhiều khuôn mặt khác nhau
```

---

# 13. Quy tắc chia dataset

Dự kiến:

```text
Train:      70%
Validation: 20%
Test:       10%
```

Ví dụ nếu có 500 ảnh:

```text
350 train
100 val
50 test
```

### Quy tắc quan trọng

Không chia các frame gần giống nhau từ cùng một video/cùng một cảnh vào cả train và test.

Ví dụ không được:

```text
video_01_frame_001 → train
video_01_frame_002 → test
video_01_frame_003 → val
```

vì các frame quá giống nhau sẽ gây **data leakage**.

Nên chia theo:

```text
scene
video
session
person/group
```

trước, sau đó mới đưa vào train/val/test.

---

# 14. Model định hướng

Ở bước training sau này, ưu tiên model YOLO bản nhỏ/nano trước.

Mục tiêu:

```text
nhẹ
train nhanh
inference nhanh
đủ tốt cho LAB
```

Không train model ở bước hiện tại.

Không tự động download hoặc chạy training nếu người dùng chưa yêu cầu bước tiếp theo.

---

# 15. Tạo .gitignore

Tạo file:

```text
.gitignore
```

ít nhất gồm:

```gitignore
venv/
.venv/

__pycache__/
*.pyc

.idea/
.vscode/

runs/

*.pt
*.onnx
*.engine
*.tflite

results/*
!results/.gitkeep

models/*
!models/.gitkeep

.DS_Store
Thumbs.db
```

Không ignore:

```text
dataset.yaml
requirements.txt
README.md
source code
```

Việc commit dataset sẽ quyết định ở bước sau tùy kích thước dữ liệu.

---

# 16. README.md tối thiểu

README phải mô tả:

```text
Project:
Multi-Person Face Mask Detection with YOLO

Classes:
0 mask
1 no_mask
2 unknown

Current phase:
Project setup / dataset preparation

AI framework:
Ultralytics YOLO

Deployment target:
Laptop/server receives images from ESP32-CAM.

ESP32-CAM does NOT run YOLO directly.
```

README cũng phải có:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
yolo checks
```

---

# 17. Kiểm tra cấu trúc tự động

Tạo script:

```text
scripts/check_project.py
```

Script phải kiểm tra:

- `dataset.yaml` tồn tại.
- Các thư mục dataset tồn tại.
- `images/train`, `images/val`, `images/test` tồn tại.
- `labels/train`, `labels/val`, `labels/test` tồn tại.
- `models/` tồn tại.
- `results/` tồn tại.
- `src/` tồn tại.
- Ba class trong `dataset.yaml` đúng thứ tự:
  - `0: mask`
  - `1: no_mask`
  - `2: unknown`

Nếu đúng:

```text
[OK] Project structure is valid.
```

Nếu sai phải báo rõ thành phần nào thiếu/sai.

Chạy bằng:

```powershell
python scripts/check_project.py
```

---

# 18. Acceptance Criteria cho Bước 1

Bước 1 chỉ được xem là hoàn thành khi toàn bộ điều kiện sau đạt:

- [ ] Project `mask_detection_yolo` đã được tạo.
- [ ] Virtual environment tạo thành công.
- [ ] `requirements.txt` tồn tại.
- [ ] Ultralytics import thành công.
- [ ] `dataset/` đúng cấu trúc.
- [ ] `models/` tồn tại.
- [ ] `results/` tồn tại.
- [ ] `scripts/` tồn tại.
- [ ] `src/` tồn tại.
- [ ] `dataset.yaml` tồn tại.
- [ ] Class ID đúng:
  - [ ] `0: mask`
  - [ ] `1: no_mask`
  - [ ] `2: unknown`
- [ ] `.gitignore` tồn tại.
- [ ] `README.md` tồn tại.
- [ ] `scripts/check_project.py` chạy thành công.
- [ ] Chưa train model.
- [ ] Chưa thêm Teachable Machine.
- [ ] Chưa tích hợp ESP32.
- [ ] Chưa tích hợp Telegram.

---

# 19. Output Antigravity Code cần trả về

Sau khi hoàn tất, hãy cung cấp:

1. Cây thư mục project thực tế.
2. Danh sách file đã tạo.
3. Nội dung `dataset.yaml`.
4. Nội dung `requirements.txt`.
5. Kết quả chạy:

```powershell
python scripts/check_project.py
```

6. Kết quả kiểm tra import Ultralytics.
7. Các lỗi hoặc cảnh báo nếu có.
8. Không tự chuyển sang bước thu thập/annotate/train dataset cho đến khi Bước 1 được xác nhận hoàn thành.

---

# 20. Bước tiếp theo sau khi hoàn thành

Sau Bước 1, quy trình tiếp tục:

```text
BƯỚC 2
Thu thập và chuẩn hóa dataset
        ↓
BƯỚC 3
Annotate bounding box
        ↓
BƯỚC 4
Kiểm tra và chia Train/Val/Test
        ↓
BƯỚC 5
Train YOLO
        ↓
BƯỚC 6
Evaluate
        ↓
BƯỚC 7
Inference nhiều người
        ↓
BƯỚC 8
Tích hợp ESP32-CAM
        ↓
BƯỚC 9
Web UI + bounding boxes
        ↓
BƯỚC 10
Telegram
```

**Không thực hiện các bước trên trong task hiện tại.**

---

# 21. Tóm tắt nhiệm vụ hiện tại

```text
TASK:
Chuẩn bị nền tảng project YOLO Multi-Person Mask Detection.

INPUT:
Chưa có dataset.

OUTPUT:
Một project Python/Ultralytics chuẩn, sẵn sàng cho bước thu thập
và annotate dataset.

CLASSES:
0 mask
1 no_mask
2 unknown

DO NOT:
- Train model
- Dùng Teachable Machine
- Tích hợp ESP32-CAM
- Tích hợp Telegram
- Tạo dataset giả
```
