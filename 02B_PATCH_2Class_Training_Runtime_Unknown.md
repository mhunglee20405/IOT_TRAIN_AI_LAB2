# PATCH — CHUYỂN PROJECT YOLO SANG 2 TRAINING CLASSES + RUNTIME UNKNOWN

## 1. Mục tiêu patch

Cập nhật project `mask_detection_yolo` theo kiến trúc mới:

```text
TRAINING CLASSES
0 = mask
1 = no_mask
```

Không train class `unknown`.

`unknown` chỉ được sinh ra ở giai đoạn inference/runtime khi độ tin cậy của model thấp hơn ngưỡng cấu hình.

Kiến trúc mục tiêu:

```text
ESP32-CAM / image
        ↓
YOLO detector
        ↓
mask / no_mask + confidence
        ↓
confidence >= threshold ?
    ┌───────────────┴───────────────┐
   YES                              NO
    ↓                                ↓
mask / no_mask                    unknown
```

Ví dụ:

```text
mask     0.94 -> mask
no_mask  0.88 -> no_mask
mask     0.42 -> unknown
```

---

# 2. Bối cảnh dataset public hiện tại

Dataset Roboflow đã được tải về tại:

```text
dataset_raw/
└── public_sources/
    └── roboflow_mask_wearing/
        ├── train/
        ├── valid/
        ├── test/
        ├── data.yaml
        └── README.roboflow
```

`data.yaml` gốc của Roboflow:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 2
names: ['mask', 'no-mask']
```

Mapping gốc:

```text
0 = mask
1 = no-mask
```

Điều này tương thích với mapping mới của project.

Chỉ cần chuẩn hóa tên hiển thị:

```text
no-mask -> no_mask
```

KHÔNG đổi class ID trong annotation `.txt`.

---

# 3. Nguyên tắc bắt buộc

1. Không sửa nội dung dataset raw gốc.
2. Không sửa `data.yaml` bên trong `dataset_raw/...`.
3. Không đổi class ID của annotation.
4. Không train class `unknown`.
5. Không tạo annotation `unknown`.
6. Không tự sinh dữ liệu giả.
7. Không train model trong task này.
8. Không xóa dataset public gốc.
9. Dataset để train phải nằm trong thư mục canonical `dataset/`.
10. Mọi thao tác import phải có thể chạy lại an toàn.
11. Không overwrite file khác nội dung mà không cảnh báo.
12. Không đưa ZIP/raw dataset vào canonical training dataset.
13. Giữ nguyên mục tiêu object detection theo vùng face/head.

---

# 4. Cấu trúc project sau patch

Project phải có dạng:

```text
mask_detection_yolo/
│
├── dataset_raw/
│   ├── public_sources/
│   │   └── roboflow_mask_wearing/
│   │       ├── train/
│   │       │   ├── images/
│   │       │   └── labels/
│   │       ├── valid/
│   │       │   ├── images/
│   │       │   └── labels/
│   │       ├── test/
│   │       │   ├── images/
│   │       │   └── labels/
│   │       ├── data.yaml
│   │       └── README.roboflow
│   │
│   ├── own_photos/
│   ├── extracted_frames/
│   └── negatives/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── SOURCES.md
│
├── scripts/
│   ├── check_project.py
│   ├── check_dataset.py
│   ├── visualize_labels.py
│   ├── extract_frames.py
│   └── import_roboflow_dataset.py
│
├── src/
├── models/
├── results/
├── dataset.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 5. Sửa `dataset.yaml` chính

Thay cấu hình cũ 3 class bằng đúng nội dung sau:

```yaml
path: dataset

train: images/train
val: images/val
test: images/test

names:
  0: mask
  1: no_mask
```

Không còn:

```yaml
2: unknown
```

---

# 6. Runtime class `unknown`

Tạo file cấu hình hoặc constant trong code inference sau này:

```python
UNKNOWN_THRESHOLD = 0.60
```

Logic runtime dự kiến:

```python
if confidence < UNKNOWN_THRESHOLD:
    final_label = "unknown"
else:
    final_label = predicted_label
```

Lưu ý:

- Chưa implement inference trong task hiện tại nếu chưa có module inference.
- Chỉ cập nhật README/spec để thống nhất kiến trúc.
- Threshold `0.60` chỉ là giá trị mặc định ban đầu.
- Sau khi evaluate model phải tune threshold dựa trên validation/test.

---

# 7. Tạo script `scripts/import_roboflow_dataset.py`

Mục tiêu:

Import dữ liệu Roboflow raw sang canonical dataset mà không sửa dữ liệu gốc.

## Source mặc định

```text
dataset_raw/public_sources/roboflow_mask_wearing/
```

## Destination mặc định

```text
dataset/
```

## Mapping split

Roboflow:

```text
train -> train
valid -> val
test  -> test
```

Canonical:

```text
dataset/images/train
dataset/images/val
dataset/images/test

dataset/labels/train
dataset/labels/val
dataset/labels/test
```

---

# 8. Quy tắc import

Script phải:

1. Kiểm tra source tồn tại.
2. Kiểm tra:
   ```text
   train/images
   train/labels
   valid/images
   valid/labels
   test/images
   test/labels
   ```
3. Không phụ thuộc vào path trong `data.yaml` raw vì file raw có thể đã được di chuyển.
4. Đọc `data.yaml` chỉ để kiểm tra class metadata.
5. Xác nhận:
   ```text
   nc = 2
   names[0] = mask
   names[1] = no-mask
   ```
   hoặc equivalent dictionary form.
6. Nếu class mapping khác thì dừng và báo lỗi.
7. Copy ảnh + label sang canonical dataset.
8. Giữ nguyên numeric class ID.
9. Không chỉnh nội dung label nếu mapping đã đúng.
10. Prefix tên file để tránh collision.

Prefix bắt buộc:

```text
rfmw_
```

Ví dụ source:

```text
train/images/abc.jpg
train/labels/abc.txt
```

Destination:

```text
dataset/images/train/rfmw_abc.jpg
dataset/labels/train/rfmw_abc.txt
```

---

# 9. Xử lý collision

Nếu file đích đã tồn tại:

- Nếu file giống hệt -> skip và log `[SKIP]`.
- Nếu tên giống nhưng nội dung khác -> dừng hoặc tạo lỗi rõ ràng.
- Không silently overwrite.

Ví dụ:

```text
[SKIP] rfmw_abc.jpg already imported.
[ERROR] Filename collision with different content: rfmw_xyz.jpg
```

---

# 10. Dry-run mode

Script phải hỗ trợ:

```powershell
python scripts/import_roboflow_dataset.py --dry-run
```

Dry-run chỉ:

- kiểm tra source
- kiểm tra class mapping
- đếm file
- dự kiến copy
- phát hiện collision

Không copy file thật.

Import thật:

```powershell
python scripts/import_roboflow_dataset.py
```

---

# 11. Output import mong muốn

Ví dụ:

```text
Source:
dataset_raw/public_sources/roboflow_mask_wearing

Class mapping:
0 -> mask
1 -> no-mask

Import mapping:
train -> train
valid -> val
test  -> test

[TRAIN]
Images found: 1354
Labels found: 1354
Copied: 1354

[VAL]
Images found: 386
Labels found: 386
Copied: 386

[TEST]
Images found: 194
Labels found: 194
Copied: 194

[OK] Roboflow dataset imported successfully.
```

Số lượng thực tế phải lấy từ filesystem, không hard-code.

---

# 12. Cập nhật `scripts/check_project.py`

Script phải kiểm tra class config mới:

```text
0 = mask
1 = no_mask
```

Không còn check:

```text
2 = unknown
```

Nếu `dataset.yaml` vẫn có `unknown` thì báo:

```text
[ERROR] Training dataset must contain exactly 2 classes: mask, no_mask.
```

Ngoài ra kiểm tra:

```text
dataset_raw/
dataset_raw/public_sources/
dataset/
scripts/
src/
models/
results/
dataset.yaml
requirements.txt
README.md
```

---

# 13. Cập nhật `scripts/check_dataset.py`

Class hợp lệ duy nhất:

```text
0
1
```

Mapping:

```python
CLASS_NAMES = {
    0: "mask",
    1: "no_mask",
}
```

Nếu gặp class `2` hoặc class khác:

```text
[ERROR] Invalid class id: 2
```

Script phải thống kê:

```text
TRAIN
Images
Labels
mask instances
no_mask instances
empty labels

VAL
...

TEST
...
```

Ngoài ra kiểm tra:

- image không có label
- label không có image
- label syntax sai
- không đủ 5 field
- class ID sai
- x/y/w/h ngoài range
- width <= 0
- height <= 0
- file label rỗng hợp lệ đối với negative image
- duplicate filename
- số lượng class imbalance

---

# 14. Negative images

Cho phép ảnh không có đối tượng.

Ví dụ:

```text
dataset/images/train/rfmw_empty_room_001.jpg
dataset/labels/train/rfmw_empty_room_001.txt
```

File `.txt` có thể rỗng.

`check_dataset.py` phải coi đây là hợp lệ và thống kê:

```text
Empty labels / negative images: N
```

Không báo lỗi chỉ vì label file rỗng.

---

# 15. Cập nhật `scripts/visualize_labels.py`

Class map mới:

```python
CLASS_NAMES = {
    0: "mask",
    1: "no_mask",
}
```

Không vẽ class `unknown`.

Script phải:

- đọc ảnh
- đọc label YOLO
- convert normalized bbox -> pixel bbox
- vẽ box
- vẽ class name
- không chỉnh ảnh gốc
- lưu preview vào:
  ```text
  results/dataset_preview/
  ```

Hỗ trợ:

```powershell
python scripts/visualize_labels.py --split train --count 20
```

Nếu không có argument thì dùng mặc định hợp lý.

---

# 16. Cập nhật README

README phải giải thích rõ:

## Training classes

```text
0 mask
1 no_mask
```

## Runtime output

```text
mask
no_mask
unknown
```

`unknown` không phải class train.

Nó được sinh ra khi:

```text
detection confidence < threshold
```

README phải ghi rõ lý do:

```text
Unknown không có một visual pattern ổn định.
Các tình huống mờ, quá xa, bị che, góc mặt xấu...
không nên bị ép thành một training class duy nhất.
```

---

# 17. Cập nhật `dataset/SOURCES.md`

Thêm nguồn Roboflow hiện tại.

Ví dụ:

```markdown
## Roboflow Mask Wearing

- Type: Public object detection dataset
- Local raw path:
  `dataset_raw/public_sources/roboflow_mask_wearing`
- Original classes:
  - 0: mask
  - 1: no-mask
- Canonical classes:
  - 0: mask
  - 1: no_mask
- License: Public Domain
- Usage:
  Used as public base data for YOLO face-mask detection.
- Notes:
  Raw source is kept unchanged.
```

Không ghi số lượng ảnh nếu script chưa đếm thật.

Sau import, có thể bổ sung statistics thực tế.

---

# 18. Không merge bằng tay

Không yêu cầu user copy thủ công:

```text
train -> dataset/images/train
```

Phải dùng:

```text
scripts/import_roboflow_dataset.py
```

Lý do:

- tránh sai split
- tránh filename collision
- có log
- có validation
- chạy lại được
- dễ audit

---

# 19. Kiểm tra source Roboflow trước import

Script phải kiểm tra file:

```text
dataset_raw/public_sources/roboflow_mask_wearing/data.yaml
```

Metadata hiện tại dự kiến:

```yaml
nc: 2
names: ['mask', 'no-mask']
```

Nếu metadata không đúng thì:

```text
[ERROR] Unexpected Roboflow class mapping.
Expected:
0 = mask
1 = no-mask
```

và dừng.

---

# 20. Quy trình chạy sau patch

Sau khi Antigravity sửa code xong, chạy theo thứ tự:

```powershell
python scripts/check_project.py
```

Sau đó:

```powershell
python scripts/import_roboflow_dataset.py --dry-run
```

Nếu không lỗi:

```powershell
python scripts/import_roboflow_dataset.py
```

Sau đó:

```powershell
python scripts/check_dataset.py
```

Cuối cùng:

```powershell
python scripts/visualize_labels.py --split train --count 20
```

---

# 21. Chưa train model

Sau khi hoàn thành task này:

```text
KHÔNG chạy yolo train
KHÔNG tải pretrained weights để bắt đầu training
KHÔNG tạo run mới
```

Chỉ dừng ở:

```text
Raw public dataset
        ↓
validated import
        ↓
canonical dataset
        ↓
dataset QA
        ↓
visual inspection
```

---

# 22. Acceptance Criteria

Task chỉ hoàn thành khi:

- [ ] `dataset.yaml` chỉ có 2 class.
- [ ] `0 = mask`.
- [ ] `1 = no_mask`.
- [ ] Không còn `2 = unknown` trong training config.
- [ ] `unknown` được mô tả là runtime fallback.
- [ ] Raw Roboflow dataset không bị chỉnh sửa.
- [ ] Có `scripts/import_roboflow_dataset.py`.
- [ ] Import script có `--dry-run`.
- [ ] Import map `valid -> val`.
- [ ] Import dùng prefix `rfmw_`.
- [ ] Không overwrite collision silently.
- [ ] `check_project.py` đã cập nhật 2 class.
- [ ] `check_dataset.py` đã cập nhật 2 class.
- [ ] `visualize_labels.py` đã cập nhật 2 class.
- [ ] Empty YOLO labels được cho phép.
- [ ] `dataset/SOURCES.md` có nguồn Roboflow.
- [ ] README mô tả 2-class training + runtime unknown.
- [ ] `python scripts/check_project.py` pass.
- [ ] `import_roboflow_dataset.py --dry-run` pass.
- [ ] Import thật hoàn thành hoặc báo rõ lỗi.
- [ ] `python scripts/check_dataset.py` pass.
- [ ] Có preview annotation để kiểm tra bằng mắt.
- [ ] Chưa train model.

---

# 23. Output Antigravity phải trả về

Sau khi hoàn thành, trả:

1. Danh sách file đã sửa.
2. Danh sách file mới tạo.
3. Nội dung `dataset.yaml`.
4. Mapping class cuối:
   ```text
   0 mask
   1 no_mask
   ```
5. Kết quả:
   ```powershell
   python scripts/check_project.py
   ```
6. Kết quả:
   ```powershell
   python scripts/import_roboflow_dataset.py --dry-run
   ```
7. Kết quả import thật nếu đã chạy.
8. Kết quả:
   ```powershell
   python scripts/check_dataset.py
   ```
9. Thống kê số ảnh + instance theo split.
10. Đường dẫn preview annotation.
11. Các warning còn tồn tại.
12. Xác nhận chưa bắt đầu training.

---

# 24. Tóm tắt kiến trúc cuối sau patch

```text
PUBLIC RAW DATA
Roboflow:
0 mask
1 no-mask
      ↓
import script
      ↓
CANONICAL DATASET
0 mask
1 no_mask
      ↓
YOLO TRAINING (step sau)
      ↓
prediction + confidence
      ↓
threshold
   ┌──┴─────────────┐
high confidence   low confidence
     ↓                ↓
mask/no_mask       unknown
```

---

# 25. Không thực hiện ngoài scope

Task hiện tại KHÔNG bao gồm:

```text
YOLO training
hyperparameter tuning
model evaluation
ESP32-CAM integration
web inference
Telegram integration
ONNX export
TensorFlow.js export
TFLite export
```

Chỉ hoàn thiện dataset pipeline theo kiến trúc 2 training classes + runtime unknown.
