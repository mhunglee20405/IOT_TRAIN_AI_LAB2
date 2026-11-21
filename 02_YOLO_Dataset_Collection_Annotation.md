# BƯỚC 2 — THU THẬP, CHUẨN HÓA VÀ GÁN NHÃN DATASET YOLO

## 1. Mục tiêu

Chuẩn bị dataset object detection cho hệ thống nhận diện nhiều người trong cùng một ảnh.

Mỗi khuôn mặt/người được phát hiện phải có:

- 1 bounding box riêng.
- 1 class riêng.
- 1 confidence riêng sau khi inference.

Ba class cố định:

```text
0: mask
1: no_mask
2: unknown
```

Bước này chưa train model.

---

# 2. Quy tắc quan trọng nhất

Bài toán là:

```text
MULTI-PERSON FACE MASK OBJECT DETECTION
```

Không phải:

```text
IMAGE CLASSIFICATION
```

Vì vậy annotation phải theo từng khuôn mặt.

Ví dụ:

```text
Ảnh group_001.jpg

Person 1 -> mask
Person 2 -> no_mask
Person 3 -> unknown
Person 4 -> mask
```

Ảnh này phải có 4 bounding box.

---

# 3. Đối tượng cần vẽ bounding box

Bounding box nên bao quanh:

```text
KHUÔN MẶT / VÙNG ĐẦU CÓ THÔNG TIN ĐỂ XÁC ĐỊNH KHẨU TRANG
```

Không vẽ box quanh toàn bộ cơ thể.

Ví dụ đúng:

```text
      mask
   +---------+
   |  FACE   |
   |  MASK   |
   +---------+
      BODY
```

Ví dụ không nên:

```text
+-------------------+
|       HEAD        |
|                   |
|       BODY        |
|                   |
|       LEGS        |
+-------------------+
```

Lý do:

```text
mask / no_mask
```

là thông tin nằm ở vùng khuôn mặt.

---

# 4. Định nghĩa class

## Class 0 — mask

Gán `mask` khi:

- Có khuôn mặt/người rõ ràng.
- Khẩu trang che mũi.
- Khẩu trang che miệng.
- Trạng thái đeo khẩu trang có thể xác định chắc chắn.

Ví dụ:

```text
Khẩu trang y tế
Khẩu trang vải
Khẩu trang màu khác nhau
```

đều có thể là `mask`.

---

## Class 1 — no_mask

Gán `no_mask` khi:

- Có khuôn mặt rõ.
- Không có khẩu trang che mũi và miệng.
- Có thể xác định chắc chắn người không đeo khẩu trang.

---

## Class 2 — unknown

Chỉ gán `unknown` khi:

> Có khuôn mặt / vùng đầu có thể định vị được nhưng trạng thái mask/no_mask không thể xác định đáng tin cậy.

Ví dụ:

```text
mặt bị che bởi tay
mặt bị che bởi vật thể
ảnh mờ
ánh sáng rất kém
mặt quay nghiêng mạnh
chỉ thấy một phần mặt
vùng miệng/mũi bị khuất
```

Không gán `unknown` cho:

```text
bàn
ghế
điện thoại
tường
background
động vật
đồ vật
```

---

# 5. Trường hợp mặt quá nhỏ

Không nên ép mọi khuôn mặt cực nhỏ thành `unknown`.

Nếu khuôn mặt nhỏ đến mức:

```text
không thể vẽ box ổn định
và
không thể xác định vùng mặt
```

thì nên bỏ qua object đó.

Nếu vẫn có thể định vị rõ khuôn mặt nhưng không đủ thông tin để xác định mask/no_mask:

```text
-> unknown
```

Mục tiêu là tránh đưa annotation nhiễu vào dataset.

---

# 6. Nguồn dữ liệu

Dataset nên kết hợp nhiều nguồn.

## Nguồn A — Tự chụp

Ưu tiên tự chụp bằng:

```text
điện thoại
webcam
ESP32-CAM
```

Nên có:

```text
1 người
2 người
3 người
4-5 người
nhiều khoảng cách
nhiều góc camera
nhiều background
nhiều điều kiện ánh sáng
```

Không nên chỉ chụp selfie một người.

---

## Nguồn B — Video tự quay

Có thể quay video nhiều người rồi trích frame.

Ví dụ:

```text
video_001.mp4
video_002.mp4
```

Không lấy mọi frame liên tiếp.

Ví dụ video 30 FPS:

```text
frame 1
frame 2
frame 3
...
```

quá giống nhau.

Nên lấy cách quãng:

```text
1 frame / 1-2 giây
```

hoặc dùng script lọc frame.

---

## Nguồn C — Dataset public

Có thể bổ sung dataset public nếu:

- Có quyền sử dụng phù hợp.
- Có ảnh khuôn mặt đeo và không đeo khẩu trang.
- Có nhiều người trong ảnh.
- Không phụ thuộc hoàn toàn vào một dataset duy nhất.

Nếu dùng nguồn public, phải ghi lại:

```text
tên dataset
nguồn
license
số ảnh sử dụng
```

trong:

```text
dataset/SOURCES.md
```

---

# 7. Mục tiêu số lượng dữ liệu

Mục tiêu ban đầu:

```text
500-1000 ảnh
```

Không cần đạt ngay trong lần đầu.

Mục tiêu instance:

```text
mask:
500+ instances

no_mask:
500+ instances

unknown:
300+ instances
```

Tốt hơn nếu đạt:

```text
1000+ mask
1000+ no_mask
500+ unknown
```

nhưng chất lượng và độ đa dạng quan trọng hơn số lượng.

---

# 8. Cân bằng dữ liệu

Không để dataset kiểu:

```text
mask      2000
no_mask    200
unknown     50
```

Nên cố gắng giữ tương đối cân bằng.

Ví dụ:

```text
mask       800
no_mask    750
unknown    450
```

vẫn chấp nhận được.

`unknown` có thể ít hơn vì đây là class đặc biệt.

---

# 9. Đa dạng dataset

Dataset phải có thay đổi về:

```text
người
giới tính
độ tuổi
góc mặt
khoảng cách
ánh sáng
background
màu khẩu trang
loại khẩu trang
camera angle
độ phân giải
số người trong ảnh
```

Không dùng gần như toàn bộ ảnh của một người.

Mục tiêu:

```text
model học đặc trưng mask/no_mask
```

không phải:

```text
model học khuôn mặt của một vài người cụ thể
```

---

# 10. Quy tắc chụp ảnh multi-person

Cần chủ động tạo các scene:

```text
Scene A:
2 người đều mask

Scene B:
2 người đều no_mask

Scene C:
1 mask + 1 no_mask

Scene D:
2 mask + 1 no_mask

Scene E:
mask + no_mask + unknown

Scene F:
4-5 người với nhiều trạng thái
```

Các người nên:

```text
đứng gần
đứng xa
đứng lệch nhau
quay nhẹ sang trái/phải
ở các vị trí khác nhau trong frame
```

---

# 11. Quy tắc tránh data leakage

Đây là yêu cầu bắt buộc.

Không được chia các ảnh gần như giống nhau từ cùng scene/video vào cả train và test.

Sai:

```text
video01_frame001 -> train
video01_frame002 -> val
video01_frame003 -> test
```

Đúng hơn:

```text
video01 -> train
video02 -> val
video03 -> test
```

Hoặc:

```text
session_A -> train
session_B -> val
session_C -> test
```

---

# 12. Tool annotation

Khuyến nghị một trong các công cụ:

```text
CVAT
Roboflow Annotate
Label Studio
LabelImg
```

Ưu tiên cho người mới:

```text
Roboflow Annotate
```

hoặc:

```text
CVAT
```

Yêu cầu export cuối cùng:

```text
YOLO format
```

---

# 13. Quy tắc annotation

Mỗi khuôn mặt:

```text
1 bounding box
```

Bounding box:

- Bao sát vùng đầu/khuôn mặt.
- Không quá rộng.
- Không cắt mất phần quan trọng của mặt nếu ảnh có đủ dữ liệu.
- Không bao luôn phần lớn cơ thể.
- Không để box trùng sai người.
- Không bỏ sót các khuôn mặt đủ rõ.

Ví dụ:

```text
IMAGE
+--------------------------------------+
|                                      |
|  +------+        +------+            |
|  | mask |        |noMask|            |
|  | FACE |        | FACE |            |
|  +------+        +------+            |
|                                      |
|              +---------+             |
|              | unknown |             |
|              |  FACE   |             |
|              +---------+             |
|                                      |
+--------------------------------------+
```

---

# 14. Quy tắc đối với occlusion

Ví dụ người bị che bởi người khác.

Nếu vẫn thấy đủ mặt và trạng thái khẩu trang:

```text
-> mask hoặc no_mask
```

Nếu thấy mặt nhưng phần quyết định trạng thái bị che:

```text
-> unknown
```

Nếu gần như không thể xác định vị trí khuôn mặt:

```text
-> bỏ qua
```

---

# 15. Quy tắc đối với góc nghiêng

Nếu người quay nghiêng nhưng vẫn thấy rõ khẩu trang:

```text
-> mask
```

Nếu quay nghiêng nhưng rõ ràng không đeo:

```text
-> no_mask
```

Nếu góc quá nghiêng khiến vùng mũi/miệng không đủ thông tin:

```text
-> unknown
```

---

# 16. Quy tắc đối với khẩu trang đeo sai

Trong project hiện tại chỉ có:

```text
mask
no_mask
unknown
```

Không có class:

```text
incorrect_mask
```

Vì vậy cần thống nhất quy tắc.

Khuyến nghị:

```text
Khẩu trang không che mũi hoặc không che miệng
-> no_mask
```

Lý do:

Mục tiêu `mask` nên biểu diễn:

```text
đeo khẩu trang đúng và có hiệu lực che mũi + miệng
```

Không dùng `unknown` cho trường hợp đeo sai nếu trạng thái nhìn thấy rõ.

---

# 17. Cấu trúc dataset sau annotation

Kết quả cuối cùng phải dạng:

```text
dataset/
├── images/
│   ├── train/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   │
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    │   ├── image001.txt
    │   ├── image002.txt
    │   └── ...
    │
    ├── val/
    └── test/
```

Tên phải khớp:

```text
image001.jpg
image001.txt
```

---

# 18. Format YOLO label

Mỗi dòng:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Ví dụ:

```text
0 0.245312 0.351852 0.184375 0.296296
1 0.581250 0.375000 0.206250 0.318519
2 0.817188 0.402778 0.150000 0.277778
```

Mọi tọa độ phải:

```text
0 <= value <= 1
```

với width/height > 0.

---

# 19. Tạo thư mục raw dataset

Bổ sung vào project:

```text
dataset_raw/
├── own_photos/
├── extracted_frames/
└── public_sources/
```

`dataset_raw/` chứa dữ liệu chưa annotate.

Không trộn raw data trực tiếp với:

```text
dataset/images/train
```

---

# 20. Tạo metadata nguồn dữ liệu

Tạo:

```text
dataset/SOURCES.md
```

Template:

```markdown
# Dataset Sources

## Own Photos
- Source: Self-collected
- Device:
- Date:
- Number of images:
- Notes:

## Own Videos
- Source: Self-recorded
- Number of videos:
- Extracted frames:
- Notes:

## Public Dataset 1
- Name:
- URL:
- License:
- Images used:
- Notes:
```

---

# 21. Script trích frame từ video

Tạo file:

```text
scripts/extract_frames.py
```

Yêu cầu:

- Input video path.
- Output folder.
- Cho phép chọn khoảng thời gian giữa hai frame.
- Default: 1 giây.
- Không overwrite ảnh cũ ngoài ý muốn.
- Tên file có nguồn video.

Ví dụ:

```powershell
python scripts/extract_frames.py --video dataset_raw/videos/group01.mp4 --output dataset_raw/extracted_frames --interval 1.0
```

Tên output:

```text
group01_frame_000001.jpg
group01_frame_000002.jpg
...
```

---

# 22. Script kiểm tra annotation

Tạo:

```text
scripts/check_dataset.py
```

Script phải kiểm tra:

1. Ảnh có label tương ứng.
2. Label có ảnh tương ứng.
3. Class ID chỉ được:
   ```text
   0
   1
   2
   ```
4. Mỗi dòng label có đúng 5 giá trị.
5. Các giá trị tọa độ nằm trong range hợp lệ.
6. Width/height > 0.
7. Thống kê số lượng:
   ```text
   images
   labels
   mask instances
   no_mask instances
   unknown instances
   ```
8. Thống kê riêng:
   ```text
   train
   val
   test
   ```
9. Cảnh báo class imbalance.
10. Không tự sửa annotation sai.

Ví dụ output:

```text
[TRAIN]
Images: 350
Labels: 350
mask: 612
no_mask: 590
unknown: 301

[VAL]
Images: 100
...

[TEST]
Images: 50
...

[OK] Dataset validation passed.
```

---

# 23. Script xem annotation

Tạo:

```text
scripts/visualize_labels.py
```

Chức năng:

- Đọc ảnh.
- Đọc YOLO annotation.
- Vẽ bounding box.
- Vẽ class name.
- Hiển thị hoặc lưu preview.

Class map:

```python
{
    0: "mask",
    1: "no_mask",
    2: "unknown"
}
```

Không thay đổi ảnh gốc.

Output preview:

```text
results/dataset_preview/
```

---

# 24. Quy tắc split dataset

Không random split đơn giản nếu ảnh đến từ video/session giống nhau.

Ưu tiên:

```text
source-based split
```

Ví dụ:

```text
Train:
session01
session02
session03

Val:
session04

Test:
session05
```

Tỷ lệ mục tiêu:

```text
70 / 20 / 10
```

Không bắt buộc chính xác tuyệt đối nếu phải giữ group/source không bị tách.

---

# 25. Không augmentation ở bước này

Bước 2 chỉ:

```text
collect
clean
annotate
split
validate
```

Chưa thực hiện:

```text
train-time augmentation
mosaic
mixup
HSV augmentation
```

Các augmentation sẽ cấu hình khi training.

Không tạo hàng nghìn ảnh augmented thủ công trước khi split.

---

# 26. Naming convention

Ảnh tự chụp:

```text
own_scene01_0001.jpg
own_scene01_0002.jpg
```

Video:

```text
vid01_frame_0001.jpg
vid01_frame_0002.jpg
```

Public source:

```text
public_<source>_0001.jpg
```

Tên file:

- Không dấu.
- Không space.
- Không ký tự đặc biệt.
- Dùng lowercase nếu có thể.

---

# 27. Chất lượng ảnh

Không loại toàn bộ ảnh khó.

Dataset cần cả:

```text
ảnh dễ
ảnh trung bình
ảnh khó
```

Nhưng loại các ảnh:

```text
hỏng file
đen hoàn toàn
không load được
duplicate hoàn toàn
không liên quan đến bài toán
```

---

# 28. Duplicate

Không để cùng một ảnh xuất hiện nhiều lần.

Nếu có near-duplicate do video:

- Chỉ giữ một số frame tiêu biểu.
- Không để near-duplicate bị chia sang train/test.

Có thể bổ sung script duplicate detection ở bước sau nếu dataset lớn.

---

# 29. Acceptance Criteria — Bước 2

Bước 2 hoàn thành khi:

- [ ] Có thư mục `dataset_raw/`.
- [ ] Có quy tắc class rõ ràng.
- [ ] Có ít nhất một batch ảnh tự thu thập.
- [ ] Dataset có ảnh nhiều người.
- [ ] Annotation dùng bounding box quanh face/head region.
- [ ] Ba class đúng:
  - [ ] `mask`
  - [ ] `no_mask`
  - [ ] `unknown`
- [ ] Có `dataset/SOURCES.md`.
- [ ] Có `scripts/extract_frames.py`.
- [ ] Có `scripts/check_dataset.py`.
- [ ] Có `scripts/visualize_labels.py`.
- [ ] Dataset được split theo source/scene.
- [ ] Không có train-test leakage rõ ràng.
- [ ] `check_dataset.py` không báo lỗi format.
- [ ] Đã kiểm tra trực quan một số annotation.
- [ ] Chưa train YOLO.

---

# 30. Output Antigravity Code cần trả về

Sau khi thực hiện task này, trả về:

1. Cây thư mục project sau khi bổ sung Step 2.
2. Các file/script đã tạo.
3. Nội dung `dataset/SOURCES.md`.
4. Cách sử dụng `extract_frames.py`.
5. Cách sử dụng `check_dataset.py`.
6. Cách sử dụng `visualize_labels.py`.
7. Thống kê dataset nếu dataset đã có.
8. Các lỗi/cảnh báo dataset nếu có.
9. Không tự động bắt đầu training.

---

# 31. Những việc Antigravity KHÔNG được làm

Không:

```text
train YOLO
download model weights để train
tự tạo annotation giả
tự đổi class
gán toàn bộ object thành unknown
dùng Teachable Machine
tích hợp ESP32
tích hợp Telegram
```

Nếu chưa có ảnh:

```text
chỉ chuẩn bị tool/script/cấu trúc
và hướng dẫn người dùng thu thập + annotate
```

---

# 32. Sau khi Step 2 hoàn thành

Bước tiếp theo:

```text
BƯỚC 3
Dataset QA + Training Configuration
        ↓
BƯỚC 4
Train YOLO
        ↓
BƯỚC 5
Evaluate
        ↓
BƯỚC 6
Inference nhiều người
        ↓
BƯỚC 7
ESP32-CAM Integration
```

Không chuyển bước nếu dataset chưa đủ sạch.

---

# 33. Tóm tắt task hiện tại

```text
TASK:
Chuẩn bị dataset object detection nhiều người.

BOX TARGET:
Face/head region.

CLASSES:
0 mask
1 no_mask
2 unknown

PRIMARY GOAL:
Dataset sạch, nhất quán, không leakage,
sẵn sàng cho YOLO training.

DO NOT TRAIN YET.
```
