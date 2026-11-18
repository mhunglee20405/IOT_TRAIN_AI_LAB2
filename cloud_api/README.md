# YOLO11n Cloud Mask Detection API Server

Mô-đun máy chủ suy luận trên đám mây (AWS EC2) phục vụ nhận diện khẩu trang thời gian thực từ ESP32-CAM Gateway.

---

## 1. Kiến Trúc Tổng Quan

```
[ ESP32-CAM ]
      │ (JPEG qua HTTP/Wi-Fi)
      ▼
[ Laptop Gateway ] ── (SSH Tunnel / HTTPS) ──► [ AWS EC2 Cloud API ]
  - Web Dashboard                                 - FastAPI + Uvicorn
  - Telegram Bot Service                          - YOLO11n (CPU inference)
  - Render Bounding Boxes                         - Timing metrics breakdown
                                                  - Model nạp từ S3 Private
```

---

## 2. Đặc Tả API

### GET `/health`
Kiểm tra sức khỏe hệ thống và trạng thái nạp mô hình.
- **Phản hồi mẫu (HTTP 200)**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/opt/mask_detection/weights/best.pt",
  "device": "cpu",
  "model_version": "yolo11n_antigravity_v1",
  "conf_floor": 0.2,
  "accept_thresh": 0.75,
  "uptime_seconds": 124.5
}
```

### POST `/v1/infer`
Gửi frame ảnh dạng `multipart/form-data` để nhận diện khuôn mặt và khẩu trang.
- **Headers**:
  - `X-API-Key`: Khóa bí mật xác thực (nếu server bật cấu hình `CLOUD_API_KEY`).
- **Form Data**:
  - `file`: Ảnh định dạng JPEG hoặc PNG (tối đa 5MB).
  - `frame_id` (tùy chọn): ID định danh frame để gateway đối chiếu độ trễ hiển thị.
- **Phản hồi mẫu (HTTP 200)**:
```json
{
  "frame_id": "frame_101",
  "model_version": "yolo11n_antigravity_v1",
  "image": {
    "width": 640,
    "height": 480
  },
  "detections": [
    {
      "class_id": 0,
      "label": "mask",
      "confidence": 0.834,
      "bbox_xyxy": [107, 68, 442, 453],
      "raw_class_name": "mask",
      "is_unknown": false
    }
  ],
  "counts": {
    "mask": 1,
    "no_mask": 0,
    "unknown": 0,
    "total": 1
  },
  "timing_ms": {
    "cloud_decode_ms": 1.2,
    "cloud_inference_ms": 26.5,
    "cloud_postprocess_ms": 0.5,
    "cloud_total_ms": 28.2
  }
}
```

---

## 3. Chạy Kiểm Thử Cục Bộ (Phase 1)

1. Cài đặt thư viện:
```bash
pip install -r cloud_api/requirements.txt
```

2. Khởi động server (CPU mode):
```bash
python -m uvicorn cloud_api.main:app --host 127.0.0.1 --port 8000 --log-level info
```

3. Chạy Smoke Test Client kiểm tra độ trễ và bounding boxes:
```bash
python cloud_api/scripts/test_client.py --url http://127.0.0.1:8000 --iterations 5
```

4. Chạy Unit Tests tự động:
```bash
python -m unittest tests/test_cloud_api.py
```

---

## 4. Hướng Dẫn Triển Khai AWS S3 & EC2 (Phase 2)

> **CẢNH BÁO CHI PHÍ**: Kiểm tra Free Tier hoặc xác nhận trước khi khởi tạo tài nguyên AWS có tính phí.

### Bước 4.1: Tạo S3 Bucket Private và Upload Model Weights
1. Tạo S3 Bucket với cài đặt Block Public Access:
```bash
aws s3api create-bucket \
    --bucket mask-yolo-weights-prod \
    --region ap-southeast-1 \
    --create-bucket-configuration LocationConstraint=ap-southeast-1
```
2. Bật mã hóa mặc định Server-Side Encryption (SSE-S3):
```bash
aws s3api put-bucket-encryption \
    --bucket mask-yolo-weights-prod \
    --server-side-encryption-configuration '{"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}'
```
3. Upload `best.pt` bằng script hỗ trợ:
```bash
python cloud_api/scripts/upload_to_s3.py --bucket mask-yolo-weights-prod --key weights/best.pt
```

### Bước 4.2: Cấu hình IAM Role Tối Thiểu Cho EC2 (Least Privilege)
Tạo Policy chỉ cấp quyền đọc duy nhất file `weights/best.pt`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowDownloadModelWeightsOnly",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::mask-yolo-weights-prod/weights/best.pt"
    }
  ]
}
```
Gắn IAM Role này vào EC2 Instance (Instance Profile). Nhờ đó, EC2 hoàn toàn **không cần lưu trữ access key hay secret key** trong mã nguồn hay file `.env`.

### Bước 4.3: Khởi Tạo và Cấu Hình EC2 Instance
- **Hệ điều hành**: Ubuntu 22.04 hoặc 24.04 LTS (x86_64).
- **Cấu hình**: `t3.medium` (2 vCPU, 4GB RAM) hoặc `c6i.large` (Compute Optimized).
- **Security Group Inbound**:
  - Ban đầu (SSH Tunnel): **Chỉ mở cổng 22** từ IP máy tính của bạn (`My IP`). Tuyệt đối không mở cổng 8000 ra `0.0.0.0/0`.
  - Nâng cấp HTTPS: Mở thêm cổng 443 (HTTPS) từ IP cố định hoặc Cloudflare/ALB.

### Bước 4.4: Thiết Lập Server Trên EC2
Đăng nhập SSH vào EC2 và chạy các lệnh sau:
```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip libgl1-mesa-glx awscli

sudo mkdir -p /opt/mask_detection/weights
sudo chown -R ubuntu:ubuntu /opt/mask_detection

# Clone repository hoặc copy thư mục cloud_api
cd /opt/mask_detection
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r cloud_api/requirements.txt

# Tải model từ S3 bằng IAM Role
python3 cloud_api/scripts/download_model_s3.py --bucket mask-yolo-weights-prod --key weights/best.pt --dest /opt/mask_detection/weights/best.pt

# Cấu hình biến môi trường
cat << 'EOF' > /opt/mask_detection/.env
CLOUD_MODEL_PATH=/opt/mask_detection/weights/best.pt
CLOUD_INFERENCE_DEVICE=cpu
CLOUD_API_HOST=127.0.0.1
CLOUD_API_PORT=8000
CLOUD_API_KEY=YOUR_STRONG_RANDOM_SECRET_KEY
EOF
```

### Bước 4.5: Cài Đặt Systemd Service
```bash
sudo cp cloud_api/systemd/yolo_cloud_api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable yolo_cloud_api
sudo systemctl start yolo_cloud_api
sudo systemctl status yolo_cloud_api
```

---

## 5. Kết Nối Gateway (Phase 3)

### Kết Nối Qua SSH Tunnel (Bảo Mật Giai Đoạn Đầu)
Trên Laptop của bạn, mở một terminal duy trì SSH Tunnel:
```bash
ssh -N -L 8000:localhost:8000 -i /path/to/your-key.pem ubuntu@<EC2-PUBLIC-IP>
```
Lúc này, Laptop gửi request tới `http://127.0.0.1:8000/v1/infer`, toàn bộ dữ liệu ảnh sẽ được mã hóa trong đường hầm SSH đến EC2 mà không cần mở port trên Internet.

### Nâng Cấp HTTPS (Production)
Cài đặt Caddy Server tự động cấp chứng chỉ Let's Encrypt trên EC2:
```caddy
your-domain.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

---

## 6. Bảo Mật & Quyền Riêng Tư (Privacy & Security)

1. **Không lưu trữ ảnh**: Dữ liệu ảnh khuôn mặt từ client chỉ lưu tạm trong RAM để giải mã thành numpy array và tự động giải phóng bởi Garbage Collector sau khi response hoàn tất.
2. **Không log secrets**: Tuyệt đối không log nội dung API Key hoặc HTTP Authorization header.
3. **Phòng chống Timing Attack**: So sánh API Key bằng `secrets.compare_digest`.
4. **Giới hạn tải**: Tự động từ chối request vượt quá 5MB và các file không đúng magic bytes JPEG/PNG.
