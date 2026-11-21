# -*- coding: utf-8 -*-
"""
cloud_api/config.py — Cấu hình cho Cloud Inference API.
======================================================
Tập hợp các tham số cấu hình:
- Model weights path (best.pt)
- Device suy luận (mặc định CPU cho EC2 t3/c6i/t4g)
- Ngưỡng hiệu chuẩn (conf_floor, accept_thresh) đồng bộ từ configs/thresholds.json
- Bảo mật: API Key, giới hạn kích thước upload, kiểm tra MIME type
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# Root workspace
ROOT = Path(__file__).resolve().parents[1]

# Cấu hình nạp từ thresholds.json nếu có
DEFAULT_THRESHOLDS_PATH = ROOT / "configs" / "thresholds.json"
DEFAULT_CONF_FLOOR = 0.20
DEFAULT_ACCEPT_THRESH = 0.75
DEFAULT_CLASS_NAMES = {0: "mask", 1: "no_mask"}
DEFAULT_UNKNOWN_LABEL = "unknown"

if DEFAULT_THRESHOLDS_PATH.exists():
    try:
        with open(DEFAULT_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            DEFAULT_CONF_FLOOR = float(cfg.get("conf_floor", DEFAULT_CONF_FLOOR))
            DEFAULT_ACCEPT_THRESH = float(cfg.get("accept_thresh", DEFAULT_ACCEPT_THRESH))
            if "class_names" in cfg:
                DEFAULT_CLASS_NAMES = {int(k): v for k, v in cfg["class_names"].items()}
            DEFAULT_UNKNOWN_LABEL = cfg.get("unknown_label", DEFAULT_UNKNOWN_LABEL)
    except Exception:
        pass


class CloudAPIConfig:
    """Đối tượng cấu hình trung tâm cho Cloud Inference API."""

    def __init__(self):
        # Đường dẫn weights model
        default_model = (
            ROOT / "runs" / "detect" / "results" / "train" / "antigravity_gpu_v1" / "weights" / "best.pt"
        )
        self.model_path = Path(os.getenv("CLOUD_MODEL_PATH", str(default_model)))

        # Thiết bị suy luận: "cpu" cho EC2 standard, hoặc "cuda" / "0" nếu có GPU EC2 (g4dn/g5)
        self.device = os.getenv("CLOUD_INFERENCE_DEVICE", "cpu").strip().lower()

        # Phiên bản mô hình
        self.model_version = os.getenv("CLOUD_MODEL_VERSION", "yolo11n_antigravity_v1")

        # Ngưỡng suy luận
        self.conf_floor = float(os.getenv("CLOUD_CONF_FLOOR", str(DEFAULT_CONF_FLOOR)))
        self.accept_thresh = float(os.getenv("CLOUD_ACCEPT_THRESH", str(DEFAULT_ACCEPT_THRESH)))
        self.class_names: Dict[int, str] = DEFAULT_CLASS_NAMES
        self.unknown_label: str = DEFAULT_UNKNOWN_LABEL

        # Bảo mật API Key: header X-API-Key
        # Khuyến nghị luôn cấu hình CLOUD_API_KEY trên production.
        self.api_key: Optional[str] = os.getenv("CLOUD_API_KEY", "").strip() or None

        # Giới hạn kích thước file upload (mặc định 5MB)
        self.max_content_length: int = int(os.getenv("CLOUD_MAX_UPLOAD_BYTES", 5 * 1024 * 1024))

        # MIME types và magic bytes được chấp nhận
        self.allowed_mime_types: List[str] = ["image/jpeg", "image/jpg", "image/png"]

        # Server network
        self.host = os.getenv("CLOUD_API_HOST", "0.0.0.0")
        self.port = int(os.getenv("CLOUD_API_PORT", "8000"))

        # Timeout suy luận (giây)
        self.inference_timeout_seconds = float(os.getenv("CLOUD_INFERENCE_TIMEOUT", "10.0"))

        # ── S3 Results Logging ──────────────────────────────────────────────
        # Bật/tắt ghi kết quả nhận diện lên S3 (mặc định: tắt)
        self.s3_results_enabled: bool = os.getenv(
            "S3_RESULTS_ENABLED", "false"
        ).strip().lower() in ("true", "1", "yes")

        # Tên S3 Bucket (Private, đã tồn tại)
        self.s3_results_bucket: str = os.getenv(
            "S3_RESULTS_BUCKET",
            "esp32-yolo-lab3-511974511954-ap-southeast-1-an",
        ).strip()

        # Prefix trong bucket (kết quả lưu tại: <prefix>/YYYY/MM/DD/<event_id>.json)
        self.s3_results_prefix: str = os.getenv("S3_RESULTS_PREFIX", "results").strip()

        # AWS Region cho boto3 client
        self.s3_results_region: str = os.getenv("S3_RESULTS_REGION", "ap-southeast-1").strip()

        # Cooldown tối thiểu giữa 2 lần ghi (giây), áp dụng khi không có state change
        self.s3_results_cooldown: float = float(os.getenv("S3_RESULTS_COOLDOWN", "30.0"))

        # Kích thước tối đa hàng đợi ghi S3 (bản ghi vượt quá sẽ bị bỏ qua)
        self.s3_results_queue_size: int = int(os.getenv("S3_RESULTS_QUEUE_SIZE", "100"))

        # Nguồn camera mặc định cho bản ghi
        self.s3_camera_source: str = os.getenv("S3_CAMERA_SOURCE", "esp32_cam").strip()


# Khởi tạo singleton cấu hình
config = CloudAPIConfig()
