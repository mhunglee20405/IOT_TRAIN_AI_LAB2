# -*- coding: utf-8 -*-
"""
cloud_api/model_service.py — Dịch vụ nạp mô hình và suy luận YOLO11n thời gian thực.
==================================================================================
- Tải mô hình YOLO11n một lần duy nhất lúc khởi động server (Lifespan).
- Thực hiện suy luận hoàn toàn trong RAM (không lưu ảnh xuống ổ đĩa).
- Đo lường chi tiết các mốc thời gian:
    + cloud_decode_ms: Giải mã buffer ảnh thành numpy array
    + cloud_inference_ms: Thời gian chạy YOLO forward pass
    + cloud_postprocess_ms: Áp ngưỡng unknown và bóc tách tọa độ bbox_xyxy
"""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from cloud_api.config import config
from cloud_api.schemas import (
    Counts,
    DetectionItem,
    ImageDimensions,
    InferenceResponse,
    TimingMetrics,
)

logger = logging.getLogger("cloud_api.model_service")


class ModelService:
    """Singleton quản lý vòng đời và suy luận của YOLO11n trên Cloud."""

    def __init__(self):
        self.model: Optional[YOLO] = None
        self.model_path: Optional[Path] = None
        self.device: str = "cpu"
        self.is_cuda: bool = False
        self.start_time: float = 0.0
        self.is_loaded: bool = False

    def load(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        """Nạp model YOLO11n vào bộ nhớ và warm-up."""
        self.model_path = model_path or config.model_path
        self.device = device or config.device
        self.is_cuda = (self.device != "cpu" and torch.cuda.is_available())

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file model weights tại: {self.model_path}. "
                "Hãy kiểm tra biến CLOUD_MODEL_PATH hoặc tải từ AWS S3."
            )

        logger.info(f"[*] Đang nạp model YOLO11n từ: {self.model_path} (Device: {self.device})")
        self.model = YOLO(str(self.model_path))

        # Warm-up 1 frame giả định kích thước 640x640 để loại bỏ cold-start latency
        logger.info("[*] Thực hiện warm-up model với frame mẫu 640x640...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model.predict(
            source=dummy_img,
            imgsz=640,
            conf=config.conf_floor,
            device=self.device,
            verbose=False,
        )

        self.start_time = time.time()
        self.is_loaded = True
        logger.info(f"[*] Model đã sẵn sàng phục vụ suy luận trên {self.device.upper()}.")

    @property
    def uptime_seconds(self) -> float:
        """Thời gian hoạt động từ khi nạp model."""
        if not self.is_loaded:
            return 0.0
        return round(time.time() - self.start_time, 2)

    def predict(self, image_bytes: bytes, frame_id: Optional[str] = None) -> InferenceResponse:
        """
        Thực hiện giải mã ảnh và suy luận nhận diện khẩu trang.
        Tuyệt đối không lưu ảnh khuôn mặt xuống ổ cứng.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model chưa được nạp vào bộ nhớ server.")

        t_start_all = time.perf_counter()

        # 1. Đo thời gian giải mã JPEG/PNG từ RAM
        t_dec_start = time.perf_counter()
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Không thể giải mã mảng nhị phân thành ảnh hợp lệ.")
        t_dec_end = time.perf_counter()
        cloud_decode_ms = (t_dec_end - t_dec_start) * 1000.0

        h, w = frame.shape[:2]

        # 2. Đo thời gian suy luận YOLO11n
        t_infer_start = time.perf_counter()
        if self.is_cuda:
            torch.cuda.synchronize()

        results = self.model.predict(
            source=frame,
            imgsz=640,
            conf=config.conf_floor,
            device=self.device,
            verbose=False,
        )[0]

        if self.is_cuda:
            torch.cuda.synchronize()
        t_infer_end = time.perf_counter()
        cloud_inference_ms = (t_infer_end - t_infer_start) * 1000.0

        # 3. Đo thời gian bóc tách bboxes và áp ngưỡng unknown đồng bộ
        t_post_start = time.perf_counter()
        detections: List[DetectionItem] = []
        counts = {"mask": 0, "no_mask": 0, "unknown": 0}

        if hasattr(results, "boxes") and results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                raw_name = results.names.get(cls_id, f"cls_{cls_id}")

                # Đồng bộ phân loại unknown dựa trên accept_thresh
                if conf < config.accept_thresh:
                    label = config.unknown_label
                    is_unknown = True
                elif raw_name == "mask":
                    label = "mask"
                    is_unknown = False
                else:
                    label = "no_mask"
                    is_unknown = False

                counts[label] += 1

                # Tọa độ bbox_xyxy trên ảnh gốc
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Giới hạn trong kích thước ảnh
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                detections.append(
                    DetectionItem(
                        class_id=cls_id,
                        label=label,
                        confidence=round(conf, 4),
                        bbox_xyxy=[x1, y1, x2, y2],
                        raw_class_name=raw_name,
                        is_unknown=is_unknown,
                    )
                )

        t_post_end = time.perf_counter()
        cloud_postprocess_ms = (t_post_end - t_post_start) * 1000.0

        t_end_all = time.perf_counter()
        cloud_total_ms = (t_end_all - t_start_all) * 1000.0

        return InferenceResponse(
            frame_id=frame_id,
            model_version=config.model_version,
            image=ImageDimensions(width=w, height=h),
            detections=detections,
            counts=Counts(
                mask=counts["mask"],
                no_mask=counts["no_mask"],
                unknown=counts["unknown"],
                total=sum(counts.values()),
            ),
            timing_ms=TimingMetrics(
                cloud_decode_ms=round(cloud_decode_ms, 2),
                cloud_inference_ms=round(cloud_inference_ms, 2),
                cloud_postprocess_ms=round(cloud_postprocess_ms, 2),
                cloud_total_ms=round(cloud_total_ms, 2),
            ),
        )


# Singleton
model_service = ModelService()
