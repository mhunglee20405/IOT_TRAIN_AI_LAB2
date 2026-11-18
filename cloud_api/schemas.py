# -*- coding: utf-8 -*-
"""
cloud_api/schemas.py — Pydantic schemas cho request và response của Cloud API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    """Chi tiết một bounding box khuôn mặt được phát hiện."""
    class_id: int = Field(..., description="ID lớp mô hình dự đoán (0: mask, 1: no_mask)")
    label: str = Field(..., description="Nhãn phân loại (mask, no_mask, hoặc unknown theo accept_thresh)")
    confidence: float = Field(..., description="Độ tin cậy của mô hình (0.0 - 1.0)")
    bbox_xyxy: List[int] = Field(..., description="Tọa độ góc [x1, y1, x2, y2] trên ảnh gốc")
    raw_class_name: str = Field(..., description="Tên lớp nguyên bản trước khi qua ngưỡng unknown")
    is_unknown: bool = Field(..., description="True nếu confidence < accept_thresh")


class Counts(BaseModel):
    """Tổng hợp số lượng khuôn mặt phân theo nhãn."""
    mask: int = Field(0, description="Số khuôn mặt đeo khẩu trang")
    no_mask: int = Field(0, description="Số khuôn mặt không đeo khẩu trang")
    unknown: int = Field(0, description="Số khuôn mặt chưa rõ (dưới accept_thresh)")
    total: int = Field(0, description="Tổng số khuôn mặt được phát hiện")


class TimingMetrics(BaseModel):
    """Bóc tách thời gian xử lý tại Cloud Inference Server (mili-giây)."""
    cloud_decode_ms: float = Field(..., description="Thời gian giải mã JPEG thành mảng BGR")
    cloud_inference_ms: float = Field(..., description="Thời gian mô hình YOLO11n suy luận")
    cloud_postprocess_ms: float = Field(..., description="Thời gian bóc tách bboxes và áp ngưỡng unknown")
    cloud_total_ms: float = Field(..., description="Tổng thời gian xử lý nội bộ server (decode + infer + postprocess)")


class ImageDimensions(BaseModel):
    """Kích thước ảnh gốc nhận từ client."""
    width: int = Field(..., description="Chiều rộng ảnh (pixel)")
    height: int = Field(..., description="Chiều cao ảnh (pixel)")


class InferenceResponse(BaseModel):
    """Phản hồi đầy đủ từ endpoint POST /v1/infer."""
    frame_id: Optional[str] = Field(None, description="ID định danh khung hình từ client gửi lên")
    model_version: str = Field(..., description="Phiên bản mô hình đang phục vụ")
    image: ImageDimensions = Field(..., description="Kích thước ảnh gốc")
    detections: List[DetectionItem] = Field(default_factory=list, description="Danh sách các đối tượng phát hiện")
    counts: Counts = Field(..., description="Số đếm tổng hợp theo nhãn")
    timing_ms: TimingMetrics = Field(..., description="Đo lường thời gian xử lý phía Cloud")


class HealthResponse(BaseModel):
    """Phản hồi kiểm tra sức khỏe hệ thống GET /health."""
    status: str = Field("healthy", description="Trạng thái server")
    model_loaded: bool = Field(..., description="Model YOLO đã nạp vào bộ nhớ hay chưa")
    model_path: str = Field(..., description="Đường dẫn file weights")
    device: str = Field(..., description="Thiết bị suy luận (cpu hoặc cuda)")
    model_version: str = Field(..., description="Phiên bản mô hình")
    conf_floor: float = Field(..., description="Ngưỡng phát hiện tối thiểu")
    accept_thresh: float = Field(..., description="Ngưỡng chấp nhận tin cậy (dưới mức này là unknown)")
    uptime_seconds: float = Field(..., description="Thời gian server hoạt động tính bằng giây")
