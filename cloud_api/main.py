# -*- coding: utf-8 -*-
"""
cloud_api/main.py — FastAPI Application cho YOLO11n Cloud Inference Server.
=============================================================================
Endpoints:
  - GET  /health   : Kiểm tra tình trạng sức khỏe server và thông tin model.
  - POST /v1/infer : Nhận ảnh qua multipart/form-data, trả về JSON tọa độ và metrics.

Thiết kế an toàn:
  - Xác thực bằng header 'X-API-Key'.
  - Kiểm tra dung lượng (tối đa 5MB) và magic bytes nhị phân của ảnh.
  - Xử lý hoàn toàn trong bộ nhớ RAM, không lưu ảnh khuôn mặt xuống đĩa.
  - Tuyệt đối không log thông tin nhạy cảm (secrets, API keys, payload ảnh).
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import sys
import time
from typing import Optional
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cloud_api.config import config
from cloud_api.model_service import model_service
from cloud_api.schemas import HealthResponse, InferenceResponse
from cloud_api.security import validate_image_payload, verify_api_key

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("cloud_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời server: Nạp model YOLO11n một lần duy nhất lúc khởi động."""
    logger.info("=" * 60)
    logger.info("  Khoi dong YOLO11n Cloud Inference Server")
    logger.info(f"  Model Path : {config.model_path}")
    logger.info(f"  Device     : {config.device.upper()}")
    logger.info(f"  Conf Floor : {config.conf_floor} | Accept Thresh : {config.accept_thresh}")
    logger.info(f"  API Key Auth: {'BAT' if config.api_key else 'TAT (Phat trien noi bo)'}")
    logger.info("=" * 60)

    try:
        model_service.load(model_path=config.model_path, device=config.device)
    except Exception as e:
        logger.error(f"[!] Loi nap model khi khoi dong server: {e}", exc_info=True)
        # Khong crash ngay de health check co the bao loi chinh xac
    yield
    logger.info("[*] Dang tat Cloud Inference Server an toan.")


app = FastAPI(
    title="YOLO11n Cloud Mask Detection API",
    description="API suy luận nhận diện khẩu trang YOLO11n trên AWS Cloud (EC2)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware (Hỗ trợ gọi từ Web Dashboard hoặc SSH Tunnel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", summary="Root endpoint")
def root():
    return {
        "service": "YOLO11n Cloud Mask Detection API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "infer_url": "/v1/infer",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Kiểm tra trạng thái hoạt động của server và model",
)
def health_check():
    """Trả về tình trạng server, model đã nạp hay chưa, và các ngưỡng suy luận."""
    if not model_service.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "model_path": str(config.model_path),
                "device": config.device,
                "model_version": config.model_version,
                "conf_floor": config.conf_floor,
                "accept_thresh": config.accept_thresh,
                "uptime_seconds": 0.0,
            },
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_path=str(model_service.model_path),
        device=model_service.device,
        model_version=config.model_version,
        conf_floor=config.conf_floor,
        accept_thresh=config.accept_thresh,
        uptime_seconds=model_service.uptime_seconds,
    )


@app.post(
    "/v1/infer",
    response_model=InferenceResponse,
    summary="Suy luận phát hiện khẩu trang trên một frame ảnh JPEG/PNG",
    dependencies=[Depends(verify_api_key)],
)
async def infer_frame(
    file: UploadFile = File(..., description="File ảnh JPEG hoặc PNG từ camera"),
    frame_id: Optional[str] = Form(None, description="Mã định danh frame của client để đồng bộ latency"),
):
    """
    Nhận frame multipart từ Gateway / ESP32-CAM và trả kết quả bounding boxes JSON.
    Quy trình:
      1. Đọc nội dung nhị phân vào bộ nhớ RAM.
      2. Xác thực dung lượng và magic bytes.
      3. Gọi ModelService giải mã và suy luận YOLO11n.
      4. Trả về kết quả JSON kèm thời gian bóc tách từng bước.
    """
    req_id = frame_id or str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    # 1. Đọc nội dung ảnh từ UploadFile
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"[{req_id}] Loi doc multipart file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    # 2. Kiểm tra tính hợp lệ và dung lượng ảnh
    validate_image_payload(content)

    # 3. Thực hiện suy luận
    try:
        # Chạy inference trong thread pool của asyncio để tránh chặn event loop khi chạy CPU bound
        loop = asyncio.get_running_loop()
        response_data: InferenceResponse = await loop.run_in_executor(
            None, model_service.predict, content, req_id
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as err:
        logger.error(f"[{req_id}] Loi suy luan model: {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference server error",
        )

    total_req_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        f"[{req_id}] Infer OK | Detections: {len(response_data.detections)} "
        f"(mask: {response_data.counts.mask}, no_mask: {response_data.counts.no_mask}, "
        f"unknown: {response_data.counts.unknown}) | "
        f"Cloud Total: {response_data.timing_ms.cloud_total_ms:.1f}ms | Handler: {total_req_ms:.1f}ms"
    )

    return response_data
