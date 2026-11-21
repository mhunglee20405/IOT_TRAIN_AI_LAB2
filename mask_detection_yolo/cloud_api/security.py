# -*- coding: utf-8 -*-
"""
cloud_api/security.py — Xác thực API Key, kiểm tra an toàn dữ liệu và bảo mật.
"""

import logging
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status

from cloud_api.config import config

logger = logging.getLogger("cloud_api.security")

# Magic bytes cho kiểm tra định dạng nhị phân ảnh
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> bool:
    """
    Xác thực API Key từ HTTP Header 'X-API-Key' sử dụng secrets.compare_digest
    để ngăn chặn các cuộc tấn công Timing Attack.
    
    Lưu ý: Tuyệt đối không in giá trị API key nhận được ra log console.
    """
    if config.api_key is None:
        # Nếu chưa cấu hình CLOUD_API_KEY (chế độ phát triển nội bộ), cho phép qua
        return True

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing 'X-API-Key' header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # So sánh an toàn chống timing attack
    is_valid = secrets.compare_digest(x_api_key.encode("utf-8"), config.api_key.encode("utf-8"))
    if not is_valid:
        logger.warning("Truy cập thất bại: API key không khớp.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return True


def validate_image_payload(content: bytes) -> str:
    """
    Kiểm tra kích thước và tính hợp lệ của nội dung ảnh (Magic bytes).
    Trả về định dạng ("jpeg" hoặc "png").
    Ném HTTPException nếu vi phạm.
    """
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image payload received",
        )

    if len(content) > config.max_content_length:
        max_mb = config.max_content_length / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Image size exceeds limit of {max_mb:.1f} MB",
        )

    if content.startswith(JPEG_MAGIC):
        return "jpeg"
    elif content.startswith(PNG_MAGIC):
        return "png"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Supported formats: JPEG, PNG",
        )
