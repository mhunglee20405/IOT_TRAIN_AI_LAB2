# -*- coding: utf-8 -*-
"""
src/cloud_client.py — HTTP Client kết nối AWS Cloud Inference API.
=================================================================
Đảm nhiệm:
  - Gửi frame ảnh JPEG từ Local Gateway lên AWS EC2 qua HTTP POST multipart.
  - Hỗ trợ kết nối qua SSH Tunnel (mặc định: http://127.0.0.1:9000/v1/infer) hoặc HTTPS.
  - Bấm giờ chính xác:
      + cloud_request_response_ms: Đo bằng time.perf_counter() trên laptop (Client RTT).
      + cloud_inference_ms: Trích xuất từ timings_ms.cloud_inference_ms trên server.
      + cloud_server_ms: Trích xuất từ timings_ms.cloud_server_ms trên server.
      + cloud_transport_overhead_est_ms: max(0.0, cloud_request_response_ms - cloud_server_ms).
        (Lưu ý: Đây là ước lượng tổng overhead gồm network + serialization + HTTP transfer,
         không được mô tả là độ trễ mạng thuần túy).
  - Chuẩn hóa detections trả về để đồng bộ quy tắc phân loại unknown và renderer với Local GPU.
  - Xử lý lỗi kết nối, timeout và response không hợp lệ (không crash pipeline).
"""

from dataclasses import dataclass, field
import io
import logging
from typing import Any, Dict, List, Optional, Tuple
import time
import uuid

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger("CloudInferenceClient")


@dataclass
class CloudInferenceResult:
    """Kết quả suy luận nhận từ Cloud Server kèm bóc tách thời gian chi tiết."""
    success: bool
    frame_id: str = ""
    model: str = "YOLO11n"
    device: str = "cpu"
    image_width: int = 0
    image_height: int = 0
    raw_detections: List[Dict[str, Any]] = field(default_factory=list)
    cloud_inference_ms: float = 0.0
    cloud_server_ms: float = 0.0
    cloud_decode_ms: float = 0.0
    cloud_predict_wall_ms: float = 0.0
    cloud_request_response_ms: float = 0.0
    cloud_transport_overhead_est_ms: float = 0.0
    error_msg: str = ""
    status_code: int = 0


class CloudInferenceClient:
    """HTTP Client cho Cloud Inference API."""

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:9000/v1/infer",
        timeout: float = 4.0,
        api_key: Optional[str] = None,
    ):
        self.api_url = api_url.strip()
        self.timeout = float(timeout)
        self.api_key = api_key.strip() if api_key else None

        # Thiết lập session với connection pool
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=4, max_retries=0)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Base URL cho health check
        if "/v1/infer" in self.api_url:
            self.base_url = self.api_url.split("/v1/infer")[0]
        else:
            self.base_url = self.api_url.rstrip("/")

    def close(self):
        """Đóng session an toàn."""
        try:
            self.session.close()
        except Exception:
            pass

    def health_check(self, timeout: float = 2.0) -> Tuple[bool, Dict[str, Any], str]:
        """
        Kiểm tra trạng thái máy chủ Cloud qua endpoint GET /health.
        Trả về (is_healthy, response_dict, message).
        """
        health_url = f"{self.base_url}/health"
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            res = self.session.get(health_url, headers=headers, timeout=timeout)
            if res.status_code == 200:
                data = res.json()
                is_loaded = data.get("model_loaded", True)
                status_str = data.get("status", "ok")
                if is_loaded and status_str in ("ok", "healthy"):
                    return True, data, "Cloud API sẵn sàng và model đã nạp."
                return False, data, f"Model chưa nạp hoặc trạng thái: {status_str}"
            return False, {}, f"Cloud /health trả về HTTP {res.status_code}"
        except requests.Timeout:
            return False, {}, f"Timeout kết nối tới Cloud /health (>{timeout}s)"
        except requests.RequestException as e:
            return False, {}, f"Lỗi kết nối tới Cloud /health: {e}"
        except Exception as e:
            return False, {}, f"Lỗi không xác định khi gọi health check: {e}"

    def infer(self, jpeg_bytes: bytes, frame_id: Optional[str] = None) -> CloudInferenceResult:
        """
        Gửi ảnh JPEG dạng multipart/form-data tới endpoint POST /v1/infer.
        Trả về CloudInferenceResult chứa detections và timing bóc tách.
        """
        fid = frame_id or str(uuid.uuid4())
        if not jpeg_bytes:
            return CloudInferenceResult(
                success=False,
                frame_id=fid,
                error_msg="Dữ liệu ảnh JPEG rỗng (0 bytes).",
            )

        headers = {
            "Accept": "application/json",
            "User-Agent": "ESP32-Local-Gateway/2.0",
            "X-Frame-ID": fid,
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        files = {
            "file": ("frame.jpg", io.BytesIO(jpeg_bytes), "image/jpeg"),
        }
        data = {
            "frame_id": fid,
        }

        t_req_start = time.perf_counter()
        try:
            response = self.session.post(
                self.api_url,
                files=files,
                data=data,
                headers=headers,
                timeout=self.timeout,
            )
            t_req_end = time.perf_counter()
            cloud_request_response_ms = (t_req_end - t_req_start) * 1000.0

            if response.status_code != 200:
                err_text = response.text[:120]
                logger.warning(f"Cloud API trả về HTTP {response.status_code}: {err_text}")
                return CloudInferenceResult(
                    success=False,
                    frame_id=fid,
                    status_code=response.status_code,
                    cloud_request_response_ms=round(cloud_request_response_ms, 2),
                    error_msg=f"HTTP {response.status_code}: {err_text}",
                )

            try:
                res_json = response.json()
            except Exception as json_err:
                return CloudInferenceResult(
                    success=False,
                    frame_id=fid,
                    status_code=response.status_code,
                    cloud_request_response_ms=round(cloud_request_response_ms, 2),
                    error_msg=f"Không thể giải mã JSON phản hồi: {json_err}",
                )

            # Bóc tách timings_ms từ response
            timings = res_json.get("timings_ms", {}) or res_json.get("timing_ms", {})
            cloud_inference_ms = float(timings.get("cloud_inference_ms", 0.0))
            cloud_server_ms = float(timings.get("cloud_server_ms", timings.get("cloud_total_ms", 0.0)))
            cloud_decode_ms = float(timings.get("cloud_decode_ms", 0.0))
            cloud_predict_wall_ms = float(timings.get("cloud_predict_wall_ms", cloud_inference_ms))

            # Ước lượng overhead vận chuyển (transport overhead)
            cloud_transport_overhead_est_ms = max(0.0, cloud_request_response_ms - cloud_server_ms)

            # Trích xuất detections
            detections = res_json.get("detections", [])
            width = int(res_json.get("image_width", res_json.get("image", {}).get("width", 0)))
            height = int(res_json.get("image_height", res_json.get("image", {}).get("height", 0)))
            model_name = res_json.get("model", res_json.get("model_version", "YOLO11n"))
            device_name = res_json.get("device", "cpu")

            return CloudInferenceResult(
                success=True,
                frame_id=res_json.get("frame_id", fid),
                model=model_name,
                device=device_name,
                image_width=width,
                image_height=height,
                raw_detections=detections,
                cloud_inference_ms=round(cloud_inference_ms, 2),
                cloud_server_ms=round(cloud_server_ms, 2),
                cloud_decode_ms=round(cloud_decode_ms, 2),
                cloud_predict_wall_ms=round(cloud_predict_wall_ms, 2),
                cloud_request_response_ms=round(cloud_request_response_ms, 2),
                cloud_transport_overhead_est_ms=round(cloud_transport_overhead_est_ms, 2),
                status_code=200,
            )

        except requests.Timeout:
            t_req_end = time.perf_counter()
            cloud_request_response_ms = (t_req_end - t_req_start) * 1000.0
            msg = f"Cloud Request Timeout (>{self.timeout}s)"
            logger.warning(msg)
            return CloudInferenceResult(
                success=False,
                frame_id=fid,
                cloud_request_response_ms=round(cloud_request_response_ms, 2),
                error_msg=msg,
            )
        except requests.RequestException as e:
            t_req_end = time.perf_counter()
            cloud_request_response_ms = (t_req_end - t_req_start) * 1000.0
            msg = f"Lỗi kết nối Cloud API: {str(e)[:120]}"
            logger.warning(msg)
            return CloudInferenceResult(
                success=False,
                frame_id=fid,
                cloud_request_response_ms=round(cloud_request_response_ms, 2),
                error_msg=msg,
            )
        except Exception as e:
            t_req_end = time.perf_counter()
            cloud_request_response_ms = (t_req_end - t_req_start) * 1000.0
            msg = f"Lỗi ngoại lệ client: {str(e)[:120]}"
            logger.error(msg, exc_info=True)
            return CloudInferenceResult(
                success=False,
                frame_id=fid,
                cloud_request_response_ms=round(cloud_request_response_ms, 2),
                error_msg=msg,
            )
