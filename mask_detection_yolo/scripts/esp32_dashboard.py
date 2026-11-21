# -*- coding: utf-8 -*-
"""
scripts/esp32_dashboard.py  —  Web Dashboard Giám Sát Khẩu Trang ESP32-CAM (FastAPI + YOLO11n)
=============================================================================================
Web Dashboard hiển thị trực tiếp luồng MJPEG và số liệu thống kê phát hiện khẩu trang:
  - Video stream MJPEG từ ESP32-CAM (/capture) với bounding box YOLO11n.
  - Phân loại: mask (xanh), no_mask (đỏ), unknown (cam/xám cho detection < accept).
  - Bộ đếm thời gian thực: mask, no_mask, unknown, tổng số khuôn mặt.
  - Đo FPS pipeline thực tế và trạng thái kết nối camera.
  - Vòng lặp capture + inference duy nhất (shared worker), không chạy YOLO trùng lặp khi mở nhiều tab.
  - Quản lý vòng đời bằng FastAPI lifespan, tự động kết nối lại khi mất sóng Wi-Fi.

Sử dụng:
    # Chạy mặc định (IP 10.62.83.32, GPU 0, cổng 8000):
    python scripts/esp32_dashboard.py

    # Tùy chỉnh tham số:
    python scripts/esp32_dashboard.py --ip 10.62.83.32 --conf 0.20 --accept 0.75 --device 0 --port 8000
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
import io
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from ultralytics import YOLO

# Đảm bảo hiển thị UTF-8 trên Windows console
if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─── Đường dẫn và thiết lập ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import torch

from src.telegram_service import TelegramBotService
from src.cloud_client import CloudInferenceClient, CloudInferenceResult
from src.metrics import (
    BENCHMARK_SCENARIOS,
    BenchmarkSession,
    LatencyRecord,
    PerformanceTracker,
)
from src.reliability import (
    RELIABILITY_SCENARIOS,
    ReliabilityBenchmarkSession,
    ReliabilityTracker,
)


DEFAULT_MODEL_PATH = (
    ROOT / "runs" / "detect" / "results" / "train" / "antigravity_gpu_v1" / "weights" / "best.pt"
)

# Cấu hình ESP32-CAM và Cloud Inference
DEFAULT_ESP32_IP = os.getenv("ESP32_IP", "10.62.83.32").strip()
DEFAULT_INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").strip().lower()
DEFAULT_CLOUD_API_URL = os.getenv("CLOUD_API_URL", "http://127.0.0.1:9000/v1/infer").strip()
DEFAULT_CLOUD_TIMEOUT = float(os.getenv("CLOUD_SOCKET_TIMEOUT", "4.0"))

# Nạp cấu hình ngưỡng tối ưu đã được hiệu chỉnh (Calibration)
CONFIG_PATH = ROOT / "configs" / "thresholds.json"
DEFAULT_CONF_FLOOR = 0.20
DEFAULT_ACCEPT_THRESH = 0.75

if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as _f_cfg:
            _th_cfg = json.load(_f_cfg)
            DEFAULT_CONF_FLOOR = float(_th_cfg.get("conf_floor", DEFAULT_CONF_FLOOR))
            DEFAULT_ACCEPT_THRESH = float(_th_cfg.get("accept_thresh", DEFAULT_ACCEPT_THRESH))
    except Exception:
        pass

# ─── Worker chia sẻ chạy ngầm (Single Pipeline Worker) ─────────────────────────
class ESP32PipelineWorker:
    """
    Worker đơn đảm nhiệm toàn bộ việc lấy ảnh từ ESP32-CAM (/capture),
    chạy YOLO11n trên GPU (RTX 5060) và dựng sẵn frame MJPEG cho tất cả client.
    """

    def __init__(
        self,
        ip: str,
        model_path: Path,
        conf_thresh: float = DEFAULT_CONF_FLOOR,
        accept_thresh: float = DEFAULT_ACCEPT_THRESH,
        device: str = "0",
        replay_dir: Optional[Path] = None,
        inference_mode: str = DEFAULT_INFERENCE_MODE,
        cloud_url: str = DEFAULT_CLOUD_API_URL,
        cloud_timeout: float = DEFAULT_CLOUD_TIMEOUT,
    ):
        self.ip = ip.strip()
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.accept_thresh = accept_thresh
        self.device = device
        self.replay_dir = replay_dir
        self.replay_images = []
        self.replay_index = 0
        if self.replay_dir and Path(self.replay_dir).exists():
            exts = (".jpg", ".jpeg", ".png")
            self.replay_images = sorted([
                p for p in Path(self.replay_dir).iterdir() if p.suffix.lower() in exts
            ])
            print(f"[*] Replay Mode kích hoạt: {len(self.replay_images)} ảnh từ {self.replay_dir}")

        if not self.ip.startswith("http://") and not self.ip.startswith("https://"):
            self.capture_url = f"http://{self.ip}/capture"
            self.flash_url = f"http://{self.ip}/flash"
        else:
            base_url = self.ip.rstrip("/")
            self.capture_url = f"{base_url}/capture"
            self.flash_url = f"{base_url}/flash"

        self.model: Optional[YOLO] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Khóa an toàn đồng bộ đa luồng
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.stop_event = threading.Event()

        # Quản lý chế độ suy luận: Local (GPU RTX 5060) vs Cloud (AWS EC2 CPU)
        self.inference_mode = inference_mode.lower()
        self.cloud_client = CloudInferenceClient(api_url=cloud_url, timeout=cloud_timeout)
        self.cloud_status = "idle"  # "idle", "online", "error", "timeout"
        self.cloud_last_error = ""
        self.cloud_timing = {
            "cloud_inference_ms": 0.0,
            "cloud_server_ms": 0.0,
            "cloud_decode_ms": 0.0,
            "cloud_predict_wall_ms": 0.0,
            "cloud_request_response_ms": 0.0,
            "cloud_transport_overhead_est_ms": 0.0,
        }
        self.mode_lock = threading.Lock()

        # Giám sát hiệu năng và Benchmark
        self.is_cuda = (str(self.device) != "cpu" and torch.cuda.is_available())
        self.tracker = PerformanceTracker(window_size=300)
        self.active_benchmark: Optional[BenchmarkSession] = None
        self.benchmark_lock = threading.Lock()

        # Giám sát độ tin cậy và phục hồi kết nối (Reliability & Recovery)
        self.reliability_tracker = ReliabilityTracker(
            initial_backoff=0.5,
            max_backoff=5.0,
            backoff_factor=1.5,
            stale_threshold_seconds=4.0,
        )
        self.active_reliability_benchmark: Optional[ReliabilityBenchmarkSession] = None
        self.reliability_lock = threading.Lock()
        self.socket_timeout = 3.0
        self.http_session: Optional[requests.Session] = None
        self.active_client_connections = 0

        # Dữ liệu trạng thái chia sẻ
        self.latest_jpeg: Optional[bytes] = None
        self.camera_status = "connecting"  # "connected", "reconnecting", "error"
        self.last_error = ""
        self.fps = 0.0
        self.counts = {"mask": 0, "no_mask": 0, "unknown": 0, "total": 0}
        self.detections = []
        self.last_capture_time = 0.0
        self.total_frames_processed = 0

        # Theo dõi End-to-End Display Latency bằng Browser Acknowledgment
        self.pending_ack_frames: Dict[str, Dict[str, float]] = {}
        self.latest_frame_id: str = ""
        self.latest_capture_start_perf: float = 0.0
        self.latest_capture_start_epoch: float = 0.0
        self.latest_render_ack_ms: float = 0.0
        self.ack_history = []
        self.ack_lock = threading.Lock()

    def record_frame_ack(self, frame_id: str) -> Optional[float]:
        """Ghi nhận ACK từ browser khi frame được giải mã và đưa vào chu kỳ render (requestAnimationFrame)."""
        t_ack = time.perf_counter()
        with self.ack_lock:
            frame_info = self.pending_ack_frames.pop(frame_id, None)
            if frame_info:
                render_ack_ms = (t_ack - frame_info["capture_start_perf"]) * 1000.0
                self.latest_render_ack_ms = render_ack_ms
                self.ack_history.append(render_ack_ms)
                if len(self.ack_history) > 30:
                    self.ack_history.pop(0)
                return render_ack_ms
        return None

    def set_inference_mode(self, mode: str) -> Tuple[bool, str]:
        """Chuyển đổi an toàn giữa Local (GPU RTX 5060) và Cloud (AWS EC2 CPU)."""
        clean_mode = str(mode).strip().lower()
        if clean_mode not in ("local", "cloud"):
            return False, f"Chế độ không hợp lệ: '{mode}'. Chỉ chấp nhận 'local' hoặc 'cloud'."

        with self.mode_lock:
            if clean_mode == self.inference_mode:
                return True, f"Hệ thống đã ở chế độ {clean_mode.upper()}."

            if clean_mode == "cloud":
                # Kiểm tra kết nối Cloud trước khi chuyển
                is_healthy, _, health_msg = self.cloud_client.health_check(timeout=2.0)
                if not is_healthy:
                    print(f"[!] Cảnh báo chuyển Cloud mode: {health_msg}")
                    self.cloud_status = "error"
                    self.cloud_last_error = health_msg
                else:
                    self.cloud_status = "online"
                    self.cloud_last_error = ""
                self.inference_mode = "cloud"
                print(f"[*] Đã chuyển sang chế độ CLOUD (AWS EC2 CPU) tại {self.cloud_client.api_url}")
                return True, f"Đã chuyển sang chế độ CLOUD (AWS EC2 CPU). Trạng thái Cloud: {self.cloud_status}"
            else:
                self.inference_mode = "local"
                if self.model is None:
                    try:
                        self._load_local_model()
                    except Exception as e:
                        print(f"[!] Lỗi nạp model local: {e}")
                        return False, f"Không thể chuyển sang LOCAL: {e}"
                print(f"[*] Đã chuyển sang chế độ LOCAL (NVIDIA RTX 5060, device={self.device})")
                return True, f"Đã chuyển sang chế độ LOCAL (NVIDIA RTX 5060, device={self.device})"

    def _load_local_model(self):
        """Nạp model YOLO11n local theo cơ chế lazy-load."""
        if self.model is not None:
            return
        print(f"[*] Đang nạp model YOLO11n từ: {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy model tại: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        print(f"[*] Model đã nạp thành công. Thiết bị inference: {self.device}")

    def start(self):
        """Khởi động worker trong một thread nền duy nhất."""
        if self.running:
            return

        if self.inference_mode == "local":
            self._load_local_model()
        else:
            print(f"[*] Khởi động ở chế độ CLOUD: bỏ qua nạp model local (lazy-load khi chuyển mode)")

        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="ESP32WorkerThread")
        self.thread.start()
        print(f"[*] Pipeline worker đã bắt đầu. URL nguồn: {self.capture_url} | Chế độ: {self.inference_mode.upper()}")

    def stop(self):
        """Dừng worker an toàn và giải phóng socket ngay lập tức."""
        self.running = False
        self.stop_event.set()
        if self.http_session:
            try:
                self.http_session.close()
            except Exception:
                pass
        if self.cloud_client:
            self.cloud_client.close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        print("[*] Pipeline worker đã dừng an toàn.")

    def _create_placeholder_frame(self, message: str, subtext: str = "") -> bytes:
        """Tạo ảnh thông báo trạng thái khi mất kết nối hoặc đang khởi động."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (24, 24, 30)  # Nền tối thanh lịch

        # Viền cảnh báo
        cv2.rectangle(img, (10, 10), (630, 470), (45, 45, 60), 2)

        # Tiêu đề lỗi / trạng thái
        cv2.putText(
            img,
            message,
            (40, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

        if subtext:
            cv2.putText(
                img,
                subtext,
                (40, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 190),
                1,
                cv2.LINE_AA,
            )

        ts_text = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            img,
            f"Thoi gian: {ts_text} | IP: {self.ip}",
            (40, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 120, 130),
            1,
            cv2.LINE_AA,
        )

        ret, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes() if ret else b""

    def _run_loop(self):
        """Vòng lặp lấy ảnh từ camera, suy luận và cập nhật bộ đệm kèm đo đạc chi tiết."""
        self.http_session = requests.Session()
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=4, max_retries=0)
        self.http_session.mount("http://", adapter)
        self.http_session.mount("https://", adapter)
        http_headers = {
            "Connection": "keep-alive",
            "Accept": "image/jpeg, */*",
            "User-Agent": "ESP32-YOLO-Pipeline/2.0",
        }
        fps_history = []

        while self.running and not self.stop_event.is_set():
            t_loop_start = time.perf_counter()
            t_capture_epoch = time.time()
            current_frame_id = f"frame_{self.total_frames_processed + 1}"
            capture_http_ms = 0.0
            decode_ms = 0.0
            inference_ms = 0.0
            render_ms = 0.0
            encode_ms = 0.0

            self.reliability_tracker.record_attempt()

            try:
                # 1. Đo HTTP capture request/response (Thời gian tải JPEG hoặc replay từ đĩa)
                t_cap_start = time.perf_counter()
                if self.replay_images:
                    replay_path = self.replay_images[self.replay_index % len(self.replay_images)]
                    self.replay_index += 1
                    with open(replay_path, "rb") as f:
                        content = f.read()
                else:
                    response = self.http_session.get(
                        self.capture_url,
                        headers=http_headers,
                        timeout=self.socket_timeout,
                    )
                    if response.status_code != 200:
                        raise ValueError(f"HTTP Status {response.status_code}")
                    content = response.content
                t_cap_end = time.perf_counter()
                capture_http_ms = (t_cap_end - t_cap_start) * 1000.0

                # 2. Đo JPEG decoding
                t_dec_start = time.perf_counter()
                img_array = np.frombuffer(content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Không thể giải mã JPEG từ ESP32-CAM")
                t_dec_end = time.perf_counter()
                decode_ms = (t_dec_end - t_dec_start) * 1000.0

                # 3. Phân nhánh suy luận: Local (GPU RTX 5060) vs Cloud (AWS EC2 CPU)
                raw_boxes = []
                is_cloud_error = False
                cloud_error_detail = ""
                cloud_res = None
                cloud_rtt_ms = 0.0

                with self.mode_lock:
                    current_mode = self.inference_mode

                if current_mode == "cloud":
                    # CLOUD MODE: Gửi JPEG lên AWS EC2, TUYỆT ĐỐI KHÔNG gọi YOLO GPU local
                    cloud_res = self.cloud_client.infer(
                        content, frame_id=f"frame_{self.total_frames_processed + 1}"
                    )
                    cloud_rtt_ms = cloud_res.cloud_request_response_ms

                    if not cloud_res.success:
                        is_cloud_error = True
                        cloud_error_detail = cloud_res.error_msg
                        self.cloud_status = "error"
                        self.cloud_last_error = cloud_error_detail
                    else:
                        self.cloud_status = "online"
                        self.cloud_last_error = ""
                        inference_ms = cloud_res.cloud_inference_ms
                        self.cloud_timing = {
                            "cloud_inference_ms": cloud_res.cloud_inference_ms,
                            "cloud_server_ms": cloud_res.cloud_server_ms,
                            "cloud_decode_ms": cloud_res.cloud_decode_ms,
                            "cloud_predict_wall_ms": cloud_res.cloud_predict_wall_ms,
                            "cloud_request_response_ms": cloud_res.cloud_request_response_ms,
                            "cloud_transport_overhead_est_ms": cloud_res.cloud_transport_overhead_est_ms,
                        }
                        for det in cloud_res.raw_detections:
                            raw_boxes.append({
                                "cls_id": int(det.get("class_id", 0)),
                                "conf": float(det.get("confidence", 0.0)),
                                "name": str(det.get("label") or det.get("raw_class_name") or "no_mask"),
                                "bbox_xyxy": det.get("bbox_xyxy", [0, 0, 0, 0]),
                            })
                else:
                    # LOCAL MODE: Chạy YOLO11n trên GPU RTX 5060 laptop
                    if self.model is None:
                        self._load_local_model()

                    t_infer_start = time.perf_counter()
                    if self.is_cuda:
                        torch.cuda.synchronize()

                    results = self.model.predict(
                        source=frame,
                        imgsz=640,
                        conf=self.conf_thresh,
                        device=self.device,
                        verbose=False,
                    )[0]

                    if self.is_cuda:
                        torch.cuda.synchronize()
                    t_infer_end = time.perf_counter()
                    inference_ms = (t_infer_end - t_infer_start) * 1000.0

                    if hasattr(results, "boxes") and results.boxes is not None:
                        for box in results.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            predicted_name = results.names.get(cls_id, f"cls_{cls_id}")
                            raw_boxes.append({
                                "cls_id": cls_id,
                                "conf": conf,
                                "name": predicted_name,
                                "bbox_xyxy": box.xyxy[0].tolist(),
                            })

                # 4. Xử lý hiển thị Bounding Box & nhãn phân loại (Dùng chung bộ lọc & renderer)
                t_ren_start = time.perf_counter()
                local_counts = {"mask": 0, "no_mask": 0, "unknown": 0}
                local_detections = []
                h, w = frame.shape[:2]

                if is_cloud_error:
                    # Lỗi Cloud: Hiển thị banner cảnh báo rõ ràng, KHÔNG tự động chuyển Local
                    cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 160), -1)
                    cv2.putText(
                        frame,
                        f"[CLOUD DISCONNECTED] {cloud_error_detail[:45]}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    for item in raw_boxes:
                        cls_id = item["cls_id"]
                        conf = item["conf"]
                        predicted_name = item["name"]

                        # Áp dụng ngưỡng accept_thresh đồng bộ
                        if conf < self.accept_thresh:
                            label = "unknown"
                            color = (130, 130, 130)  # Xám bạc
                        elif predicted_name == "mask":
                            label = "mask"
                            color = (0, 200, 0)      # Xanh lá
                        else:
                            label = "no_mask"
                            color = (0, 0, 240)      # Đỏ tươi

                        local_counts[label] += 1
                        coords = [int(v) for v in item["bbox_xyxy"]]
                        if len(coords) == 4:
                            x1, y1, x2, y2 = coords
                            x1 = max(0, min(x1, w))
                            y1 = max(0, min(y1, h))
                            x2 = max(0, min(x2, w))
                            y2 = max(0, min(y2, h))
                        else:
                            continue

                        local_detections.append(
                            {
                                "label": label,
                                "conf": round(conf, 3),
                                "box": [x1, y1, x2, y2],
                            }
                        )

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label_str = f"{label} {conf:.2f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = 0.55
                        thick = 2
                        (tw, th), baseline = cv2.getTextSize(label_str, font, scale, thick)
                        cv2.rectangle(
                            frame,
                            (x1, max(y1 - th - baseline - 4, 0)),
                            (x1 + tw + 6, max(y1, th + baseline + 4)),
                            color,
                            -1,
                        )
                        cv2.putText(
                            frame,
                            label_str,
                            (x1 + 3, max(y1 - baseline - 2, th)),
                            font,
                            scale,
                            (255, 255, 255),
                            thick - 1,
                            cv2.LINE_AA,
                        )

                    # Thanh trạng thái phía trên khung hình
                    cv2.rectangle(frame, (0, 0), (w, 34), (20, 20, 25), -1)
                    if current_mode == "cloud":
                        status_text = (
                            f"[CLOUD EC2] Mask: {local_counts['mask']} | "
                            f"No mask: {local_counts['no_mask']} | "
                            f"Unknown: {local_counts['unknown']} | "
                            f"Infer: {inference_ms:.1f}ms | RTT: {cloud_rtt_ms:.0f}ms"
                        )
                    else:
                        status_text = (
                            f"[LOCAL GPU] Mask: {local_counts['mask']} | "
                            f"No mask: {local_counts['no_mask']} | "
                            f"Unknown: {local_counts['unknown']} | "
                            f"Infer: {inference_ms:.1f}ms"
                        )
                    cv2.putText(
                        frame,
                        status_text,
                        (10, 23),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                total_faces = sum(local_counts.values())
                t_ren_end = time.perf_counter()
                render_ms = (t_ren_end - t_ren_start) * 1000.0

                # 5. Đo JPEG encoding (Tối ưu: quality=75, optimize=0 để giảm encode latency)
                t_enc_start = time.perf_counter()
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75, int(cv2.IMWRITE_JPEG_OPTIMIZE), 0]
                ret, jpeg_buf = cv2.imencode(".jpg", frame, encode_param)
                if not ret:
                    raise ValueError("Không thể nén ảnh JPEG")
                jpeg_bytes = jpeg_buf.tobytes()
                t_enc_end = time.perf_counter()
                encode_ms = (t_enc_end - t_enc_start) * 1000.0

                # 6. Tính toán các mốc thời gian tổng hợp
                t_loop_end = time.perf_counter()
                if current_mode == "cloud":
                    server_processing_ms = decode_ms + cloud_rtt_ms + render_ms + encode_ms
                else:
                    server_processing_ms = decode_ms + inference_ms + render_ms + encode_ms
                total_loop_ms = (t_loop_end - t_loop_start) * 1000.0
                instant_fps = 1000.0 / max(total_loop_ms, 0.001)

                fps_history.append(instant_fps)
                if len(fps_history) > 10:
                    fps_history.pop(0)
                smooth_fps = sum(fps_history) / len(fps_history)

                # Ghi nhận phục hồi kết nối nếu vừa trải qua gián đoạn
                rec_duration = self.reliability_tracker.record_success()
                if rec_duration is not None:
                    print(f"[*] [RECOVERY] Camera đã kết nối lại thành công sau {rec_duration:.2f}s! Tiếp tục inference.")

                # Ghi nhận bản ghi đo lường hiệu năng
                rec = LatencyRecord(
                    frame_index=self.total_frames_processed + 1,
                    timestamp=time.time(),
                    success=not is_cloud_error,
                    inference_mode=current_mode,
                    capture_http_ms=round(capture_http_ms, 2),
                    decode_ms=round(decode_ms, 2),
                    inference_ms=round(inference_ms, 2),
                    render_ms=round(render_ms, 2),
                    encode_ms=round(encode_ms, 2),
                    server_processing_ms=round(server_processing_ms, 2),
                    total_loop_ms=round(total_loop_ms, 2),
                    fps=round(instant_fps, 1),
                    mask_count=local_counts["mask"],
                    no_mask_count=local_counts["no_mask"],
                    unknown_count=local_counts["unknown"],
                    total_faces=total_faces,
                    cloud_inference_ms=round(self.cloud_timing.get("cloud_inference_ms", 0.0), 2) if current_mode == "cloud" else 0.0,
                    cloud_server_ms=round(self.cloud_timing.get("cloud_server_ms", 0.0), 2) if current_mode == "cloud" else 0.0,
                    cloud_decode_ms=round(self.cloud_timing.get("cloud_decode_ms", 0.0), 2) if current_mode == "cloud" else 0.0,
                    cloud_predict_wall_ms=round(self.cloud_timing.get("cloud_predict_wall_ms", 0.0), 2) if current_mode == "cloud" else 0.0,
                    cloud_request_response_ms=round(cloud_rtt_ms, 2) if current_mode == "cloud" else 0.0,
                    cloud_transport_overhead_est_ms=round(self.cloud_timing.get("cloud_transport_overhead_est_ms", 0.0), 2) if current_mode == "cloud" else 0.0,
                    e2e_display_ms=round(self.latest_render_ack_ms, 2) if self.latest_render_ack_ms > 0 else 0.0,
                    error_msg=f"Cloud Error: {cloud_error_detail}" if is_cloud_error else "",
                )
                self.tracker.record_frame(rec)

                # Cập nhật phiên benchmark hiệu năng nếu đang chạy
                with self.benchmark_lock:
                    if self.active_benchmark and self.active_benchmark.is_active:
                        finished = self.active_benchmark.add_record(rec)
                        if finished:
                            print(f"[*] Benchmark {self.active_benchmark.scenario_id} đã hoàn tất.")

                # Cập nhật phiên benchmark độ tin cậy nếu đang chạy
                with self.reliability_lock:
                    if self.active_reliability_benchmark and self.active_reliability_benchmark.is_active:
                        self.active_reliability_benchmark.record_sample(
                            camera_status="connected",
                            is_stale=False,
                            data_age_seconds=0.0,
                            active_connections=self.active_client_connections,
                            event_note=f"Frame OK ({instant_fps:.1f} FPS) [{current_mode.upper()}]",
                        )

                with self.lock:
                    self.latest_jpeg = jpeg_bytes
                    self.camera_status = "connected"
                    self.last_error = f"Cloud Error: {cloud_error_detail}" if is_cloud_error else ""
                    self.fps = round(smooth_fps, 1)
                    self.counts = {
                        "mask": local_counts["mask"],
                        "no_mask": local_counts["no_mask"],
                        "unknown": local_counts["unknown"],
                        "total": total_faces,
                    }
                    self.detections = local_detections
                    self.last_capture_time = time.time()
                    self.total_frames_processed += 1
                    self.new_frame_event.set()

                # Đăng ký frame_id vào bảng theo dõi ACK hiển thị
                with self.ack_lock:
                    self.latest_frame_id = current_frame_id
                    self.latest_capture_start_perf = t_loop_start
                    self.latest_capture_start_epoch = t_capture_epoch
                    self.pending_ack_frames[current_frame_id] = {
                        "capture_start_perf": t_loop_start,
                        "capture_start_epoch": t_capture_epoch,
                    }
                    if len(self.pending_ack_frames) > 60:
                        oldest_key = next(iter(self.pending_ack_frames))
                        self.pending_ack_frames.pop(oldest_key, None)

            except requests.Timeout as timeout_err:
                error_msg = f"HTTP Timeout (>{self.socket_timeout}s)"
                backoff_sleep = self.reliability_tracker.record_failure(error_msg, is_timeout=True)
                print(f"[!] Timeout kết nối camera ({self.ip}): {error_msg} -> Retry sau {backoff_sleep:.1f}s")
                self._handle_loop_error(error_msg, backoff_sleep)

            except (requests.RequestException, ValueError, Exception) as err:
                error_msg = str(err)
                backoff_sleep = self.reliability_tracker.record_failure(error_msg, is_timeout=False)
                print(f"[!] Lỗi kết nối camera ({self.ip}): {error_msg} -> Retry sau {backoff_sleep:.1f}s")
                self._handle_loop_error(error_msg, backoff_sleep)

        if self.http_session:
            try:
                self.http_session.close()
            except Exception:
                pass

    def _handle_loop_error(self, error_msg: str, backoff_sleep: float):
        """Xử lý lỗi kết nối: cập nhật bộ đếm, placeholder frame và ngủ ngắt được (interruptible)."""
        fail_rec = LatencyRecord(
            frame_index=self.total_frames_processed + 1,
            timestamp=time.time(),
            success=False,
            error_msg=error_msg,
        )
        self.tracker.record_frame(fail_rec)

        with self.benchmark_lock:
            if self.active_benchmark and self.active_benchmark.is_active:
                self.active_benchmark.add_record(fail_rec)

        is_stale, age, stale_reason = self.reliability_tracker.evaluate_staleness()

        with self.reliability_lock:
            if self.active_reliability_benchmark and self.active_reliability_benchmark.is_active:
                self.active_reliability_benchmark.record_sample(
                    camera_status="reconnecting",
                    is_stale=is_stale,
                    data_age_seconds=age,
                    active_connections=self.active_client_connections,
                    event_note=f"Lỗi: {error_msg[:45]} (Retry {backoff_sleep:.1f}s)",
                )

        placeholder = self._create_placeholder_frame(
            "ESP32-CAM Dang thu ket noi lai...",
            f"Loi: {error_msg[:55]} | Retry sau {backoff_sleep:.1f}s",
        )
        with self.lock:
            self.camera_status = "reconnecting"
            self.last_error = error_msg
            self.fps = 0.0
            self.latest_jpeg = placeholder
            self.counts = {"mask": 0, "no_mask": 0, "unknown": 0, "total": 0}
            self.detections = []
            self.new_frame_event.set()

        # Ngủ an toàn có thể bị đánh thức lập tức khi server dừng (stop_event)
        self.stop_event.wait(timeout=backoff_sleep)

    def get_status(self) -> Dict[str, Any]:
        """Lấy dữ liệu trạng thái hiện tại phục vụ API, bao gồm độ tươi mới và reliability."""
        rel_summary = self.reliability_tracker.get_summary()
        with self.lock:
            return {
                "camera_status": self.camera_status,
                "camera_ip": self.ip,
                "capture_url": self.capture_url,
                "flash_url": self.flash_url,
                "fps": self.fps,
                "counts": self.counts.copy(),
                "detections": list(self.detections),
                "device": str(self.device),
                "conf_floor": self.conf_thresh,
                "accept_thresh": self.accept_thresh,
                "last_error": self.last_error,
                "last_capture_time": self.last_capture_time,
                "total_frames": self.total_frames_processed,
                "active_client_connections": self.active_client_connections,
                # Cloud metadata & 3 loại latency của Lab 3
                "inference_mode": self.inference_mode,
                "cloud_status": self.cloud_status,
                "cloud_url": self.cloud_client.api_url if self.cloud_client else "",
                "cloud_last_error": self.cloud_last_error,
                "cloud_inference_ms": self.cloud_timing.get("cloud_inference_ms", 0.0),
                "cloud_request_response_ms": self.cloud_timing.get("cloud_request_response_ms", 0.0),
                "cloud_server_ms": self.cloud_timing.get("cloud_server_ms", 0.0),
                "cloud_transport_overhead_est_ms": self.cloud_timing.get("cloud_transport_overhead_est_ms", 0.0),
                "e2e_display_ms": round(self.latest_render_ack_ms, 1) if self.latest_render_ack_ms > 0 else 0.0,
                "e2e_display_avg_ms": round(sum(self.ack_history) / len(self.ack_history), 1) if self.ack_history else 0.0,
                "display_latency_method": "Browser ACK (requestAnimationFrame + HTTP POST ACK)",
                "display_latency_note": "Ước lượng đo từ khi Gateway bắt đầu lấy frame ESP32-CAM (/capture) đến khi server nhận ACK sau khi trình duyệt render frame trong requestAnimationFrame (bao gồm overhead gửi HTTP ACK, không phải thời điểm xuất hiện vật lý photon trên màn hình).",
                # Đánh giá dữ liệu cũ (Stale)
                "is_stale": rel_summary["is_stale"],
                "data_age_seconds": rel_summary["data_age_seconds"],
                "stale_reason": rel_summary["stale_reason"],
                "stale_threshold_seconds": rel_summary["stale_threshold_seconds"],
                # Tổng hợp độ tin cậy
                "reliability": rel_summary,
            }

    def toggle_flash(self, timeout: float = 4.0) -> Tuple[bool, str]:
        """Gọi endpoint /flash trên ESP32-CAM để bật/tắt đèn chiếu sáng."""
        try:
            res = requests.get(self.flash_url, timeout=timeout)
            if res.status_code == 200:
                resp_text = res.text.strip() or "Thao tác flash thành công."
                return True, resp_text
            return False, f"ESP32-CAM trả về HTTP {res.status_code}"
        except Exception as e:
            return False, f"Lỗi kết nối flash ({e})"

    def get_yolo_snapshot(
        self, stale_threshold: float = 5.0
    ) -> Tuple[bool, Dict[str, Any], Optional[bytes], str]:
        """
        Lấy trạng thái và frame JPEG mới nhất cho Telegram Bot hoặc API.
        Trả về: (is_fresh, status_dict, jpeg_bytes, reason)
        Từ chối cung cấp dữ liệu cũ nếu camera rớt mạng hoặc dữ liệu vượt ngưỡng stale_threshold.
        """
        status = self.get_status()
        with self.lock:
            if self.camera_status != "connected":
                return False, status, None, f"Camera đang ở trạng thái '{self.camera_status}'"

            age = time.time() - self.last_capture_time
            if self.last_capture_time <= 0 or age > stale_threshold:
                return (
                    False,
                    status,
                    None,
                    f"Dữ liệu cũ (cập nhật cách đây {age:.1f}s, vượt ngưỡng tươi mới {stale_threshold}s)",
                )

            if self.latest_jpeg is None:
                return False, status, None, "Chưa có khung hình JPEG nào sẵn sàng"

            return True, status, self.latest_jpeg, "Dữ liệu hợp lệ và thời gian thực"

    def start_benchmark(self, scenario_id: str, duration_seconds: float = 60.0) -> BenchmarkSession:
        """Khởi động phiên benchmark hiệu năng thực nghiệm theo kịch bản."""
        with self.benchmark_lock:
            if self.active_benchmark and self.active_benchmark.is_active:
                progress = self.active_benchmark.get_progress()
                raise ValueError(
                    f"Phiên benchmark '{self.active_benchmark.scenario_id}' đang chạy ({progress.get('remaining_seconds', 0)}s còn lại). Hãy dừng hoặc đợi hoàn tất."
                )
            session = BenchmarkSession(scenario_id=scenario_id, duration_seconds=duration_seconds)
            session.start()
            self.active_benchmark = session
            return session

    def stop_benchmark(self) -> Optional[Dict[str, Any]]:
        """Dừng benchmark trước hạn và lưu kết quả."""
        with self.benchmark_lock:
            if self.active_benchmark and self.active_benchmark.is_active:
                self.active_benchmark.stop()
                return self.active_benchmark.get_progress()
            return None

    def get_benchmark_status(self) -> Dict[str, Any]:
        """Lấy trạng thái và tiến độ phiên benchmark."""
        with self.benchmark_lock:
            if self.active_benchmark:
                return self.active_benchmark.get_progress()
            return {"is_active": False, "completed": False, "message": "Chưa có phiên benchmark nào đang chạy"}

    def start_reliability_benchmark(self, scenario_id: str, duration_seconds: float = 30.0) -> ReliabilityBenchmarkSession:
        """Khởi động phiên kiểm thử độ tin cậy và phục hồi kết nối (R1–R6)."""
        with self.reliability_lock:
            if self.active_reliability_benchmark and self.active_reliability_benchmark.is_active:
                progress = self.active_reliability_benchmark.get_progress()
                raise ValueError(
                    f"Phiên kiểm thử '{self.active_reliability_benchmark.scenario_id}' đang chạy ({progress.get('remaining_seconds', 0)}s còn lại). Hãy dừng trước khi chạy kịch bản mới."
                )
            session = ReliabilityBenchmarkSession(
                scenario_id=scenario_id,
                duration_seconds=duration_seconds,
                tracker=self.reliability_tracker,
            )
            session.start()
            self.active_reliability_benchmark = session
            return session

    def stop_reliability_benchmark(self) -> Optional[Dict[str, Any]]:
        """Dừng phiên kiểm thử độ tin cậy."""
        with self.reliability_lock:
            if self.active_reliability_benchmark and self.active_reliability_benchmark.is_active:
                self.active_reliability_benchmark.stop()
                return self.active_reliability_benchmark.get_progress()
            return None

    def get_reliability_benchmark_status(self) -> Dict[str, Any]:
        """Lấy tiến độ phiên kiểm thử độ tin cậy."""
        with self.reliability_lock:
            if self.active_reliability_benchmark:
                return self.active_reliability_benchmark.get_progress()
            return {"is_active": False, "completed": False, "message": "Chưa có phiên kiểm thử độ tin cậy nào đang chạy"}

    async def stream_generator(self):
        """Generator sinh ra luồng multipart MJPEG cho mỗi client kết nối."""
        self.active_client_connections += 1
        last_frame_bytes = None
        try:
            while self.running and not self.stop_event.is_set():
                with self.lock:
                    frame_bytes = self.latest_jpeg

                with self.ack_lock:
                    frame_id = self.latest_frame_id
                    capture_start_epoch = self.latest_capture_start_epoch

                if frame_bytes is not None and frame_bytes != last_frame_bytes:
                    last_frame_bytes = frame_bytes
                    header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame_bytes)).encode("ascii") + b"\r\n"
                        b"X-Frame-ID: " + frame_id.encode("ascii") + b"\r\n"
                        b"X-Capture-Start-Ms: " + f"{capture_start_epoch * 1000.0:.1f}".encode("ascii") + b"\r\n\r\n"
                    )
                    yield (header + frame_bytes + b"\r\n")
                # Nghỉ ngắn tránh chiếm 100% CPU vòng lặp async
                await asyncio.sleep(0.03)
        finally:
            self.active_client_connections = max(0, self.active_client_connections - 1)



# ─── FastAPI App & Lifespan ───────────────────────────────────────────────────
global_worker: Optional[ESP32PipelineWorker] = None
global_telegram_service: Optional[TelegramBotService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi động worker chung khi FastAPI start
    if global_worker:
        global_worker.start()

    # Khởi động Telegram Bot poller nếu được cấu hình
    if global_telegram_service:
        global_telegram_service.start()

    yield

    # Tắt Telegram Bot trước
    if global_telegram_service:
        global_telegram_service.stop()

    # Tắt worker YOLO
    if global_worker:
        global_worker.stop()


app = FastAPI(
    title="ESP32-CAM YOLO11n Face Mask Detection Dashboard",
    description="Web dashboard giám sát trực tiếp luồng camera ESP32 và đếm số khẩu trang",
    lifespan=lifespan,
)


@app.get("/api/status")
def get_api_status() -> JSONResponse:
    """Trả về dữ liệu JSON chứa số đếm, FPS và tình trạng camera."""
    if not global_worker:
        return JSONResponse({"status": "worker_not_initialized"})
    return JSONResponse(global_worker.get_status())


@app.get("/video")
async def get_video_stream():
    """Luồng video MJPEG truyền trực tiếp tới client."""
    if not global_worker:
        return HTMLResponse("Worker not ready", status_code=503)
    return StreamingResponse(
        global_worker.stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/frame_ack")
async def handle_frame_ack(request: Request) -> JSONResponse:
    """Xác nhận frame đã được giải mã và render trên chu kỳ hiển thị của trình duyệt (requestAnimationFrame)."""
    if not global_worker:
        return JSONResponse({"ok": False, "message": "Worker chưa sẵn sàng"}, status_code=503)
    try:
        data = await request.json()
        frame_id = str(data.get("frame_id", "")).strip()
        if not frame_id:
            return JSONResponse({"ok": False, "message": "Thiếu frame_id"}, status_code=400)
        measured_ms = global_worker.record_frame_ack(frame_id)
        return JSONResponse({
            "ok": True,
            "frame_id": frame_id,
            "render_ack_ms": round(measured_ms, 1) if measured_ms is not None else None,
        })
    except Exception as e:
        return JSONResponse({"ok": False, "message": str(e)}, status_code=400)



@app.api_route("/api/flash", methods=["GET", "POST"])
def trigger_flash_api() -> JSONResponse:
    """Kích hoạt endpoint /flash trên ESP32-CAM."""
    if not global_worker:
        return JSONResponse({"ok": False, "message": "Worker chưa sẵn sàng"}, status_code=503)
    success, msg = global_worker.toggle_flash()
    return JSONResponse({"ok": success, "message": msg})


@app.get("/api/metrics")
def get_metrics_endpoint() -> JSONResponse:
    """Trả về thống kê độ trễ chi tiết các công đoạn, FPS thực tế và trạng thái reliability."""
    if not global_worker:
        return JSONResponse({"status": "error", "message": "Worker chưa sẵn sàng"}, status_code=503)
    rel_summary = global_worker.reliability_tracker.get_summary()
    return JSONResponse(
        global_worker.tracker.get_metrics_api_data(
            global_worker.camera_status,
            rel_summary,
            inference_mode=global_worker.inference_mode,
        )
    )


@app.get("/api/reliability")
def get_reliability_endpoint() -> JSONResponse:
    """Trả về tổng hợp các chỉ số độ tin cậy, retry backoff và thời gian phục hồi."""
    if not global_worker:
        return JSONResponse({"status": "error", "message": "Worker chưa sẵn sàng"}, status_code=503)
    return JSONResponse(global_worker.reliability_tracker.get_summary())


@app.post("/api/benchmark/start")
async def start_benchmark_endpoint(
    request: Request,
    scenario: Optional[str] = None,
    duration: Optional[float] = None,
) -> JSONResponse:
    """Khởi động phiên benchmark hiệu năng thực nghiệm (hỗ trợ cả query param và JSON body)."""
    if not global_worker:
        return JSONResponse({"status": "error", "message": "Worker chưa sẵn sàng"}, status_code=503)

    # Đọc body dự phòng nếu client gửi JSON payload thay vì query params
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    target_scenario = (scenario or body.get("scenario") or "S1").strip().upper()
    target_duration = float(duration if duration is not None else body.get("duration", 60.0))

    try:
        session = global_worker.start_benchmark(target_scenario, target_duration)
        return JSONResponse({
            "status": "ok",
            "message": f"Đã bắt đầu benchmark kịch bản {session.scenario_id}",
            "progress": session.get_progress(),
        })
    except ValueError as val_err:
        return JSONResponse({"status": "error", "message": str(val_err)}, status_code=409)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


@app.get("/api/benchmark/status")
def get_benchmark_status_endpoint() -> JSONResponse:
    """Lấy tiến độ phiên benchmark."""
    if not global_worker:
        return JSONResponse({"status": "error", "message": "Worker chưa sẵn sàng"}, status_code=503)
    return JSONResponse(global_worker.get_benchmark_status())


@app.post("/api/benchmark/stop")
def stop_benchmark_endpoint() -> JSONResponse:
    """Dừng phiên benchmark hiện tại."""
    if not global_worker:
        return JSONResponse({"status": "error", "message": "Worker chưa sẵn sàng"}, status_code=503)
    result = global_worker.stop_benchmark()
    return JSONResponse({"status": "ok", "message": "Đã dừng benchmark", "result": result})


@app.post("/api/reliability/start")
async def start_reliability_endpoint(
    request: Request,
    scenario: Optional[str] = None,
    duration: Optional[float] = None,
) -> JSONResponse:
    """Khởi động phiên kiểm thử độ tin cậy và phục hồi kết nối (R1–R6)."""
    if not global_worker:
        return JSONResponse({"status": "error", "message": "Worker chưa sẵn sàng"}, status_code=503)

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    target_scenario = (scenario or body.get("scenario") or "R1").strip().upper()
    target_duration = float(duration if duration is not None else body.get("duration", 30.0))

    try:
        session = global_worker.start_reliability_benchmark(target_scenario, target_duration)
        return JSONResponse({
            "status": "ok",
            "message": f"Đã bắt đầu kiểm thử độ tin cậy kịch bản {session.scenario_id}",
            "progress": session.get_progress(),
        })
    except ValueError as val_err:
        return JSONResponse({"status": "error", "message": str(val_err)}, status_code=409)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


@app.get("/api/reliability/status")
def get_reliability_status_endpoint() -> JSONResponse:
    """Lấy tiến độ phiên kiểm thử độ tin cậy."""
    if not global_worker:
        return JSONResponse({"status": "error", "message": "Worker chưa sẵn sàng"}, status_code=503)
    return JSONResponse(global_worker.get_reliability_benchmark_status())


@app.post("/api/reliability/stop")
def stop_reliability_endpoint() -> JSONResponse:
    """Dừng phiên kiểm thử độ tin cậy hiện tại."""
    if not global_worker:
        return JSONResponse({"status": "error", "message": "Worker chưa sẵn sàng"}, status_code=503)
    result = global_worker.stop_reliability_benchmark()
    return JSONResponse({"status": "ok", "message": "Đã dừng kiểm thử độ tin cậy", "result": result})



# ─── Giao diện Web Dashboard (HTML + CSS + JS) ────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ESP32-CAM Mask Detection Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090a0f;
      --card-bg: rgba(18, 20, 29, 0.75);
      --border: rgba(255, 255, 255, 0.08);
      --text: #f0f2f5;
      --text-muted: #8c93a4;
      --green: #10b981;
      --green-glow: rgba(16, 185, 129, 0.25);
      --red: #ef4444;
      --red-glow: rgba(239, 68, 68, 0.25);
      --amber: #f59e0b;
      --amber-glow: rgba(245, 158, 11, 0.25);
      --blue: #3b82f6;
      --blue-glow: rgba(59, 130, 246, 0.25);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: radial-gradient(circle at 50% -20%, #171b2c 0%, var(--bg) 60%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      padding: 1.25rem 2rem;
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(12px);
      background: rgba(10, 12, 18, 0.6);
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 50;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }

    .brand-icon {
      width: 40px;
      height: 40px;
      border-radius: 10px;
      background: linear-gradient(135deg, #2563eb, #10b981);
      display: grid;
      place-items: center;
      box-shadow: 0 0 15px rgba(37, 99, 235, 0.4);
    }

    .brand h1 {
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .brand p {
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    .header-badges {
      display: flex;
      gap: 0.75rem;
      align-items: center;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--text-muted);
    }

    .status-connected .status-dot {
      background: var(--green);
      box-shadow: 0 0 8px var(--green);
      animation: pulse 2s infinite;
    }

    .status-reconnecting .status-dot {
      background: var(--amber);
      box-shadow: 0 0 8px var(--amber);
      animation: pulse 1s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.9); }
    }

    main {
      flex: 1;
      padding: 1.75rem 2rem;
      max-width: 1440px;
      margin: 0 auto;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }

    /* Cards Grid */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1.25rem;
    }

    .stat-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.25rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      position: relative;
      overflow: hidden;
      backdrop-filter: blur(8px);
      transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .stat-card:hover {
      transform: translateY(-2px);
      border-color: rgba(255, 255, 255, 0.16);
    }

    .stat-card::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
    }

    .card-mask::after { background: var(--green); }
    .card-nomask::after { background: var(--red); }
    .card-unknown::after { background: var(--amber); }
    .card-total::after { background: var(--blue); }

    .stat-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--text-muted);
      font-size: 0.85rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .stat-val {
      font-size: 2.5rem;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
      line-height: 1;
    }

    .card-mask .stat-val { color: var(--green); text-shadow: 0 0 20px var(--green-glow); }
    .card-nomask .stat-val { color: var(--red); text-shadow: 0 0 20px var(--red-glow); }
    .card-unknown .stat-val { color: var(--amber); text-shadow: 0 0 20px var(--amber-glow); }
    .card-total .stat-val { color: var(--blue); text-shadow: 0 0 20px var(--blue-glow); }

    .stat-footer {
      font-size: 0.78rem;
      color: var(--text-muted);
    }

    /* Content Layout */
    .content-grid {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 1.5rem;
    }

    @media (max-width: 1024px) {
      .content-grid { grid-template-columns: 1fr; }
    }

    .video-panel {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 18px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }

    .panel-header {
      padding: 0.9rem 1.25rem;
      background: rgba(255, 255, 255, 0.03);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.9rem;
      font-weight: 600;
    }

    .video-wrapper {
      position: relative;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 480px;
    }

    .video-wrapper img {
      width: 100%;
      height: auto;
      max-height: 640px;
      object-fit: contain;
      display: block;
    }

    .video-overlay-fps {
      position: absolute;
      bottom: 12px;
      right: 12px;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(6px);
      padding: 0.35rem 0.75rem;
      border-radius: 8px;
      font-size: 0.8rem;
      font-family: 'JetBrains Mono', monospace;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .info-panel {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .info-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
    }

    .info-card h3 {
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--text);
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.6rem;
    }

    .info-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.82rem;
    }

    .info-label { color: var(--text-muted); }
    .info-value {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
      color: var(--text);
    }

    .detection-list {
      max-height: 220px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .det-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.45rem 0.75rem;
      background: rgba(255, 255, 255, 0.03);
      border-radius: 8px;
      font-size: 0.8rem;
      border-left: 3px solid var(--text-muted);
    }

    .det-mask { border-left-color: var(--green); }
    .det-nomask { border-left-color: var(--red); }
    .det-unknown { border-left-color: var(--amber); }

    /* Performance & Benchmark Styles */
    .perf-grid {
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 1.5rem;
    }
    @media (max-width: 1024px) {
      .perf-grid { grid-template-columns: 1fr; }
    }
    .perf-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 18px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .perf-body {
      padding: 1.25rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .perf-stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 0.75rem;
    }
    .perf-metric {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.75rem 0.85rem;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    .perf-lbl {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
    }
    .perf-val {
      font-size: 1.35rem;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text);
    }
    .perf-sub {
      font-size: 0.7rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }
    .perf-breakdown {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.5rem;
      background: rgba(0, 0, 0, 0.25);
      border-radius: 10px;
      padding: 0.75rem 1rem;
      border: 1px solid var(--border);
      font-size: 0.78rem;
    }
    .bk-item {
      display: flex;
      justify-content: space-between;
      color: var(--text-muted);
    }
    .bk-item strong {
      color: var(--text);
      font-family: 'JetBrains Mono', monospace;
    }

    footer {
      padding: 1.25rem 2rem;
      text-align: center;
      font-size: 0.75rem;
      color: var(--text-muted);
      border-top: 1px solid var(--border);
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M9 12l2 2 4-4"></path>
          <path d="M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"></path>
        </svg>
      </div>
      <div>
        <h1>ESP32-CAM Face Mask Detection</h1>
        <p>Hệ thống giám sát đa khuôn mặt • YOLO11n + RTX 5060</p>
      </div>
    </div>

    <div class="header-badges">
      <div id="staleBadge" class="badge" style="display: none; background: rgba(239, 68, 68, 0.2); border-color: var(--red); color: var(--red); font-weight: 700;">
        ⚠️ DỮ LIỆU CŨ (STALE)
      </div>
      <div id="cameraStatusBadge" class="badge status-connecting">
        <span class="status-dot"></span>
        <span id="cameraStatusText">CONNECTING</span>
      </div>
      <div class="badge" style="font-family: 'JetBrains Mono';">
        <span style="color: var(--text-muted)">IP:</span> <span id="cameraIpText">192.168.1.153</span>
      </div>
    </div>
  </header>

  <main>
    <!-- Counter Cards -->
    <div class="stats-grid">
      <div class="stat-card card-mask">
        <div class="stat-header">
          <span>Đeo khẩu trang</span>
          <span style="color: var(--green); font-size: 1.1rem;">✓</span>
        </div>
        <div id="countMask" class="stat-val">0</div>
        <div class="stat-footer">Phát hiện chuẩn quy định</div>
      </div>

      <div class="stat-card card-nomask">
        <div class="stat-header">
          <span>Không khẩu trang</span>
          <span style="color: var(--red); font-size: 1.1rem;">✕</span>
        </div>
        <div id="countNoMask" class="stat-val">0</div>
        <div class="stat-footer">Cảnh báo vi phạm</div>
      </div>

      <div class="stat-card card-unknown">
        <div class="stat-header">
          <span>Chưa rõ / Nghi ngờ</span>
          <span style="color: var(--amber); font-size: 1.1rem;">?</span>
        </div>
        <div id="countUnknown" class="stat-val">0</div>
        <div class="stat-footer">Độ tin cậy &lt; 60%</div>
      </div>

      <div class="stat-card card-total">
        <div class="stat-header">
          <span>Tổng số mặt</span>
          <span style="color: var(--blue); font-size: 1.1rem;">∑</span>
        </div>
        <div id="countTotal" class="stat-val">0</div>
        <div class="stat-footer">Trong khung hình hiện tại</div>
      </div>
    </div>

    <!-- Main Content Area -->
    <div class="content-grid">
      <!-- Video Live Stream -->
      <div class="video-panel">
        <div class="panel-header">
          <span>Luồng trực tiếp (Live Stream MJPEG)</span>
          <span style="font-size: 0.78rem; color: var(--green); font-weight: 600;">● REALTIME</span>
        </div>
        <div class="video-wrapper">
          <canvas id="videoCanvas" style="width: 100%; height: auto; max-height: 640px; object-fit: contain; display: block;"></canvas>
          <img id="mjpegStream" src="" alt="ESP32-CAM Stream" style="display: none; width: 100%; height: auto; max-height: 640px; object-fit: contain;">
          <div class="video-overlay-fps">
            <span style="color: var(--text-muted);">FPS:</span> <span id="fpsOverlay">0.0</span>
          </div>
        </div>
      </div>

      <!-- Info & Detection Panel -->
      <div class="info-panel">
        <div class="info-card">
          <h3>Thông số hệ thống</h3>
          <div class="info-row">
            <span class="info-label">AI Model</span>
            <span class="info-value">YOLO11n Face Mask</span>
          </div>
          <div class="info-row">
            <span class="info-label">Chế độ / Thiết bị</span>
            <span id="infoInferDevice" class="info-value" style="font-weight: 700;">YOLO11n</span>
          </div>
          <div class="info-row" id="rowCloudInfer" style="display: none;">
            <span class="info-label">1. Cloud Inference</span>
            <span id="infoCloudInfer" class="info-value" style="font-family: 'JetBrains Mono'; color: var(--green);">-- ms</span>
          </div>
          <div class="info-row" id="rowCloudRTT" style="display: none;">
            <span class="info-label">2. Local–Cloud Response</span>
            <span id="infoCloudRTT" class="info-value" style="font-family: 'JetBrains Mono'; color: #38bdf8;">-- ms</span>
          </div>
          <div class="info-row" id="rowE2EDisplay">
            <span class="info-label" title="End-to-End Display (Render-ACK Latency)">3. End-to-End Display</span>
            <span id="infoE2EDisplay" class="info-value" style="font-family: 'JetBrains Mono'; color: #f59e0b;">Đang đo qua Browser ACK...</span>
          </div>
          <div style="font-size: 0.68rem; color: var(--text-muted); line-height: 1.35; margin: -0.25rem 0 0.5rem 0; padding: 0 0.2rem;">
            <em>(Đo từ Gateway GET /capture đến Browser ACK sau khi decode &amp; render trong rAF. Không phải thời điểm photon vật lý).</em>
          </div>
          <div class="info-row">
            <span class="info-label">Độ phân giải input</span>
            <span class="info-value">640 × 640</span>
          </div>
          <div class="info-row">
            <span class="info-label">Ngưỡng Conf / Accept</span>
            <span class="info-value">0.20 / 0.75</span>
          </div>
          <div class="info-row">
            <span class="info-label">Tốc độ xử lý</span>
            <span id="infoFps" class="info-value">0.0 FPS</span>
          </div>
          <div style="margin-top: 0.65rem; border-top: 1px solid var(--border); padding-top: 0.75rem;">
            <button id="btnFlash" onclick="toggleFlash()" style="width: 100%; padding: 0.55rem; border-radius: 8px; border: 1px solid var(--border); background: rgba(255, 255, 255, 0.08); color: #fff; font-weight: 600; font-size: 0.82rem; cursor: pointer; transition: background 0.2s;">
              💡 Bật / Tắt Flash ESP32-CAM
            </button>
            <div id="flashMsg" style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.35rem; text-align: center;"></div>
          </div>
        </div>

        <div class="info-card" style="flex: 1;">
          <h3>Chi tiết khuôn mặt hiện tại</h3>
          <div id="detectionList" class="detection-list">
            <div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 1.5rem 0;">
              Chưa phát hiện khuôn mặt
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Performance & Reliability Grid -->
    <div class="perf-grid">
      <!-- Performance Monitoring Card -->
      <div class="perf-card">
        <div class="panel-header">
          <span>⚡ Giám Sát Hiệu Năng Thực Tế (Performance Metrics)</span>
          <span id="perfWindowTag" style="font-size: 0.75rem; color: var(--text-muted); font-family: monospace;">Mẫu: 0 frames</span>
        </div>
        <div class="perf-body">
          <div class="perf-stats">
            <div class="perf-metric">
              <span class="perf-lbl">Inference (RTX 5060)</span>
              <div class="perf-val" id="metricInferAvg" style="color: var(--green);">-- ms</div>
              <span class="perf-sub" id="metricInferP95">P95: -- ms</span>
            </div>
            <div class="perf-metric">
              <span class="perf-lbl">Server Processing</span>
              <div class="perf-val" id="metricProcAvg" style="color: var(--blue);">-- ms</div>
              <span class="perf-sub" id="metricProcP95">P95: -- ms</span>
            </div>
            <div class="perf-metric">
              <span class="perf-lbl">HTTP Capture (Mạng)</span>
              <div class="perf-val" id="metricCapAvg" style="color: var(--amber);">-- ms</div>
              <span class="perf-sub" id="metricCapP95">P95: -- ms</span>
            </div>
            <div class="perf-metric">
              <span class="perf-lbl">Pipeline Loop FPS</span>
              <div class="perf-val" id="metricFpsAvg">-- FPS</div>
              <span class="perf-sub" id="metricLoopP95">Loop P95: -- ms</span>
            </div>
          </div>
          <div class="perf-breakdown">
            <div class="bk-item"><span>JPEG Decode:</span> <strong id="bkDecode">-- ms</strong></div>
            <div class="bk-item"><span>Box Rendering:</span> <strong id="bkRender">-- ms</strong></div>
            <div class="bk-item"><span>JPEG Encode:</span> <strong id="bkEncode">-- ms</strong></div>
            <div class="bk-item"><span>Tổng xử lý Server:</span> <strong id="bkProc">-- ms</strong></div>
            <div class="bk-item"><span>Lỗi / Tổng frames:</span> <strong id="bkFrames">0 / 0</strong></div>
          </div>
        </div>
      </div>

      <!-- Reliability & Recovery Monitoring Card -->
      <div class="perf-card">
        <div class="panel-header">
          <span>🛡️ Độ Tin Cậy & Phục Hồi Kết Nối (Reliability & Recovery)</span>
          <span id="relStatusBadge" class="badge" style="font-size: 0.72rem;">SẴN SÀNG</span>
        </div>
        <div class="perf-body">
          <div class="perf-stats" style="grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));">
            <div class="perf-metric">
              <span class="perf-lbl">Mất Kết Nối / Lỗi</span>
              <div class="perf-val" id="relFailures" style="color: var(--red);">0</div>
              <span class="perf-sub" id="relTimeouts">Timeouts: 0</span>
            </div>
            <div class="perf-metric">
              <span class="perf-lbl">Phục Hồi Thành Công</span>
              <div class="perf-val" id="relRecoveries" style="color: var(--green);">0</div>
              <span class="perf-sub" id="relSuccessRate">Tỷ lệ: 100%</span>
            </div>
            <div class="perf-metric">
              <span class="perf-lbl">Recovery Time (TB)</span>
              <div class="perf-val" id="relAvgRecTime" style="color: var(--blue);">-- s</div>
              <span class="perf-sub" id="relP95RecTime">P95: -- s</span>
            </div>
            <div class="perf-metric">
              <span class="perf-lbl">Trạng Thái Dữ Liệu</span>
              <div class="perf-val" id="relDataFreshness" style="color: var(--green); font-size: 1.05rem;">FRESH</div>
              <span class="perf-sub" id="relDataAge">Age: 0.0s</span>
            </div>
          </div>

          <div style="margin-top: 0.75rem; border-top: 1px solid var(--border); padding-top: 0.75rem;">
            <div style="display: flex; flex-direction: column; gap: 0.6rem;">
              <div>
                <label style="font-size: 0.75rem; color: var(--text-muted); display: block; margin-bottom: 0.25rem;">Kịch bản kiểm thử độ tin cậy (R1–R6):</label>
                <select id="relScenario" style="width: 100%; padding: 0.4rem 0.5rem; background: rgba(255,255,255,0.06); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: 0.78rem; font-family: inherit;">
                  <option value="R1" selected>R1: Mất kết nối thoáng qua (1-2s)</option>
                  <option value="R2">R2: Mất kết nối Wi-Fi tạm thời (10-15s)</option>
                  <option value="R3">R3: Mất nguồn / mất mạng kéo dài (>30s)</option>
                  <option value="R4">R4: HTTP Timeout / mạng trễ cao (>3s)</option>
                  <option value="R5">R5: Tải đồng thời nhiều Web Client (5-10 tabs)</option>
                  <option value="R6">R6: Graceful Shutdown khi có tải (&lt;3s)</option>
                </select>
              </div>
              <div style="display: flex; gap: 0.5rem; align-items: center;">
                <div style="flex: 1;">
                  <label style="font-size: 0.72rem; color: var(--text-muted); display: block; margin-bottom: 0.2rem;">Thời lượng (giây):</label>
                  <input id="relDuration" type="number" min="5" max="180" value="30" style="width: 100%; padding: 0.35rem 0.5rem; background: rgba(255,255,255,0.06); border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-size: 0.8rem; font-family: monospace;">
                </div>
                <div style="flex: 1.2; padding-top: 1.1rem;">
                  <button id="btnReliability" onclick="toggleReliabilityTest()" style="width: 100%; padding: 0.5rem; background: linear-gradient(135deg, #0ea5e9, #6366f1); border: none; border-radius: 6px; color: #fff; font-weight: 700; font-size: 0.78rem; cursor: pointer;">
                    🛡️ Chạy Kiểm Thử R
                  </button>
                </div>
              </div>
              <div id="relProgressBox" style="display: none; background: rgba(0,0,0,0.3); border-radius: 6px; padding: 0.6rem; border: 1px solid var(--border);">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 0.25rem;">
                  <span id="relProgressText">Đang theo dõi...</span>
                  <span id="relCountdown" style="font-family: monospace; font-weight: 700;">30.0s</span>
                </div>
                <div style="width: 100%; height: 5px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
                  <div id="relProgressBar" style="width: 0%; height: 100%; background: #0ea5e9; transition: width 0.3s;"></div>
                </div>
              </div>
              <div id="relResultMsg" style="font-size: 0.72rem; color: var(--text-muted); line-height: 1.35;"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Experimental Benchmark Card -->
      <div class="perf-card">
        <div class="panel-header">
          <span>🧪 Đánh Giá Hiệu Năng Thực Nghiệm (Benchmark S0–S4)</span>
          <span id="bmStatusBadge" class="badge" style="font-size: 0.72rem;">SẴN SÀNG</span>
        </div>
        <div class="perf-body">
          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div>
              <label style="font-size: 0.78rem; color: var(--text-muted); display: block; margin-bottom: 0.3rem;">Kịch bản thử nghiệm:</label>
              <select id="bmScenario" style="width: 100%; padding: 0.45rem 0.6rem; background: rgba(255,255,255,0.06); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: 0.8rem; font-family: inherit;">
                <option value="S0">S0: Không có người (Khung cảnh trống)</option>
                <option value="S1" selected>S1: Một người không đeo khẩu trang</option>
                <option value="S2">S2: Một người đeo khẩu trang đúng cách</option>
                <option value="S3">S3: Nhiều người, trạng thái khẩu trang khác nhau</option>
                <option value="S4">S4: Mặt nhỏ, quay nghiêng hoặc thiếu sáng</option>
              </select>
            </div>
            <div style="display: flex; gap: 0.75rem; align-items: center;">
              <div style="flex: 1;">
                <label style="font-size: 0.78rem; color: var(--text-muted); display: block; margin-bottom: 0.3rem;">Thời lượng (giây):</label>
                <input id="bmDuration" type="number" min="5" max="300" value="60" style="width: 100%; padding: 0.45rem 0.6rem; background: rgba(255,255,255,0.06); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: 0.85rem; font-family: monospace;">
              </div>
              <div style="flex: 1; padding-top: 1.25rem;">
                <button id="btnBenchmark" onclick="toggleBenchmark()" style="width: 100%; padding: 0.55rem; background: linear-gradient(135deg, #2563eb, #10b981); border: none; border-radius: 8px; color: #fff; font-weight: 700; font-size: 0.82rem; cursor: pointer;">
                  ▶ Chạy Benchmark
                </button>
              </div>
            </div>
            <!-- Benchmark Progress -->
            <div id="bmProgressBox" style="display: none; background: rgba(0,0,0,0.3); border-radius: 8px; padding: 0.75rem; border: 1px solid var(--border);">
              <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 0.35rem;">
                <span id="bmProgressText">Đang ghi nhận...</span>
                <span id="bmCountdown" style="font-family: monospace; font-weight: 700;">60.0s</span>
              </div>
              <div style="width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
                <div id="bmProgressBar" style="width: 0%; height: 100%; background: var(--green); transition: width 0.3s;"></div>
              </div>
            </div>
            <div id="bmResultMsg" style="font-size: 0.75rem; color: var(--text-muted); line-height: 1.4;"></div>
          </div>
        </div>
      </div>
    </div>
  </main>

  <footer>
    Multi-Person Face Mask Detection • ESP32-CAM + FastAPI + Ultralytics YOLO11n
  </footer>

  <script>
    async function updateStatus() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();

        // Cập nhật bộ đếm
        document.getElementById('countMask').textContent = data.counts.mask || 0;
        document.getElementById('countNoMask').textContent = data.counts.no_mask || 0;
        document.getElementById('countUnknown').textContent = data.counts.unknown || 0;
        document.getElementById('countTotal').textContent = data.counts.total || 0;

        // Cập nhật FPS
        const fpsStr = (data.fps || 0.0).toFixed(1);
        document.getElementById('fpsOverlay').textContent = fpsStr;
        document.getElementById('infoFps').textContent = fpsStr + ' FPS';

        // Cập nhật IP
        if (data.camera_ip) {
          document.getElementById('cameraIpText').textContent = data.camera_ip;
        }

        // Cập nhật chế độ suy luận và 3 chỉ số Latency của Lab 3
        const devEl = document.getElementById('infoInferDevice');
        const rowInf = document.getElementById('rowCloudInfer');
        const rowRtt = document.getElementById('rowCloudRTT');
        const elInf = document.getElementById('infoCloudInfer');
        const elRtt = document.getElementById('infoCloudRTT');

        if (data.inference_mode === 'cloud') {
          if (devEl) {
            devEl.textContent = 'AWS EC2 CPU (Cloud)';
            devEl.style.color = '#38bdf8';
          }
          if (rowInf) rowInf.style.display = 'flex';
          if (rowRtt) rowRtt.style.display = 'flex';
          if (elInf) elInf.textContent = (data.cloud_inference_ms || 0).toFixed(1) + ' ms';
          if (elRtt) elRtt.textContent = (data.cloud_request_response_ms || 0).toFixed(1) + ' ms';
        } else {
          if (devEl) {
            devEl.textContent = 'NVIDIA RTX 5060 (Local GPU)';
            devEl.style.color = 'var(--green)';
          }
          if (rowInf) rowInf.style.display = 'none';
          if (rowRtt) rowRtt.style.display = 'none';
        }

        // Cập nhật 3. End-to-End Display Latency
        const elE2E = document.getElementById('infoE2EDisplay');
        if (elE2E) {
          if (data.e2e_display_ms && data.e2e_display_ms > 0) {
            const avgStr = data.e2e_display_avg_ms > 0 ? ' (TB: ' + data.e2e_display_avg_ms.toFixed(1) + ' ms)' : '';
            elE2E.textContent = data.e2e_display_ms.toFixed(1) + ' ms' + avgStr;
            elE2E.style.color = '#f59e0b';
          } else {
            elE2E.textContent = 'Đang đo qua Browser ACK...';
            elE2E.style.color = 'var(--text-muted)';
          }
        }

        // Cập nhật cảnh báo dữ liệu cũ (Stale)
        const staleBadge = document.getElementById('staleBadge');
        if (data.is_stale) {
          staleBadge.style.display = 'inline-flex';
          staleBadge.textContent = '⚠️ DỮ LIỆU CŨ (' + (data.data_age_seconds || 0).toFixed(1) + 's)';
        } else {
          staleBadge.style.display = 'none';
        }

        // Cập nhật trạng thái camera
        const badge = document.getElementById('cameraStatusBadge');
        const text = document.getElementById('cameraStatusText');
        badge.className = 'badge';

        if (data.camera_status === 'connected') {
          badge.classList.add('status-connected');
          text.textContent = 'CONNECTED';
        } else if (data.camera_status === 'reconnecting') {
          badge.classList.add('status-reconnecting');
          text.textContent = 'RECONNECTING';
        } else {
          badge.classList.add('status-connecting');
          text.textContent = (data.camera_status || 'UNKNOWN').toUpperCase();
        }

        // Cập nhật danh sách detection
        const detList = document.getElementById('detectionList');
        if (data.detections && data.detections.length > 0) {
          detList.innerHTML = data.detections.map((d, i) => {
            let cls = 'det-unknown';
            let labelVn = 'Chưa rõ';
            if (d.label === 'mask') { cls = 'det-mask'; labelVn = 'Có khẩu trang'; }
            if (d.label === 'no_mask') { cls = 'det-nomask'; labelVn = 'Không khẩu trang'; }

            return `
              <div class="det-item ${cls}">
                <div>
                  <strong>#${i + 1} ${labelVn}</strong>
                  <div style="font-size: 0.72rem; color: var(--text-muted); font-family: monospace;">
                    box: [${d.box.join(', ')}]
                  </div>
                </div>
                <span style="font-family: monospace; font-weight: 700;">${(d.conf * 100).toFixed(1)}%</span>
              </div>
            `;
          }).join('');
        } else {
          detList.innerHTML = `
            <div style="color: var(--text-muted); font-size: 0.8rem; text-align: center; padding: 1.5rem 0;">
              Chưa phát hiện khuôn mặt
            </div>
          `;
        }

      } catch (err) {
        console.warn('Lỗi cập nhật trạng thái:', err);
      }
    }

    // Polling định kỳ mỗi 500ms
    setInterval(updateStatus, 500);
    updateStatus();

    // ─── Luồng Video MJPEG kèm Browser Acknowledgment (End-to-End Display Latency) ───
    async function startMjpegStreamWithAck() {
      const canvas = document.getElementById('videoCanvas');
      const imgFallback = document.getElementById('mjpegStream');
      const ctx = canvas ? canvas.getContext('2d') : null;

      if (!canvas || !ctx || !window.ReadableStream || !window.createImageBitmap) {
        if (canvas) canvas.style.display = 'none';
        if (imgFallback) {
          imgFallback.style.display = 'block';
          imgFallback.src = '/video';
        }
        return;
      }

      function appendBuffer(b1, b2) {
        const merged = new Uint8Array(b1.length + b2.length);
        merged.set(b1, 0);
        merged.set(b2, b1.length);
        return merged;
      }

      function findSubarray(source, sub) {
        for (let i = 0; i <= source.length - sub.length; i++) {
          let found = true;
          for (let j = 0; j < sub.length; j++) {
            if (source[i + j] !== sub[j]) {
              found = false;
              break;
            }
          }
          if (found) return i;
        }
        return -1;
      }

      const boundaryBytes = new Uint8Array([45, 45, 102, 114, 97, 109, 101]); // '--frame'
      const headerDelimBytes = new Uint8Array([13, 10, 13, 10]); // CRLF CRLF
      const textDecoder = new TextDecoder('ascii');
      let streamFailCount = 0;

      function fallbackToNativeImg() {
        console.warn('Chuyển sang chế độ fallback: sử dụng thẻ <img> gốc cho stream MJPEG.');
        if (canvas) canvas.style.display = 'none';
        if (imgFallback) {
          imgFallback.style.display = 'block';
          imgFallback.src = '/video';
        }
      }

      async function connectStream() {
        try {
          const response = await fetch('/video');
          if (!response.ok || !response.body) {
            throw new Error('HTTP ' + response.status);
          }
          streamFailCount = 0;
          const reader = response.body.getReader();
          let buffer = new Uint8Array(0);

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer = appendBuffer(buffer, value);

            while (true) {
              const bIdx = findSubarray(buffer, boundaryBytes);
              if (bIdx === -1) {
                if (buffer.length > boundaryBytes.length) {
                  buffer = buffer.subarray(buffer.length - boundaryBytes.length);
                }
                break;
              }
              const hEnd = findSubarray(buffer.subarray(bIdx), headerDelimBytes);
              if (hEnd === -1) break;

              const headerStart = bIdx;
              const headerEnd = bIdx + hEnd;
              const headerStr = textDecoder.decode(buffer.subarray(headerStart, headerEnd));

              const lenMatch = headerStr.match(/Content-Length:\\s*(\\d+)/i);
              const idMatch = headerStr.match(/X-Frame-ID:\\s*(\\S+)/i);

              if (!lenMatch) {
                buffer = buffer.subarray(headerEnd + 4);
                continue;
              }

              const contentLength = parseInt(lenMatch[1], 10);
              const frameStart = headerEnd + 4;
              const frameEnd = frameStart + contentLength;

              if (buffer.length < frameEnd) {
                break; // Cần thêm dữ liệu cho trọn frame JPEG
              }

              const jpegBytes = buffer.slice(frameStart, frameEnd);
              const frameId = idMatch ? idMatch[1] : '';
              buffer = buffer.subarray(frameEnd);

              if (jpegBytes.length > 0) {
                try {
                  const blob = new Blob([jpegBytes], { type: 'image/jpeg' });
                  const bitmap = await createImageBitmap(blob);
                  requestAnimationFrame(() => {
                    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
                      canvas.width = bitmap.width;
                      canvas.height = bitmap.height;
                    }
                    ctx.drawImage(bitmap, 0, 0);
                    bitmap.close();

                    // Xác nhận frame đã render trên chu kỳ hiển thị của màn hình (Browser ACK)
                    if (frameId) {
                      fetch('/api/frame_ack', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ frame_id: frameId }),
                        keepalive: true,
                      }).catch(() => {});
                    }
                  });
                } catch (decErr) {
                  // Bỏ qua frame giải mã lỗi nếu có
                }
              }
            }
          }
        } catch (err) {
          streamFailCount++;
          console.warn('Kết nối luồng video gián đoạn (' + streamFailCount + '/3):', err);
          if (streamFailCount >= 3) {
            fallbackToNativeImg();
          } else {
            setTimeout(connectStream, 2000);
          }
        }
      }

      connectStream();
    }

    startMjpegStreamWithAck();

    async function toggleFlash() {
      const btn = document.getElementById('btnFlash');
      const msgDiv = document.getElementById('flashMsg');
      btn.disabled = true;
      msgDiv.textContent = 'Đang gửi tín hiệu...';
      try {
        const res = await fetch('/api/flash');
        const data = await res.json();
        msgDiv.textContent = (data.ok ? '✓ ' : '✕ ') + data.message;
        msgDiv.style.color = data.ok ? 'var(--green)' : 'var(--red)';
      } catch (e) {
        msgDiv.textContent = '✕ Lỗi kết nối';
        msgDiv.style.color = 'var(--red)';
      } finally {
        btn.disabled = false;
      }
    }

    // ─── Performance & Reliability Metrics Polling ────────────────────────────
    async function updateMetrics() {
      try {
        const res = await fetch('/api/metrics');
        if (!res.ok) return;
        const data = await res.json();
        const bd = data.latency_breakdown_ms || {};
        const tot = data.latency_totals_ms || {};
        const fps = data.fps || {};
        const rel = data.reliability || {};
        const relM = rel.metrics || {};
        const relT = rel.recovery_time_seconds || {};

        if (bd.yolo_inference) {
          document.getElementById('metricInferAvg').textContent = (bd.yolo_inference.avg || 0) + ' ms';
          document.getElementById('metricInferP95').textContent = 'P95: ' + (bd.yolo_inference.p95 || 0) + ' ms';
        }
        if (tot.server_processing) {
          document.getElementById('metricProcAvg').textContent = (tot.server_processing.avg || 0) + ' ms';
          document.getElementById('metricProcP95').textContent = 'P95: ' + (tot.server_processing.p95 || 0) + ' ms';
          document.getElementById('bkProc').textContent = (tot.server_processing.avg || 0) + ' ms';
        }
        if (bd.http_capture) {
          document.getElementById('metricCapAvg').textContent = (bd.http_capture.avg || 0) + ' ms';
          document.getElementById('metricCapP95').textContent = 'P95: ' + (bd.http_capture.p95 || 0) + ' ms';
        }
        document.getElementById('metricFpsAvg').textContent = (fps.average || 0.0).toFixed(1) + ' FPS';
        if (tot.total_loop) {
          document.getElementById('metricLoopP95').textContent = 'Loop P95: ' + (tot.total_loop.p95 || 0) + ' ms';
        }

        if (bd.jpeg_decode) document.getElementById('bkDecode').textContent = (bd.jpeg_decode.avg || 0) + ' ms';
        if (bd.box_rendering) document.getElementById('bkRender').textContent = (bd.box_rendering.avg || 0) + ' ms';
        if (bd.jpeg_encode) document.getElementById('bkEncode').textContent = (bd.jpeg_encode.avg || 0) + ' ms';

        document.getElementById('bkFrames').textContent = (data.total_failed_frames || 0) + ' / ' + (data.total_processed_frames || 0);
        document.getElementById('perfWindowTag').textContent = 'Mẫu: ' + (data.window_sample_size || 0) + ' frames';

        // Cập nhật các chỉ số Reliability
        document.getElementById('relFailures').textContent = relM.total_connection_failures || 0;
        document.getElementById('relTimeouts').textContent = 'Timeouts: ' + (relM.total_timeouts || 0);
        document.getElementById('relRecoveries').textContent = relM.total_recoveries || 0;
        document.getElementById('relSuccessRate').textContent = 'Tỷ lệ: ' + (relM.recovery_success_rate_pct !== undefined ? relM.recovery_success_rate_pct : 100) + '%';
        document.getElementById('relAvgRecTime').textContent = (relT.avg || 0) + ' s';
        document.getElementById('relP95RecTime').textContent = 'P95: ' + (relT.p95 || 0) + ' s';

        const freshEl = document.getElementById('relDataFreshness');
        if (rel.is_stale) {
          freshEl.textContent = 'STALE';
          freshEl.style.color = 'var(--red)';
        } else {
          freshEl.textContent = 'FRESH';
          freshEl.style.color = 'var(--green)';
        }
        document.getElementById('relDataAge').textContent = 'Age: ' + (rel.data_age_seconds || 0) + 's';

      } catch (e) {
        console.warn('Lỗi lấy metrics:', e);
      }
    }
    setInterval(updateMetrics, 1000);
    updateMetrics();

    // ─── Experimental Benchmark Controller (S0–S4) ────────────────────────────
    let isBenchmarking = false;
    let trackTimer = null;

    async function toggleBenchmark() {
      const btn = document.getElementById('btnBenchmark');
      const scenario = document.getElementById('bmScenario').value;
      const duration = parseFloat(document.getElementById('bmDuration').value) || 60;
      const pBox = document.getElementById('bmProgressBox');
      const resMsg = document.getElementById('bmResultMsg');

      if (!isBenchmarking) {
        btn.disabled = true;
        resMsg.textContent = 'Đang khởi động phiên benchmark...';
        try {
          const res = await fetch(`/api/benchmark/start?scenario=${scenario}&duration=${duration}`, { method: 'POST' });
          const data = await res.json();
          if (data.status === 'ok') {
            isBenchmarking = true;
            btn.textContent = '⏹ Dừng Benchmark';
            btn.style.background = 'var(--red)';
            pBox.style.display = 'block';
            resMsg.textContent = `Đang chạy kịch bản ${scenario} trong ${duration}s...`;
            document.getElementById('bmStatusBadge').textContent = 'ĐANG ĐO (' + scenario + ')';
            document.getElementById('bmStatusBadge').className = 'badge status-reconnecting';
            trackBenchmark();
          } else {
            resMsg.textContent = '✕ Lỗi: ' + data.message;
          }
        } catch (e) {
          resMsg.textContent = '✕ Lỗi gọi API: ' + e;
        } finally {
          btn.disabled = false;
        }
      } else {
        btn.disabled = true;
        try {
          await fetch('/api/benchmark/stop', { method: 'POST' });
          resMsg.textContent = 'Đã gửi lệnh dừng benchmark.';
        } catch (e) {
          resMsg.textContent = 'Lỗi dừng: ' + e;
        } finally {
          btn.disabled = false;
        }
      }
    }

    async function trackBenchmark() {
      if (trackTimer) clearInterval(trackTimer);
      trackTimer = setInterval(async () => {
        try {
          const res = await fetch('/api/benchmark/status');
          if (!res.ok) return;
          const st = await res.json();
          if (!st.is_active) {
            clearInterval(trackTimer);
            trackTimer = null;
            isBenchmarking = false;
            const btn = document.getElementById('btnBenchmark');
            btn.textContent = '▶ Chạy Benchmark';
            btn.style.background = 'linear-gradient(135deg, #2563eb, #10b981)';
            document.getElementById('bmProgressBar').style.width = '100%';
            document.getElementById('bmStatusBadge').textContent = 'HOÀN THÀNH';
            document.getElementById('bmStatusBadge').className = 'badge status-connected';
            document.getElementById('bmResultMsg').innerHTML = `
              ✓ <b>Benchmark hoàn thành!</b> Ghi nhận: <b>${st.recorded_frames} frames</b>.<br>
              File CSV: <code style="color: var(--green);">${st.csv_path || ''}</code><br>
              Báo cáo: <code style="color: var(--blue);">${st.report_path || ''}</code>
            `;
            return;
          }

          document.getElementById('bmCountdown').textContent = (st.remaining_seconds || 0).toFixed(1) + 's';
          document.getElementById('bmProgressBar').style.width = (st.progress_pct || 0) + '%';
          document.getElementById('bmProgressText').textContent = `Đã ghi nhận ${st.recorded_frames} frames (${st.elapsed_seconds}s)`;
        } catch (e) {
          console.warn('Lỗi tracking benchmark:', e);
        }
      }, 800);
    }

    // ─── Reliability Benchmark Controller (R1–R6) ─────────────────────────────
    let isTestingReliability = false;
    let relTrackTimer = null;

    async function toggleReliabilityTest() {
      const btn = document.getElementById('btnReliability');
      const scenario = document.getElementById('relScenario').value;
      const duration = parseFloat(document.getElementById('relDuration').value) || 30;
      const pBox = document.getElementById('relProgressBox');
      const resMsg = document.getElementById('relResultMsg');

      if (!isTestingReliability) {
        btn.disabled = true;
        resMsg.textContent = 'Đang khởi động phiên kiểm thử độ tin cậy...';
        try {
          const res = await fetch(`/api/reliability/start?scenario=${scenario}&duration=${duration}`, { method: 'POST' });
          const data = await res.json();
          if (data.status === 'ok') {
            isTestingReliability = true;
            btn.textContent = '⏹ Dừng Test R';
            btn.style.background = 'var(--red)';
            pBox.style.display = 'block';
            resMsg.textContent = `Đang chạy kịch bản ${scenario} trong ${duration}s...`;
            document.getElementById('relStatusBadge').textContent = 'ĐANG ĐO (' + scenario + ')';
            document.getElementById('relStatusBadge').className = 'badge status-reconnecting';
            trackReliability();
          } else {
            resMsg.textContent = '✕ Lỗi: ' + data.message;
          }
        } catch (e) {
          resMsg.textContent = '✕ Lỗi gọi API: ' + e;
        } finally {
          btn.disabled = false;
        }
      } else {
        btn.disabled = true;
        try {
          await fetch('/api/reliability/stop', { method: 'POST' });
          resMsg.textContent = 'Đã gửi lệnh dừng kiểm thử độ tin cậy.';
        } catch (e) {
          resMsg.textContent = 'Lỗi dừng: ' + e;
        } finally {
          btn.disabled = false;
        }
      }
    }

    async function trackReliability() {
      if (relTrackTimer) clearInterval(relTrackTimer);
      relTrackTimer = setInterval(async () => {
        try {
          const res = await fetch('/api/reliability/status');
          if (!res.ok) return;
          const st = await res.json();
          if (!st.is_active) {
            clearInterval(relTrackTimer);
            relTrackTimer = null;
            isTestingReliability = false;
            const btn = document.getElementById('btnReliability');
            btn.textContent = '🛡️ Chạy Kiểm Thử R';
            btn.style.background = 'linear-gradient(135deg, #0ea5e9, #6366f1)';
            document.getElementById('relProgressBar').style.width = '100%';
            document.getElementById('relStatusBadge').textContent = 'HOÀN THÀNH';
            document.getElementById('relStatusBadge').className = 'badge status-connected';
            document.getElementById('relResultMsg').innerHTML = `
              ✓ <b>Kiểm thử hoàn thành!</b> Ghi nhận: <b>${st.recorded_samples} mẫu</b>.<br>
              File CSV: <code style="color: var(--green);">${st.csv_path || ''}</code><br>
              Báo cáo: <code style="color: var(--blue);">${st.report_path || ''}</code>
            `;
            return;
          }

          document.getElementById('relCountdown').textContent = (st.remaining_seconds || 0).toFixed(1) + 's';
          document.getElementById('relProgressBar').style.width = (st.progress_pct || 0) + '%';
          document.getElementById('relProgressText').textContent = `Đã ghi nhận ${st.recorded_samples} mẫu (${st.elapsed_seconds}s)`;
        } catch (e) {
          console.warn('Lỗi tracking reliability:', e);
        }
      }, 800);
    }

  </script>


</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Giao diện Dashboard chính."""
    current_ip = global_worker.ip if global_worker else DEFAULT_ESP32_IP
    rendered_html = DASHBOARD_HTML.replace(
        '<span id="cameraIpText">192.168.1.153</span>',
        f'<span id="cameraIpText">{current_ip}</span>',
    )
    return HTMLResponse(content=rendered_html, status_code=200)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Trả về 204 No Content cho request favicon của browser."""
    return Response(status_code=204)


# ─── Hàm main CLI ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Khởi chạy Web Dashboard Giám Sát Khẩu Trang ESP32-CAM (FastAPI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ip",
        type=str,
        default=DEFAULT_ESP32_IP,
        help=f"Địa chỉ IP của ESP32-CAM (mặc định: {DEFAULT_ESP32_IP})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_FLOOR,
        help="Ngưỡng tự tin phát hiện tối thiểu (detection confidence floor)",
    )
    parser.add_argument(
        "--accept",
        type=float,
        default=DEFAULT_ACCEPT_THRESH,
        help="Ngưỡng tự tin chấp nhận phân loại mask/no_mask (dưới ngưỡng coi là unknown)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Thiết bị chạy inference: '0' (RTX 5060) hoặc 'cpu'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Đường dẫn file weights YOLO11n đã train",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Địa chỉ host máy chủ Web",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Cổng phục vụ Web Dashboard",
    )
    parser.add_argument(
        "--telegram-token",
        type=str,
        default=None,
        help="Bot Token Telegram (mặc định đọc từ TELEGRAM_BOT_TOKEN trong .env)",
    )
    parser.add_argument(
        "--telegram-chat-id",
        type=str,
        default=None,
        help="Danh sách Chat ID được phép, cách nhau bởi dấu phẩy (mặc định đọc từ .env)",
    )
    parser.add_argument(
        "--stale-threshold",
        type=float,
        default=float(os.getenv("TELEGRAM_STALE_THRESHOLD", "5.0")),
        help="Ngưỡng thời gian tối đa (giây) coi dữ liệu YOLO là thời gian thực",
    )
    parser.add_argument(
        "--disable-telegram",
        action="store_true",
        help="Vô hiệu hóa Telegram Bot kể cả khi có Token",
    )
    parser.add_argument(
        "--replay-dir",
        type=str,
        default=None,
        help="Đường dẫn thư mục ảnh để replay giả lập luồng camera khi offline/benchmark",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        default=(DEFAULT_INFERENCE_MODE == "cloud"),
        help="Kích hoạt chế độ Cloud Inference trên AWS EC2 CPU (mặc định: Local GPU)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_INFERENCE_MODE,
        choices=["local", "cloud"],
        help="Chế độ suy luận: 'local' (RTX 5060 GPU) hoặc 'cloud' (AWS EC2 CPU)",
    )
    parser.add_argument(
        "--cloud-url",
        type=str,
        default=DEFAULT_CLOUD_API_URL,
        help="URL của Cloud Inference API (mặc định: http://127.0.0.1:9000/v1/infer)",
    )
    parser.add_argument(
        "--cloud-timeout",
        type=float,
        default=DEFAULT_CLOUD_TIMEOUT,
        help="Timeout (giây) khi gửi request lên Cloud",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not (0 < args.conf <= args.accept <= 1):
        print(f"[!] Lỗi: Ngưỡng yêu cầu 0 < conf ({args.conf}) <= accept ({args.accept}) <= 1")
        sys.exit(1)

    model_p = Path(args.model)
    if not model_p.exists():
        print(f"[!] Lỗi: Không tìm thấy file model weights tại: {model_p}")
        sys.exit(1)

    active_mode = "cloud" if args.cloud else args.mode.lower()

    global global_worker, global_telegram_service
    global_worker = ESP32PipelineWorker(
        ip=args.ip,
        model_path=model_p,
        conf_thresh=args.conf,
        accept_thresh=args.accept,
        device=args.device,
        replay_dir=Path(args.replay_dir) if args.replay_dir else None,
        inference_mode=active_mode,
        cloud_url=args.cloud_url,
        cloud_timeout=args.cloud_timeout,
    )

    # Cấu hình Telegram Bot Service nếu có Token và không bị disable (không log token ra console)
    env_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    bot_token = (args.telegram_token if args.telegram_token is not None else env_token).strip()

    env_chat_ids = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
    chat_ids_raw = (args.telegram_chat_id if args.telegram_chat_id is not None else env_chat_ids).strip()

    if bot_token and not args.disable_telegram:
        allowed_ids = [cid.strip() for cid in chat_ids_raw.split(",") if cid.strip()]
        global_telegram_service = TelegramBotService(
            token=bot_token,
            allowed_chat_ids=allowed_ids if allowed_ids else None,
            worker=global_worker,
            stale_threshold=args.stale_threshold,
        )
        masked_token = (bot_token[:4] + "..." + bot_token[-4:]) if len(bot_token) > 8 else "***"
        telegram_status_str = f"BẬT (Token: {masked_token} | Chat ID: {allowed_ids or 'Mọi ID (Cảnh báo: Nên giới hạn)'})"

    else:
        global_telegram_service = None
        telegram_status_str = "TẮT (Chưa có TELEGRAM_BOT_TOKEN hoặc dùng --disable-telegram)"

    print("\n" + "=" * 66)
    print("  ESP32-CAM Face Mask Detection Web Dashboard + Telegram Bot")
    print("=" * 66)
    print(f"  ESP32-CAM Source : http://{args.ip}/capture")
    print(f"  ESP32-CAM Flash  : http://{args.ip}/flash")
    print(f"  Model Weights    : {model_p}")
    print(f"  Inference Device : {args.device} (NVIDIA RTX 5060)")
    print(f"  Web Dashboard    : http://{args.host}:{args.port}/")
    print(f"  MJPEG Stream     : http://{args.host}:{args.port}/video")
    print(f"  API Status       : http://{args.host}:{args.port}/api/status")
    print(f"  API Flash        : http://{args.host}:{args.port}/api/flash")
    print(f"  Telegram Bot     : {telegram_status_str}")
    print("=" * 66 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

