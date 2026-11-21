# -*- coding: utf-8 -*-
"""
cloud_api/s3_logger.py — Ghi lịch sử kết quả nhận diện lên AWS S3 (non-blocking).
====================================================================================
Thiết kế:
  - Hàng đợi bất đồng bộ (queue.Queue) + 1 background daemon thread.
  - Không chặn inference pipeline: enqueue trả về ngay lập tức.
  - Chỉ ghi sự kiện khi có thay đổi trạng thái (state change) hoặc hết cooldown.
  - Sử dụng boto3 với IAM Role của EC2, không hardcode credentials.
  - Retry tối đa 2 lần với exponential backoff khi gặp lỗi S3.
  - Mặc định chỉ lưu metadata JSON, không lưu ảnh khuôn mặt.

Lưu ý quan trọng:
  - cloud_request_response_ms KHÔNG được EC2 tự suy diễn.
    Chỉ ghi vào JSON nếu Gateway gửi kèm giá trị này qua form data.
  - Nhãn unknown được áp dụng đúng quy tắc: confidence < accept_thresh.
"""

import json
import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cloud_api.s3_logger")


class S3ResultLogger:
    """
    Ghi kết quả nhận diện lên S3 dạng JSON, chạy background thread.

    Quy tắc trigger ghi sự kiện:
      1. State change: Tổng hợp nhãn thay đổi so với lần ghi trước.
      2. Cooldown: Nếu không có state change, ghi tối đa 1 record mỗi cooldown_seconds.
      3. Nếu queue đầy (maxsize), bỏ bản ghi mới nhất và log warning.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "results",
        region: str = "ap-southeast-1",
        cooldown_seconds: float = 30.0,
        max_queue_size: int = 100,
        max_retries: int = 2,
        enabled: bool = True,
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region = region
        self.cooldown_seconds = cooldown_seconds
        self.max_retries = max_retries
        self.enabled = enabled

        # State tracking for event triggering
        self._last_state: Optional[Tuple[int, int, int]] = None  # (mask, no_mask, unknown)
        self._last_log_time: float = 0.0
        self._is_first_log: bool = True
        self._lock = threading.Lock()

        # Queue and background thread
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._s3_client = None

        # Statistics
        self._stats = {
            "enqueued": 0,
            "uploaded": 0,
            "dropped": 0,
            "errors": 0,
        }

    def start(self):
        """Khởi tạo boto3 client và bắt đầu background writer thread."""
        if not self.enabled:
            logger.info("[S3Logger] S3 results logging is DISABLED.")
            return

        try:
            import boto3
            from botocore.config import Config as BotoConfig

            boto_config = BotoConfig(
                region_name=self.region,
                retries={"max_attempts": 0, "mode": "standard"},  # Tự quản lý retry
                connect_timeout=5,
                read_timeout=10,
            )
            self._s3_client = boto3.client("s3", config=boto_config)

            # Kiểm tra kết nối cơ bản bằng HeadBucket
            try:
                self._s3_client.head_bucket(Bucket=self.bucket)
                logger.info(
                    f"[S3Logger] Kết nối S3 thành công. Bucket: {self.bucket}, "
                    f"Prefix: {self.prefix}/, Cooldown: {self.cooldown_seconds}s"
                )
            except Exception as e:
                logger.warning(
                    f"[S3Logger] Không thể xác minh bucket '{self.bucket}': {e}. "
                    "Logging vẫn được bật — sẽ retry khi ghi."
                )

        except ImportError:
            logger.error(
                "[S3Logger] Thư viện boto3 chưa được cài đặt. "
                "Chạy: pip install boto3. S3 logging bị vô hiệu hóa."
            )
            self.enabled = False
            return

        # Khởi động background writer thread
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._writer_loop,
            name="s3-result-writer",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("[S3Logger] Background writer thread đã khởi động.")

    def stop(self, timeout: float = 5.0):
        """Dừng background thread và flush queue còn lại."""
        if not self._worker_thread or not self._worker_thread.is_alive():
            return

        logger.info("[S3Logger] Đang dừng background writer thread...")
        self._stop_event.set()
        self._worker_thread.join(timeout=timeout)

        if self._worker_thread.is_alive():
            logger.warning("[S3Logger] Writer thread chưa dừng sau timeout.")

        remaining = self._queue.qsize()
        if remaining > 0:
            logger.warning(f"[S3Logger] Còn {remaining} records trong queue chưa được ghi.")

        logger.info(
            f"[S3Logger] Thống kê: enqueued={self._stats['enqueued']}, "
            f"uploaded={self._stats['uploaded']}, "
            f"dropped={self._stats['dropped']}, "
            f"errors={self._stats['errors']}"
        )

    def should_log(
        self,
        mask_count: int,
        no_mask_count: int,
        unknown_count: int,
    ) -> Tuple[bool, str]:
        """
        Kiểm tra xem có nên ghi sự kiện này không.

        Returns:
            (should_log: bool, trigger_reason: str)
        """
        if not self.enabled:
            return False, ""

        current_state = (mask_count, no_mask_count, unknown_count)
        now = time.monotonic()

        with self._lock:
            # 1. Lần đầu tiên (chưa từng ghi)
            if self._is_first_log:
                self._is_first_log = False
                self._last_state = current_state
                self._last_log_time = now
                return True, "initial"

            # 2. State change detection
            if current_state != self._last_state:
                self._last_state = current_state
                self._last_log_time = now
                return True, "state_change"

            # 3. Cooldown — ghi lại định kỳ dù không đổi trạng thái
            elapsed = now - self._last_log_time
            if elapsed >= self.cooldown_seconds:
                self._last_state = current_state
                self._last_log_time = now
                return True, "cooldown"

        return False, ""

    def enqueue(
        self,
        inference_response_dict: Dict[str, Any],
        trigger_reason: str,
        frame_id: Optional[str] = None,
        camera_source: str = "esp32_cam",
        model_version: str = "",
        device: str = "cpu",
        conf_floor: float = 0.20,
        accept_thresh: float = 0.75,
        gateway_request_response_ms: Optional[float] = None,
    ):
        """
        Đưa bản ghi vào hàng đợi ghi S3 (non-blocking).

        Args:
            inference_response_dict: Kết quả InferenceResponse đã serialize thành dict.
            trigger_reason: Lý do ghi ("state_change", "cooldown", "initial").
            frame_id: ID frame từ client.
            camera_source: Nguồn camera (mặc định "esp32_cam").
            model_version: Phiên bản model.
            device: Thiết bị inference (cpu/cuda).
            conf_floor: Ngưỡng phát hiện tối thiểu.
            accept_thresh: Ngưỡng chấp nhận (dưới mức này gán unknown).
            gateway_request_response_ms: RTT đo bởi Gateway (nếu có). Không tự suy diễn.
        """
        if not self.enabled or self._s3_client is None:
            return

        event_id = str(uuid.uuid4())
        now_utc = datetime.now(timezone.utc)

        # Xây dựng bản ghi JSON
        record = {
            "event_id": event_id,
            "frame_id": frame_id,
            "timestamp_utc": now_utc.isoformat(),
            "camera_source": camera_source,
            "model": {
                "version": model_version,
                "device": device,
                "conf_floor": conf_floor,
                "accept_thresh": accept_thresh,
            },
            "image": inference_response_dict.get("image", {}),
            "detections": inference_response_dict.get("detections", []),
            "counts": inference_response_dict.get("counts", {}),
            "timing_ms": inference_response_dict.get("timing_ms", {}),
            "trigger_reason": trigger_reason,
        }

        # Chỉ ghi gateway_request_response_ms nếu Gateway thực sự cung cấp
        if gateway_request_response_ms is not None:
            record["gateway_request_response_ms"] = round(gateway_request_response_ms, 2)

        # Tính S3 key: results/YYYY/MM/DD/<event_id>.json
        date_prefix = now_utc.strftime("%Y/%m/%d")
        s3_key = f"{self.prefix}/{date_prefix}/{event_id}.json"

        try:
            self._queue.put_nowait((s3_key, record))
            self._stats["enqueued"] += 1
        except queue.Full:
            self._stats["dropped"] += 1
            logger.warning(
                f"[S3Logger] Queue đầy ({self._queue.maxsize}). "
                f"Bỏ qua event {event_id}."
            )

    def _writer_loop(self):
        """Background thread: đọc queue và ghi lên S3 với retry."""
        logger.info("[S3Logger] Writer loop bắt đầu.")

        while not self._stop_event.is_set():
            try:
                s3_key, record = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._upload_with_retry(s3_key, record)
            self._queue.task_done()

        # Flush remaining items khi đang shutdown
        while not self._queue.empty():
            try:
                s3_key, record = self._queue.get_nowait()
                self._upload_with_retry(s3_key, record)
                self._queue.task_done()
            except queue.Empty:
                break

        logger.info("[S3Logger] Writer loop kết thúc.")

    def _upload_with_retry(self, s3_key: str, record: Dict[str, Any]):
        """Upload 1 JSON record lên S3 với retry và exponential backoff."""
        json_body = json.dumps(record, ensure_ascii=False, default=str)
        json_bytes = json_body.encode("utf-8")

        for attempt in range(1, self.max_retries + 2):  # 1 lần chính + max_retries lần retry
            try:
                self._s3_client.put_object(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Body=json_bytes,
                    ContentType="application/json",
                    ServerSideEncryption="AES256",
                )
                self._stats["uploaded"] += 1
                logger.debug(f"[S3Logger] Uploaded: s3://{self.bucket}/{s3_key}")
                return
            except Exception as e:
                if attempt <= self.max_retries:
                    wait = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s, 2s...
                    logger.warning(
                        f"[S3Logger] Upload attempt {attempt} failed for {s3_key}: {e}. "
                        f"Retry in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                else:
                    self._stats["errors"] += 1
                    logger.error(
                        f"[S3Logger] Upload FAILED after {self.max_retries + 1} attempts "
                        f"for {s3_key}: {e}"
                    )

    @property
    def stats(self) -> Dict[str, int]:
        """Thống kê hoạt động logger."""
        return dict(self._stats)
