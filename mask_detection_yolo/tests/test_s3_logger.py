# -*- coding: utf-8 -*-
"""
tests/test_s3_logger.py — Unit tests & Integration test cho S3ResultLogger.
=============================================================================
Kiểm thử:
  1. Unit tests (mock boto3):
     - should_log: state_change, cooldown, initial, skip
     - enqueue: ghi vào queue, queue đầy → drop
     - _upload_with_retry: thành công, retry, thất bại hết
     - JSON record format: đủ fields, không có gateway_rtt khi None
     - start/stop lifecycle
  2. Integration test (ghi thật lên S3):
     - Cần IAM Role hoặc credentials cấu hình sẵn
     - Ghi 1 JSON record lên bucket thật, đọc lại xác minh
     - Chạy: python -m pytest tests/test_s3_logger.py::TestS3LoggerIntegration -v
"""

import io
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Đảm bảo import từ project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloud_api.s3_logger import S3ResultLogger


def _make_response_dict(mask=1, no_mask=0, unknown=0, confidence=0.92):
    """Tạo dict mô phỏng InferenceResponse.model_dump()."""
    detections = []
    for i in range(mask):
        detections.append({
            "class_id": 0,
            "raw_class_name": "mask",
            "label": "mask",
            "confidence": confidence,
            "bbox_xyxy": [100 + i * 50, 50, 300 + i * 50, 400],
            "is_unknown": False,
        })
    for i in range(no_mask):
        detections.append({
            "class_id": 1,
            "raw_class_name": "no_mask",
            "label": "no_mask",
            "confidence": confidence,
            "bbox_xyxy": [100 + i * 50, 50, 300 + i * 50, 400],
            "is_unknown": False,
        })
    for i in range(unknown):
        detections.append({
            "class_id": 0,
            "raw_class_name": "mask",
            "label": "unknown",
            "confidence": 0.3,
            "bbox_xyxy": [100 + i * 50, 50, 300 + i * 50, 400],
            "is_unknown": True,
        })

    return {
        "frame_id": "frame_test_001",
        "model_version": "yolo11n_antigravity_v1",
        "image": {"width": 640, "height": 480},
        "detections": detections,
        "counts": {
            "mask": mask,
            "no_mask": no_mask,
            "unknown": unknown,
            "total": mask + no_mask + unknown,
        },
        "timing_ms": {
            "cloud_decode_ms": 1.2,
            "cloud_inference_ms": 26.5,
            "cloud_postprocess_ms": 0.5,
            "cloud_total_ms": 28.2,
        },
    }


class TestShouldLog(unittest.TestCase):
    """Kiểm thử logic trigger ghi sự kiện."""

    def setUp(self):
        self.logger = S3ResultLogger(
            bucket="test-bucket",
            cooldown_seconds=5.0,
            enabled=True,
        )

    def test_initial_always_logs(self):
        """Lần đầu tiên luôn được ghi."""
        should, reason = self.logger.should_log(1, 0, 0)
        self.assertTrue(should)
        self.assertEqual(reason, "initial")

    def test_state_change_triggers_log(self):
        """Thay đổi trạng thái → ghi sự kiện."""
        self.logger.should_log(1, 0, 0)  # initial
        should, reason = self.logger.should_log(0, 1, 0)  # changed
        self.assertTrue(should)
        self.assertEqual(reason, "state_change")

    def test_no_change_within_cooldown_skips(self):
        """Không đổi trạng thái + chưa hết cooldown → bỏ qua."""
        self.logger.should_log(1, 0, 0)  # initial
        should, reason = self.logger.should_log(1, 0, 0)  # same, within cooldown
        self.assertFalse(should)
        self.assertEqual(reason, "")

    def test_cooldown_triggers_log(self):
        """Hết cooldown dù không đổi trạng thái → ghi lại."""
        self.logger.should_log(1, 0, 0)  # initial
        # Giả lập hết cooldown
        self.logger._last_log_time = time.monotonic() - 10.0
        should, reason = self.logger.should_log(1, 0, 0)
        self.assertTrue(should)
        self.assertEqual(reason, "cooldown")

    def test_disabled_always_skips(self):
        """Logger bị tắt → luôn bỏ qua."""
        self.logger.enabled = False
        should, reason = self.logger.should_log(1, 0, 0)
        self.assertFalse(should)


class TestEnqueue(unittest.TestCase):
    """Kiểm thử hàng đợi ghi sự kiện."""

    def test_enqueue_adds_to_queue(self):
        """Enqueue thành công đưa item vào queue."""
        logger_obj = S3ResultLogger(
            bucket="test-bucket",
            enabled=True,
        )
        # Mock s3_client để enqueue không bị bỏ qua
        logger_obj._s3_client = MagicMock()

        response_dict = _make_response_dict(mask=1)
        logger_obj.enqueue(
            inference_response_dict=response_dict,
            trigger_reason="state_change",
            frame_id="frame_123",
            model_version="test_v1",
        )
        self.assertEqual(logger_obj._queue.qsize(), 1)
        self.assertEqual(logger_obj._stats["enqueued"], 1)

    def test_enqueue_full_queue_drops(self):
        """Queue đầy → bỏ bản ghi mới và tăng dropped counter."""
        logger_obj = S3ResultLogger(
            bucket="test-bucket",
            max_queue_size=2,
            enabled=True,
        )
        logger_obj._s3_client = MagicMock()

        for i in range(3):
            logger_obj.enqueue(
                inference_response_dict=_make_response_dict(),
                trigger_reason="test",
                frame_id=f"frame_{i}",
            )

        self.assertEqual(logger_obj._queue.qsize(), 2)
        self.assertEqual(logger_obj._stats["enqueued"], 2)
        self.assertEqual(logger_obj._stats["dropped"], 1)

    def test_enqueue_disabled_does_nothing(self):
        """Logger tắt → enqueue không làm gì."""
        logger_obj = S3ResultLogger(bucket="test", enabled=False)
        logger_obj.enqueue(
            inference_response_dict=_make_response_dict(),
            trigger_reason="test",
        )
        self.assertEqual(logger_obj._queue.qsize(), 0)

    def test_enqueue_without_gateway_rtt(self):
        """Không có gateway_rtt_ms → JSON record không chứa field này."""
        logger_obj = S3ResultLogger(bucket="test", enabled=True)
        logger_obj._s3_client = MagicMock()

        logger_obj.enqueue(
            inference_response_dict=_make_response_dict(),
            trigger_reason="test",
            gateway_request_response_ms=None,
        )

        s3_key, record = logger_obj._queue.get_nowait()
        self.assertNotIn("gateway_request_response_ms", record)

    def test_enqueue_with_gateway_rtt(self):
        """Có gateway_rtt_ms → JSON record chứa field này."""
        logger_obj = S3ResultLogger(bucket="test", enabled=True)
        logger_obj._s3_client = MagicMock()

        logger_obj.enqueue(
            inference_response_dict=_make_response_dict(),
            trigger_reason="test",
            gateway_request_response_ms=45.67,
        )

        s3_key, record = logger_obj._queue.get_nowait()
        self.assertIn("gateway_request_response_ms", record)
        self.assertEqual(record["gateway_request_response_ms"], 45.67)


class TestJsonRecordFormat(unittest.TestCase):
    """Kiểm tra định dạng JSON record đầy đủ."""

    def test_record_has_all_required_fields(self):
        """Bản ghi JSON phải có đủ tất cả fields theo yêu cầu."""
        logger_obj = S3ResultLogger(bucket="test", enabled=True)
        logger_obj._s3_client = MagicMock()

        logger_obj.enqueue(
            inference_response_dict=_make_response_dict(mask=1, no_mask=1),
            trigger_reason="state_change",
            frame_id="frame_check_format",
            camera_source="esp32_cam_01",
            model_version="yolo11n_v1",
            device="cpu",
            conf_floor=0.20,
            accept_thresh=0.75,
        )

        s3_key, record = logger_obj._queue.get_nowait()

        # Kiểm tra top-level fields
        self.assertIn("event_id", record)
        self.assertIn("frame_id", record)
        self.assertIn("timestamp_utc", record)
        self.assertIn("camera_source", record)
        self.assertIn("model", record)
        self.assertIn("image", record)
        self.assertIn("detections", record)
        self.assertIn("counts", record)
        self.assertIn("timing_ms", record)
        self.assertIn("trigger_reason", record)

        # Kiểm tra model sub-fields
        model = record["model"]
        self.assertEqual(model["version"], "yolo11n_v1")
        self.assertEqual(model["device"], "cpu")
        self.assertEqual(model["conf_floor"], 0.20)
        self.assertEqual(model["accept_thresh"], 0.75)

        # Kiểm tra S3 key format: results/YYYY/MM/DD/<uuid>.json
        self.assertTrue(s3_key.startswith("results/"))
        self.assertTrue(s3_key.endswith(".json"))
        parts = s3_key.split("/")
        self.assertEqual(len(parts), 5)  # results/YYYY/MM/DD/event_id.json

        # Kiểm tra timestamp ISO 8601
        self.assertIn("T", record["timestamp_utc"])
        self.assertTrue(record["timestamp_utc"].endswith("+00:00"))

        # Kiểm tra detections
        self.assertEqual(len(record["detections"]), 2)
        self.assertEqual(record["counts"]["total"], 2)


class TestUploadWithRetry(unittest.TestCase):
    """Kiểm thử upload S3 với retry logic."""

    def test_upload_success_first_attempt(self):
        """Upload thành công ngay lần đầu."""
        logger_obj = S3ResultLogger(bucket="test", max_retries=2, enabled=True)
        logger_obj._s3_client = MagicMock()

        record = {"event_id": "test", "data": "sample"}
        logger_obj._upload_with_retry("test/key.json", record)

        logger_obj._s3_client.put_object.assert_called_once()
        self.assertEqual(logger_obj._stats["uploaded"], 1)
        self.assertEqual(logger_obj._stats["errors"], 0)

    def test_upload_retry_then_success(self):
        """Thất bại lần 1, thành công lần 2."""
        logger_obj = S3ResultLogger(bucket="test", max_retries=2, enabled=True)
        logger_obj._s3_client = MagicMock()
        logger_obj._s3_client.put_object.side_effect = [
            Exception("Temporary network error"),
            None,  # Success on retry
        ]

        record = {"event_id": "test"}
        logger_obj._upload_with_retry("test/key.json", record)

        self.assertEqual(logger_obj._s3_client.put_object.call_count, 2)
        self.assertEqual(logger_obj._stats["uploaded"], 1)

    def test_upload_all_retries_fail(self):
        """Tất cả retry đều thất bại → ghi lỗi."""
        logger_obj = S3ResultLogger(bucket="test", max_retries=1, enabled=True)
        logger_obj._s3_client = MagicMock()
        logger_obj._s3_client.put_object.side_effect = Exception("Persistent error")

        record = {"event_id": "test"}
        logger_obj._upload_with_retry("test/key.json", record)

        self.assertEqual(logger_obj._s3_client.put_object.call_count, 2)  # 1 chính + 1 retry
        self.assertEqual(logger_obj._stats["errors"], 1)
        self.assertEqual(logger_obj._stats["uploaded"], 0)

    def test_upload_puts_correct_content_type(self):
        """Upload sử dụng ContentType application/json và SSE AES256."""
        logger_obj = S3ResultLogger(bucket="my-bucket", enabled=True)
        logger_obj._s3_client = MagicMock()

        record = {"event_id": "abc", "counts": {"mask": 1}}
        logger_obj._upload_with_retry("results/2026/11/18/abc.json", record)

        call_kwargs = logger_obj._s3_client.put_object.call_args
        self.assertEqual(call_kwargs.kwargs["Bucket"], "my-bucket")
        self.assertEqual(call_kwargs.kwargs["Key"], "results/2026/11/18/abc.json")
        self.assertEqual(call_kwargs.kwargs["ContentType"], "application/json")
        self.assertEqual(call_kwargs.kwargs["ServerSideEncryption"], "AES256")


class TestStartStop(unittest.TestCase):
    """Kiểm thử lifecycle start/stop."""

    @patch("cloud_api.s3_logger.S3ResultLogger._writer_loop")
    def test_start_creates_client_and_thread(self, mock_writer):
        """start() khởi tạo boto3 client và bắt đầu writer thread."""
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_botocore = MagicMock()

        with patch.dict(sys.modules, {"boto3": mock_boto3, "botocore": mock_botocore, "botocore.config": mock_botocore}):
            logger_obj = S3ResultLogger(bucket="test", enabled=True)
            logger_obj.start()

            mock_boto3.client.assert_called_once()
            self.assertIsNotNone(logger_obj._s3_client)
            self.assertIsNotNone(logger_obj._worker_thread)

            logger_obj.stop(timeout=2.0)

    def test_start_disabled_does_nothing(self):
        """start() khi disabled không tạo thread."""
        logger_obj = S3ResultLogger(bucket="test", enabled=False)
        logger_obj.start()
        self.assertIsNone(logger_obj._worker_thread)

    @patch.dict(os.environ, {}, clear=False)
    def test_start_without_boto3_disables(self):
        """start() khi boto3 chưa cài → tự disable."""
        logger_obj = S3ResultLogger(bucket="test", enabled=True)

        with patch.dict("sys.modules", {"boto3": None}):
            with patch("builtins.__import__", side_effect=ImportError("No boto3")):
                logger_obj.start()

        self.assertFalse(logger_obj.enabled)


class TestEndToEndMocked(unittest.TestCase):
    """Kiểm thử end-to-end với mock boto3 client."""

    def test_full_pipeline_enqueue_and_upload(self):
        """Enqueue → writer thread → upload thành công."""
        logger_obj = S3ResultLogger(
            bucket="test-bucket",
            cooldown_seconds=0.1,
            enabled=True,
        )
        logger_obj._s3_client = MagicMock()

        # Khởi động writer thread
        import threading
        logger_obj._stop_event.clear()
        logger_obj._worker_thread = threading.Thread(
            target=logger_obj._writer_loop, daemon=True
        )
        logger_obj._worker_thread.start()

        try:
            # Enqueue 1 bản ghi
            logger_obj.enqueue(
                inference_response_dict=_make_response_dict(mask=2, no_mask=1),
                trigger_reason="state_change",
                frame_id="e2e_test",
                model_version="v1",
            )

            # Đợi writer xử lý
            time.sleep(1.0)

            self.assertEqual(logger_obj._stats["enqueued"], 1)
            self.assertEqual(logger_obj._stats["uploaded"], 1)
            logger_obj._s3_client.put_object.assert_called_once()

            # Kiểm tra nội dung ghi
            call_kwargs = logger_obj._s3_client.put_object.call_args.kwargs
            body = json.loads(call_kwargs["Body"].decode("utf-8"))
            self.assertEqual(body["frame_id"], "e2e_test")
            self.assertEqual(body["counts"]["mask"], 2)
            self.assertEqual(body["counts"]["no_mask"], 1)
            self.assertEqual(body["trigger_reason"], "state_change")

        finally:
            logger_obj.stop(timeout=3.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Test — Ghi thật lên S3
# Chạy: python -m pytest tests/test_s3_logger.py::TestS3LoggerIntegration -v
# Yêu cầu: IAM Role hoặc AWS credentials đã cấu hình, bucket tồn tại.
# ═══════════════════════════════════════════════════════════════════════════════


@unittest.skipUnless(
    os.getenv("S3_INTEGRATION_TEST", "").lower() in ("true", "1", "yes"),
    "Bỏ qua integration test S3. Đặt S3_INTEGRATION_TEST=true để chạy.",
)
class TestS3LoggerIntegration(unittest.TestCase):
    """Integration test: ghi JSON thật lên S3 và đọc lại."""

    BUCKET = os.getenv(
        "S3_RESULTS_BUCKET",
        "esp32-yolo-lab3-511974511954-ap-southeast-1-an",
    )
    PREFIX = "results/_integration_test"

    def test_write_and_read_real_s3(self):
        """Ghi 1 JSON record lên S3 thật, đọc lại và xác minh."""
        import boto3
        from datetime import datetime, timezone

        s3 = boto3.client("s3", region_name="ap-southeast-1")

        # Tạo logger
        logger_obj = S3ResultLogger(
            bucket=self.BUCKET,
            prefix=self.PREFIX,
            region="ap-southeast-1",
            cooldown_seconds=0,
            enabled=True,
        )
        logger_obj._s3_client = s3

        # Tạo bản ghi
        response_dict = _make_response_dict(mask=1, no_mask=1, unknown=0)

        logger_obj.enqueue(
            inference_response_dict=response_dict,
            trigger_reason="integration_test",
            frame_id="frame_integration_001",
            camera_source="esp32_cam_test",
            model_version="yolo11n_integration_test",
            device="cpu",
        )

        # Lấy item từ queue và upload trực tiếp
        s3_key, record = logger_obj._queue.get_nowait()
        logger_obj._upload_with_retry(s3_key, record)

        self.assertEqual(logger_obj._stats["uploaded"], 1)
        print(f"\n[Integration] Uploaded to: s3://{self.BUCKET}/{s3_key}")

        # Đọc lại từ S3 và xác minh
        response = s3.get_object(Bucket=self.BUCKET, Key=s3_key)
        body = json.loads(response["Body"].read().decode("utf-8"))

        self.assertEqual(body["frame_id"], "frame_integration_001")
        self.assertEqual(body["camera_source"], "esp32_cam_test")
        self.assertEqual(body["model"]["version"], "yolo11n_integration_test")
        self.assertEqual(body["counts"]["mask"], 1)
        self.assertEqual(body["counts"]["no_mask"], 1)
        self.assertEqual(body["trigger_reason"], "integration_test")
        self.assertIn("event_id", body)
        self.assertIn("timestamp_utc", body)
        self.assertIn("timing_ms", body)

        print(f"[Integration] Record verified successfully!")
        print(f"[Integration] Event ID: {body['event_id']}")
        print(f"[Integration] Timestamp: {body['timestamp_utc']}")

        # Cleanup: xóa test object
        try:
            s3.delete_object(Bucket=self.BUCKET, Key=s3_key)
            print(f"[Integration] Cleaned up test object: {s3_key}")
        except Exception as e:
            print(f"[Integration] Warning: could not cleanup {s3_key}: {e}")


if __name__ == "__main__":
    unittest.main()
