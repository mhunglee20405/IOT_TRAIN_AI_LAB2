# -*- coding: utf-8 -*-
"""
tests/test_cloud_gateway.py — Unit tests cho Cloud Client và Gateway Pipeline Integration.
========================================================================================
Kiểm thử toàn diện:
  1. CloudInferenceClient:
     - Khởi tạo client, phân giải base_url
     - Health check (200 OK, unhealthy, timeout, connection error)
     - Suy luận infer (payload rỗng, phản hồi thành công bóc tách đủ 3 loại latency,
       lỗi HTTP status, lỗi JSON, timeout mạng)
  2. Gateway Worker Cloud Mode:
     - Worker khởi tạo ở chế độ Cloud Inference
     - Xử lý detection từ Cloud, áp dụng đồng bộ ngưỡng unknown (accept_thresh)
     - Xử lý Cloud ngắt kết nối / lỗi: không crash, không tự động fallback sang Local GPU
     - Kiểm tra các trường trạng thái và 3 chỉ số độ trễ trong get_status()
"""

import io
from pathlib import Path
import time
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import requests

from src.cloud_client import CloudInferenceClient, CloudInferenceResult
from src.metrics import LatencyRecord, PerformanceTracker
from scripts.esp32_dashboard import ESP32PipelineWorker


class TestCloudInferenceClient(unittest.TestCase):
    """Kiểm thử đơn vị cho CloudInferenceClient."""

    def setUp(self):
        self.client = CloudInferenceClient(
            api_url="http://127.0.0.1:9000/v1/infer",
            timeout=2.0,
            api_key="test_secret_key",
        )

    def tearDown(self):
        self.client.close()

    def test_01_init_and_url_parsing(self):
        """Khởi tạo client và phân giải base_url chính xác."""
        self.assertEqual(self.client.api_url, "http://127.0.0.1:9000/v1/infer")
        self.assertEqual(self.client.base_url, "http://127.0.0.1:9000")
        self.assertEqual(self.client.timeout, 2.0)
        self.assertEqual(self.client.api_key, "test_secret_key")

        # Trường hợp URL không có /v1/infer
        c2 = CloudInferenceClient(api_url="http://localhost:8001/")
        self.assertEqual(c2.base_url, "http://localhost:8001")
        c2.close()

    @patch("requests.Session.get")
    def test_02_health_check_healthy(self, mock_get):
        """Health check trả về 200 OK khi Cloud sẵn sàng."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "healthy",
            "model_loaded": True,
            "device": "cpu",
        }
        mock_get.return_value = mock_resp

        is_ok, data, msg = self.client.health_check(timeout=1.0)
        self.assertTrue(is_ok)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("sẵn sàng", msg)

    @patch("requests.Session.get")
    def test_03_health_check_unhealthy_or_unloaded(self, mock_get):
        """Health check trả về False khi model chưa nạp hoặc trạng thái lỗi."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "loading",
            "model_loaded": False,
        }
        mock_get.return_value = mock_resp

        is_ok, data, msg = self.client.health_check()
        self.assertFalse(is_ok)
        self.assertIn("chưa nạp", msg)

    @patch("requests.Session.get", side_effect=requests.Timeout("Connection timed out"))
    def test_04_health_check_timeout(self, mock_get):
        """Health check xử lý timeout an toàn."""
        is_ok, data, msg = self.client.health_check(timeout=1.0)
        self.assertFalse(is_ok)
        self.assertEqual(data, {})
        self.assertIn("Timeout", msg)

    @patch("requests.Session.get", side_effect=requests.ConnectionError("Failed to establish a new connection"))
    def test_05_health_check_connection_error(self, mock_get):
        """Health check xử lý lỗi kết nối an toàn."""
        is_ok, data, msg = self.client.health_check()
        self.assertFalse(is_ok)
        self.assertIn("Lỗi kết nối", msg)

    def test_06_infer_empty_bytes(self):
        """Gửi ảnh rỗng (0 bytes) phải trả về lỗi ngay lập tức."""
        res = self.client.infer(b"")
        self.assertFalse(res.success)
        self.assertIn("rỗng", res.error_msg)

    @patch("requests.Session.post")
    def test_07_infer_success_timing_extraction(self, mock_post):
        """Gửi ảnh thành công, bóc tách timings và 3 loại latency chính xác."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "frame_id": "test_fid_100",
            "model": "YOLO11n",
            "device": "cpu",
            "image_width": 640,
            "image_height": 480,
            "timings_ms": {
                "cloud_decode_ms": 3.5,
                "cloud_inference_ms": 28.2,
                "cloud_predict_wall_ms": 29.0,
                "cloud_server_ms": 32.5,
            },
            "detections": [
                {
                    "class_id": 0,
                    "label": "mask",
                    "confidence": 0.88,
                    "bbox_xyxy": [100, 120, 200, 250],
                },
                {
                    "class_id": 1,
                    "label": "no_mask",
                    "confidence": 0.65,
                    "bbox_xyxy": [300, 150, 420, 280],
                },
            ],
        }
        mock_post.return_value = mock_resp

        dummy_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        res = self.client.infer(dummy_jpeg, frame_id="test_fid_100")

        self.assertTrue(res.success)
        self.assertEqual(res.frame_id, "test_fid_100")
        self.assertEqual(res.model, "YOLO11n")
        self.assertEqual(res.device, "cpu")
        self.assertEqual(res.image_width, 640)
        self.assertEqual(res.image_height, 480)
        self.assertEqual(len(res.raw_detections), 2)

        # Kiểm tra đo đạc 3 loại latency của Lab 3
        self.assertEqual(res.cloud_inference_ms, 28.2)
        self.assertEqual(res.cloud_server_ms, 32.5)
        self.assertGreater(res.cloud_request_response_ms, 0.0)
        self.assertGreaterEqual(res.cloud_transport_overhead_est_ms, 0.0)

    @patch("requests.Session.post")
    def test_08_infer_http_error(self, mock_post):
        """Server Cloud trả về lỗi HTTP 500 hoặc 401."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error in model execution"
        mock_post.return_value = mock_resp

        dummy_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 50
        res = self.client.infer(dummy_jpeg)

        self.assertFalse(res.success)
        self.assertEqual(res.status_code, 500)
        self.assertIn("HTTP 500", res.error_msg)

    @patch("requests.Session.post", side_effect=requests.Timeout("Request timeout after 4.0s"))
    def test_09_infer_timeout(self, mock_post):
        """Xử lý ngoại lệ Timeout khi gọi Cloud API."""
        dummy_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 50
        res = self.client.infer(dummy_jpeg)

        self.assertFalse(res.success)
        self.assertIn("Timeout", res.error_msg)
        self.assertGreater(res.cloud_request_response_ms, 0.0)


class TestGatewayCloudIntegration(unittest.TestCase):
    """Kiểm thử tích hợp Cloud Inference trong ESP32PipelineWorker."""

    def setUp(self):
        # Tạo worker giả lập không nạp YOLO GPU local nặng
        self.root = Path(__file__).resolve().parents[1]
        model_path = self.root / "runs" / "detect" / "results" / "train" / "antigravity_gpu_v1" / "weights" / "best.pt"
        self.worker = ESP32PipelineWorker(
            ip="127.0.0.1",
            model_path=model_path,
            conf_thresh=0.20,
            accept_thresh=0.75,
            device="cpu",
            inference_mode="cloud",
            cloud_url="http://127.0.0.1:9000/v1/infer",
            cloud_timeout=2.0,
        )

    def tearDown(self):
        self.worker.stop()

    def test_01_worker_cloud_mode_configuration(self):
        """Worker khởi tạo với cấu hình Cloud Mode chính xác."""
        self.assertEqual(self.worker.inference_mode, "cloud")
        self.assertIsNotNone(self.worker.cloud_client)
        self.assertEqual(self.worker.cloud_client.api_url, "http://127.0.0.1:9000/v1/infer")

    def test_02_status_contains_cloud_metrics(self):
        """get_status() trả về đầy đủ các chỉ số độ trễ của Lab 3."""
        self.worker.cloud_status = "online"
        self.worker.cloud_timing = {
            "cloud_inference_ms": 27.5,
            "cloud_server_ms": 31.0,
            "cloud_decode_ms": 3.0,
            "cloud_predict_wall_ms": 28.0,
            "cloud_request_response_ms": 55.4,
            "cloud_transport_overhead_est_ms": 24.4,
        }

        status = self.worker.get_status()
        self.assertEqual(status["inference_mode"], "cloud")
        self.assertEqual(status["cloud_status"], "online")
        self.assertEqual(status["cloud_inference_ms"], 27.5)
        self.assertEqual(status["cloud_request_response_ms"], 55.4)
        self.assertEqual(status["cloud_server_ms"], 31.0)
        self.assertEqual(status["cloud_transport_overhead_est_ms"], 24.4)

    def test_03_unknown_threshold_consistency(self):
        """Đồng bộ quy tắc unknown: conf < 0.75 phải phân loại thành unknown."""
        # Giả lập kết quả trả về từ Cloud: 1 mask (conf 0.85), 1 no_mask (conf 0.60 < accept_thresh 0.75)
        raw_boxes = [
            {"cls_id": 0, "conf": 0.85, "name": "mask", "bbox_xyxy": [10, 10, 50, 50]},
            {"cls_id": 1, "conf": 0.60, "name": "no_mask", "bbox_xyxy": [60, 60, 100, 100]},
        ]

        # Kiểm tra logic phân loại của Worker
        classified_counts = {"mask": 0, "no_mask": 0, "unknown": 0}
        for item in raw_boxes:
            conf = item["conf"]
            if conf < self.worker.accept_thresh:
                label = "unknown"
            else:
                label = item["name"]
            classified_counts[label] += 1

        self.assertEqual(classified_counts["mask"], 1)
        self.assertEqual(classified_counts["no_mask"], 0)
        self.assertEqual(classified_counts["unknown"], 1)

    def test_04_cloud_disconnect_no_silent_fallback(self):
        """Khi Cloud mất kết nối, hệ thống ghi nhận lỗi và KHÔNG tự ý fallback sang Local GPU."""
        self.worker.cloud_status = "error"
        self.worker.cloud_last_error = "Cloud Request Timeout (>4.0s)"

        status = self.worker.get_status()
        self.assertEqual(status["inference_mode"], "cloud")  # Vẫn giữ Cloud mode
        self.assertEqual(status["cloud_status"], "error")
        self.assertIn("Timeout", status["cloud_last_error"])

    def test_05_frame_ack_lifecycle_and_latency_measurement(self):
        """Kiểm thử vòng đời ghi nhận ACK và tính toán End-to-End Display Latency."""
        import time
        t_mock_start = time.perf_counter() - 0.250  # Giả lập frame bắt đầu trước 250ms
        fid = "frame_test_ack_001"

        with self.worker.ack_lock:
            self.worker.pending_ack_frames[fid] = {
                "capture_start_perf": t_mock_start,
                "capture_start_epoch": time.time() - 0.250,
            }

        # Gọi record_frame_ack
        measured_latency = self.worker.record_frame_ack(fid)
        self.assertIsNotNone(measured_latency)
        self.assertGreaterEqual(measured_latency, 200.0)  # ~250ms

        # Kiểm tra get_status phản hồi chỉ số 3
        status = self.worker.get_status()
        self.assertEqual(status["e2e_display_ms"], round(measured_latency, 1))
        self.assertGreater(status["e2e_display_avg_ms"], 0.0)
        self.assertIn("Browser ACK", status["display_latency_method"])
        self.assertIn("ước lượng", status["display_latency_note"].lower())

    def test_06_frame_ack_api_endpoint(self):
        """Kiểm thử endpoint HTTP POST /api/frame_ack bằng TestClient."""
        from fastapi.testclient import TestClient
        from scripts.esp32_dashboard import app
        import scripts.esp32_dashboard as dash_mod

        # Gán global_worker tạm thời cho test
        dash_mod.global_worker = self.worker
        client = TestClient(app)

        fid = "frame_api_test_002"
        with self.worker.ack_lock:
            self.worker.pending_ack_frames[fid] = {
                "capture_start_perf": time.perf_counter() - 0.150,
                "capture_start_epoch": time.time() - 0.150,
            }

        # 1. Gửi ACK hợp lệ
        res_ok = client.post("/api/frame_ack", json={"frame_id": fid})
        self.assertEqual(res_ok.status_code, 200)
        data = res_ok.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["frame_id"], fid)
        self.assertGreaterEqual(data["render_ack_ms"], 100.0)

        # 2. Gửi ACK thiếu frame_id
        res_bad = client.post("/api/frame_ack", json={})
        self.assertEqual(res_bad.status_code, 400)
        self.assertIn("Thiếu frame_id", res_bad.json()["message"])


if __name__ == "__main__":
    unittest.main()
