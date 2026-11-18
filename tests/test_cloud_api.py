# -*- coding: utf-8 -*-
"""
tests/test_cloud_api.py — Unit & Integration tests cho Cloud Inference API.
=============================================================================
Kiểm thử toàn diện các endpoint /health và /v1/infer bằng TestClient:
  - Kiểm tra trạng thái nạp model và health check
  - Kiểm tra cơ chế xác thực API Key (hợp lệ, sai key, thiếu key)
  - Kiểm tra validation dữ liệu (ảnh rỗng, sai định dạng magic bytes, quá dung lượng)
  - Kiểm tra suy luận trên ảnh thực tế và cấu trúc JSON trả về
  - Kiểm tra tính nhất quán của ngưỡng unknown và phân tách thời gian timing_ms
"""

import io
from pathlib import Path
import unittest

import cv2
from fastapi.testclient import TestClient
import numpy as np

from cloud_api.config import config
from cloud_api.main import app
from cloud_api.model_service import model_service


class TestCloudInferenceAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Đảm bảo model được nạp trên CPU cho môi trường test
        cls.orig_device = config.device
        cls.orig_api_key = config.api_key
        config.device = "cpu"
        model_service.load(model_path=config.model_path, device="cpu")
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        config.device = cls.orig_device
        config.api_key = cls.orig_api_key

    def setUp(self):
        # Reset API key về None mặc định cho mỗi test trừ khi test chỉ định riêng
        config.api_key = None

    def test_01_health_check_healthy(self):
        """GET /health phải trả về HTTP 200, status=healthy và model_loaded=True."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["device"], "cpu")
        self.assertIn("conf_floor", data)
        self.assertIn("accept_thresh", data)
        self.assertGreaterEqual(data["uptime_seconds"], 0.0)

    def test_02_root_endpoint(self):
        """GET / phải trả về thông tin dịch vụ và liên kết tài liệu."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["service"], "YOLO11n Cloud Mask Detection API")
        self.assertIn("docs_url", data)

    def test_03_api_key_authentication(self):
        """Kiểm thử xác thực bảo mật bằng header X-API-Key."""
        config.api_key = "test_super_secret_key_123"

        # Tạo ảnh mẫu nhỏ hợp lệ
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        file_bytes = buf.tobytes()

        # 1. Không truyền header X-API-Key -> 401
        res_missing = self.client.post(
            "/v1/infer",
            files={"file": ("test.jpg", io.BytesIO(file_bytes), "image/jpeg")},
        )
        self.assertEqual(res_missing.status_code, 401)
        self.assertIn("Missing 'X-API-Key'", res_missing.json()["detail"])

        # 2. Sai API Key -> 401
        res_wrong = self.client.post(
            "/v1/infer",
            files={"file": ("test.jpg", io.BytesIO(file_bytes), "image/jpeg")},
            headers={"X-API-Key": "wrong_key_xyz"},
        )
        self.assertEqual(res_wrong.status_code, 401)
        self.assertIn("Invalid API Key", res_wrong.json()["detail"])

        # 3. Đúng API Key -> 200
        res_ok = self.client.post(
            "/v1/infer",
            files={"file": ("test.jpg", io.BytesIO(file_bytes), "image/jpeg")},
            headers={"X-API-Key": "test_super_secret_key_123"},
        )
        self.assertEqual(res_ok.status_code, 200)

    def test_04_invalid_image_payloads(self):
        """Kiểm thử validation khi nhận payload lỗi hoặc sai định dạng."""
        config.api_key = None

        # 1. File rỗng 0 bytes -> 400
        res_empty = self.client.post(
            "/v1/infer",
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
        )
        self.assertEqual(res_empty.status_code, 400)
        self.assertIn("Empty image payload", res_empty.json()["detail"])

        # 2. File văn bản không phải ảnh (sai magic bytes) -> 400
        res_text = self.client.post(
            "/v1/infer",
            files={"file": ("text.txt", io.BytesIO(b"Hello world not an image"), "text/plain")},
        )
        self.assertEqual(res_text.status_code, 400)
        self.assertIn("Invalid image format", res_text.json()["detail"])

        # 3. File vượt quá giới hạn max_content_length -> 413
        orig_max = config.max_content_length
        try:
            config.max_content_length = 500  # Đặt tạm 500 bytes
            dummy_large = b"\xff\xd8\xff" + b"\x00" * 600
            res_large = self.client.post(
                "/v1/infer",
                files={"file": ("large.jpg", io.BytesIO(dummy_large), "image/jpeg")},
            )
            self.assertEqual(res_large.status_code, 413)
            self.assertIn("exceeds limit", res_large.json()["detail"])
        finally:
            config.max_content_length = orig_max

    def test_05_inference_with_sample_image(self):
        """Kiểm thử suy luận với ảnh mẫu thực tế từ dataset ESP32."""
        config.api_key = None

        sample_img_path = (
            Path(__file__).resolve().parents[1]
            / "dataset_raw"
            / "own_photos"
            / "session01_mask"
            / "esp32_20261017_132116_883_1.jpg"
        )
        if sample_img_path.exists():
            with open(sample_img_path, "rb") as f:
                img_bytes = f.read()
        else:
            # Fallback tạo ảnh nhân tạo nếu không có file
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", img)
            img_bytes = buf.tobytes()

        frame_id = "frame_unit_test_999"
        res = self.client.post(
            "/v1/infer",
            files={"file": ("camera.jpg", io.BytesIO(img_bytes), "image/jpeg")},
            data={"frame_id": frame_id},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Kiểm tra metadata và frame_id
        self.assertEqual(data["frame_id"], frame_id)
        self.assertIn("model_version", data)

        # Kiểm tra kích thước ảnh
        self.assertGreater(data["image"]["width"], 0)
        self.assertGreater(data["image"]["height"], 0)

        # Kiểm tra cấu trúc timing_ms
        timing = data["timing_ms"]
        self.assertGreater(timing["cloud_decode_ms"], 0.0)
        self.assertGreater(timing["cloud_inference_ms"], 0.0)
        self.assertGreaterEqual(timing["cloud_postprocess_ms"], 0.0)
        self.assertGreater(timing["cloud_total_ms"], 0.0)

        # Kiểm tra counts
        counts = data["counts"]
        self.assertIn("mask", counts)
        self.assertIn("no_mask", counts)
        self.assertIn("unknown", counts)
        self.assertEqual(counts["total"], counts["mask"] + counts["no_mask"] + counts["unknown"])

        # Kiểm tra các bounding box nếu có
        for det in data["detections"]:
            self.assertIn("class_id", det)
            self.assertIn("label", det)
            self.assertIn("confidence", det)
            self.assertIn("bbox_xyxy", det)
            self.assertEqual(len(det["bbox_xyxy"]), 4)
            # Kiểm tra quy tắc unknown
            if det["confidence"] < config.accept_thresh:
                self.assertEqual(det["label"], config.unknown_label)
                self.assertTrue(det["is_unknown"])


if __name__ == "__main__":
    unittest.main()
