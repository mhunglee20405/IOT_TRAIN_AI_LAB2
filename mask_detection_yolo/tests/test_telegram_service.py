# -*- coding: utf-8 -*-
"""
tests/test_telegram_service.py
==============================
Unit tests cho TelegramBotService bằng unittest.mock (hoàn toàn không cần camera thật hay Telegram API thật).

Chạy test từ thư mục gốc mask_detection_yolo:
    .\\venv\\Scripts\\python.exe -m unittest tests/test_telegram_service.py
"""

import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# Thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telegram_service import TelegramBotService


class DummyWorker:
    """Mock worker mô phỏng ESP32PipelineWorker."""

    def __init__(self):
        self.ip = "192.168.1.153"
        self.camera_status = "connected"
        self.counts = {"mask": 2, "no_mask": 1, "unknown": 0, "total": 3}
        self.fps = 18.5
        self.last_capture_time = time.time()
        self.latest_jpeg = b"\xff\xd8\xff\xe0mock_jpeg_bytes\xff\xd9"
        self.flash_result = (True, "Đèn Flash bật thành công")

    def get_yolo_snapshot(self, stale_threshold: float = 5.0):
        age = time.time() - self.last_capture_time
        status = {
            "camera_status": self.camera_status,
            "camera_ip": self.ip,
            "counts": self.counts,
            "fps": self.fps,
            "last_capture_time": self.last_capture_time,
        }
        if self.camera_status != "connected":
            return False, status, None, f"Camera đang ở trạng thái {self.camera_status}"
        if age > stale_threshold:
            return False, status, None, f"Dữ liệu cũ ({age:.1f}s > {stale_threshold}s)"
        return True, status, self.latest_jpeg, "Dữ liệu hợp lệ"

    def toggle_flash(self, timeout: float = 4.0):
        return self.flash_result


class TestTelegramBotService(unittest.TestCase):

    def setUp(self):
        self.mock_worker = DummyWorker()
        self.service = TelegramBotService(
            token="123456:TEST_TOKEN_FOR_MOCK",
            allowed_chat_ids=["111222", "333444"],
            worker=self.mock_worker,
            stale_threshold=5.0,
        )
        self.service.session = MagicMock()

    def test_authorization_allowed(self):
        """Chat ID nằm trong danh sách được phép thì phải trả về True."""
        self.assertTrue(self.service.is_chat_allowed("111222"))
        self.assertTrue(self.service.is_chat_allowed(333444))

    def test_authorization_denied(self):
        """Chat ID lạ không nằm trong danh sách phải bị từ chối và nhận thông báo lỗi."""
        self.assertFalse(self.service.is_chat_allowed("999999"))

        with patch.object(self.service, "send_message") as mock_send:
            self.service.handle_command("999999", "/status")
            mock_send.assert_called_once()
            args, _ = mock_send.call_args
            self.assertEqual(args[0], "999999")
            self.assertIn("Bạn không có quyền", args[1])

    def test_handle_start(self):
        """Lệnh /start phải gửi tin nhắn hướng dẫn các lệnh khả dụng."""
        with patch.object(self.service, "send_message") as mock_send:
            self.service.handle_command("111222", "/start")
            mock_send.assert_called_once()
            _, text = mock_send.call_args[0]
            self.assertIn("/status", text)
            self.assertIn("/photo", text)
            self.assertIn("/flash", text)

    def test_handle_status_fresh_data(self):
        """Lệnh /status khi dữ liệu mới phải gửi đúng số lượng mask/no_mask và FPS."""
        self.mock_worker.counts = {"mask": 5, "no_mask": 2, "unknown": 1, "total": 8}
        self.mock_worker.fps = 20.2
        self.mock_worker.last_capture_time = time.time()

        with patch.object(self.service, "send_message") as mock_send:
            self.service.handle_command("111222", "/status")
            mock_send.assert_called_once()
            _, text = mock_send.call_args[0]
            self.assertIn("Đeo khẩu trang (mask):</b> 5", text)
            self.assertIn("Không khẩu trang (no_mask):</b> 2", text)
            self.assertIn("Chưa rõ (unknown):</b> 1", text)
            self.assertIn("Tổng số khuôn mặt:</b> 8", text)
            self.assertIn("20.2 FPS", text)

    def test_handle_status_stale_data(self):
        """Lệnh /status khi dữ liệu cũ (> stale_threshold) phải từ chối và cảnh báo người dùng."""
        # Giả lập dữ liệu nhận từ 10 giây trước (vượt ngưỡng 5s)
        self.mock_worker.last_capture_time = time.time() - 10.0

        with patch.object(self.service, "send_message") as mock_send:
            self.service.handle_command("111222", "/status")
            mock_send.assert_called_once()
            _, text = mock_send.call_args[0]
            self.assertIn("Không thể cung cấp dữ liệu thời gian thực", text)
            self.assertIn("Dữ liệu cũ", text)

    def test_handle_status_camera_disconnected(self):
        """Lệnh /status khi camera mất kết nối phải thông báo lỗi rõ ràng."""
        self.mock_worker.camera_status = "reconnecting"

        with patch.object(self.service, "send_message") as mock_send:
            self.service.handle_command("111222", "/status")
            mock_send.assert_called_once()
            _, text = mock_send.call_args[0]
            self.assertIn("Không thể cung cấp dữ liệu thời gian thực", text)
            self.assertIn("RECONNECTING", text)

    def test_handle_photo_fresh(self):
        """Lệnh /photo khi dữ liệu mới phải gọi send_photo với bytes JPEG và caption."""
        self.mock_worker.last_capture_time = time.time()

        with patch.object(self.service, "send_photo") as mock_send_photo:
            self.service.handle_command("111222", "/photo")
            mock_send_photo.assert_called_once()
            chat_id, photo_bytes, caption = mock_send_photo.call_args[0]
            self.assertEqual(chat_id, "111222")
            self.assertEqual(photo_bytes, self.mock_worker.latest_jpeg)
            self.assertIn("Mask:", caption)

    def test_handle_photo_stale_or_disconnected(self):
        """Lệnh /photo khi camera mất sóng phải từ chối gửi ảnh cũ và gửi tin nhắn cảnh báo."""
        self.mock_worker.camera_status = "disconnected"

        with patch.object(self.service, "send_photo") as mock_send_photo, \
             patch.object(self.service, "send_message") as mock_send_msg:
            self.service.handle_command("111222", "/photo")
            mock_send_photo.assert_not_called()
            mock_send_msg.assert_called_once()
            _, text = mock_send_msg.call_args[0]
            self.assertIn("Không thể chụp ảnh thời gian thực", text)

    def test_handle_flash_success(self):
        """Lệnh /flash gọi worker.toggle_flash và gửi kết quả về Telegram."""
        self.mock_worker.flash_result = (True, "Flash toggled on")

        with patch.object(self.service, "send_message") as mock_send:
            self.service.handle_command("111222", "/flash")
            mock_send.assert_called_once()
            _, text = mock_send.call_args[0]
            self.assertIn("Flash toggled on", text)

    def test_handle_flash_failure(self):
        """Lệnh /flash xử lý lỗi khi ESP32-CAM không phản hồi."""
        self.mock_worker.flash_result = (False, "Timeout kết nối tới ESP32-CAM")

        with patch.object(self.service, "send_message") as mock_send:
            self.service.handle_command("111222", "/flash")
            mock_send.assert_called_once()
            _, text = mock_send.call_args[0]
            self.assertIn("Lỗi thao tác đèn Flash", text)
            self.assertIn("Timeout", text)


if __name__ == "__main__":
    unittest.main()
