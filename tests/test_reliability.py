# -*- coding: utf-8 -*-
"""
tests/test_reliability.py  —  Unit & Integration Tests cho Reliability & Recovery
==================================================================================
Kiểm thử tự động logic Exponential Backoff, tính Recovery Time, phát hiện Stale Data,
phiên kiểm thử độ tin cậy R1–R6 và cơ chế dừng an toàn (Graceful Shutdown) bằng mock.
"""

from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from src.reliability import (
    RELIABILITY_SCENARIOS,
    ReliabilityBenchmarkSession,
    ReliabilityTracker,
)


class TestReliabilityTracker(unittest.TestCase):
    """Kiểm thử lớp ReliabilityTracker."""

    def setUp(self):
        self.tracker = ReliabilityTracker(
            initial_backoff=0.5,
            max_backoff=4.0,
            backoff_factor=2.0,
            stale_threshold_seconds=2.0,
        )

    def test_initial_state(self):
        """Kiểm tra trạng thái ban đầu."""
        summary = self.tracker.get_summary()
        self.assertEqual(summary["camera_status"], "connecting")
        self.assertTrue(summary["is_stale"])
        self.assertEqual(summary["metrics"]["total_connection_failures"], 0)
        self.assertEqual(summary["metrics"]["total_recoveries"], 0)

    def test_success_recording(self):
        """Ghi nhận thành công bình thường."""
        self.tracker.record_attempt()
        rec_time = self.tracker.record_success()
        self.assertIsNone(rec_time)  # Chưa có lỗi trước đó nên không có recovery

        summary = self.tracker.get_summary()
        self.assertEqual(summary["camera_status"], "connected")
        self.assertFalse(summary["is_stale"])
        self.assertEqual(summary["consecutive_failures"], 0)
        self.assertEqual(summary["current_backoff_seconds"], 0.0)

    def test_exponential_backoff_escalation_and_ceiling(self):
        """Kiểm tra backoff tăng theo cấp số nhân và chạm trần max_backoff."""
        # Lỗi 1: 0.5 * (2.0 ^ 0) = 0.5s
        b1 = self.tracker.record_failure("Lỗi 1")
        self.assertEqual(b1, 0.5)

        # Lỗi 2: 0.5 * (2.0 ^ 1) = 1.0s
        b2 = self.tracker.record_failure("Lỗi 2")
        self.assertEqual(b2, 1.0)

        # Lỗi 3: 0.5 * (2.0 ^ 2) = 2.0s
        b3 = self.tracker.record_failure("Lỗi 3")
        self.assertEqual(b3, 2.0)

        # Lỗi 4: 0.5 * (2.0 ^ 3) = 4.0s (Chạm trần max_backoff = 4.0s)
        b4 = self.tracker.record_failure("Lỗi 4")
        self.assertEqual(b4, 4.0)

        # Lỗi 5: Vẫn giữ ở mức trần 4.0s
        b5 = self.tracker.record_failure("Lỗi 5")
        self.assertEqual(b5, 4.0)

        summary = self.tracker.get_summary()
        self.assertEqual(summary["consecutive_failures"], 5)
        self.assertEqual(summary["camera_status"], "reconnecting")
        self.assertTrue(summary["is_stale"])

    def test_recovery_time_calculation(self):
        """Kiểm tra tính toán Recovery Time khi kết nối lại thành công."""
        # Giả lập đã hoạt động bình thường
        self.tracker.record_success()
        self.tracker.last_successful_frame_time = time.time() - 5.0

        # Mất kết nối
        self.tracker.record_failure("Mất Wi-Fi")
        time.sleep(0.05)

        # Phục hồi thành công
        rec_time = self.tracker.record_success()
        self.assertIsNotNone(rec_time)
        self.assertGreater(rec_time, 0.0)

        summary = self.tracker.get_summary()
        self.assertEqual(summary["camera_status"], "connected")
        self.assertFalse(summary["is_stale"])
        self.assertEqual(summary["metrics"]["total_recoveries"], 1)
        self.assertEqual(summary["metrics"]["recovery_success_rate_pct"], 100.0)
        self.assertGreater(summary["recovery_time_seconds"]["avg"], 0.0)

    def test_staleness_detection(self):
        """Kiểm tra logic phát hiện dữ liệu cũ (Stale Data)."""
        # Chưa từng nhận frame -> stale
        is_stale, age, reason = self.tracker.evaluate_staleness()
        self.assertTrue(is_stale)

        # Nhận frame tươi mới -> fresh
        self.tracker.record_success()
        is_stale, age, reason = self.tracker.evaluate_staleness()
        self.assertFalse(is_stale)

        # Giả lập frame cũ quá 3.0s (vượt ngưỡng 2.0s)
        self.tracker.last_successful_frame_time = time.time() - 3.5
        is_stale, age, reason = self.tracker.evaluate_staleness()
        self.assertTrue(is_stale)
        self.assertIn("vượt ngưỡng", reason)


class TestReliabilityBenchmarkSession(unittest.TestCase):
    """Kiểm thử phiên benchmark độ tin cậy R1–R6 và xuất file."""

    def test_session_lifecycle_and_file_export(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            tracker = ReliabilityTracker(stale_threshold_seconds=2.0)
            session = ReliabilityBenchmarkSession(
                scenario_id="R1",
                duration_seconds=0.2,
                output_dir=out_path,
                tracker=tracker,
            )

            session.start()
            self.assertTrue(session.is_active)
            self.assertFalse(session.completed)

            tracker.record_success()
            session.record_sample("connected", False, 0.1, active_connections=2, event_note="Sample 1")

            time.sleep(0.25)
            # Mẫu này sẽ kích hoạt hoàn tất do vượt duration 0.2s
            finished = session.record_sample("connected", False, 0.1, active_connections=2, event_note="Sample 2")
            self.assertTrue(finished)
            self.assertFalse(session.is_active)
            self.assertTrue(session.completed)

            # Kiểm tra file CSV và Markdown báo cáo
            self.assertIsNotNone(session.csv_path)
            self.assertIsNotNone(session.report_path)
            self.assertTrue(session.csv_path.exists())
            self.assertTrue(session.report_path.exists())

            csv_text = session.csv_path.read_text(encoding="utf-8")
            md_text = session.report_path.read_text(encoding="utf-8")
            self.assertIn("timestamp,elapsed_seconds", csv_text)
            self.assertIn("Báo Cáo Kiểm Thử Độ Tin Cậy", md_text)
            self.assertIn("Kịch Bản R1", md_text)


class TestGracefulShutdownAndFastUnblock(unittest.TestCase):
    """Kiểm tra cơ chế đánh thức nhanh khi shutdown (không bị treo trong backoff sleep)."""

    def test_stop_event_unblocks_immediately(self):
        stop_event = threading.Event()
        start = time.perf_counter()

        def background_setter():
            time.sleep(0.05)
            stop_event.set()

        t = threading.Thread(target=background_setter, daemon=True)
        t.start()

        # Giả lập ngủ 5.0 giây (như backoff sleep) nhưng dùng stop_event.wait
        stopped = stop_event.wait(timeout=5.0)
        duration = time.perf_counter() - start

        self.assertTrue(stopped)
        self.assertLess(duration, 0.5)  # Phải được đánh thức trong < 0.5s thay vì chờ hết 5s


class TestReliabilityAPIIntegration(unittest.TestCase):
    """Kiểm thử tích hợp các endpoint API Reliability và xác minh sửa lỗi quản lý benchmark (Issue #13)."""

    def setUp(self):
        import scripts.esp32_dashboard as dash
        from fastapi.testclient import TestClient
        self.dash = dash
        self.client = TestClient(dash.app)

        # Mock worker
        self.mock_worker = MagicMock()
        self.mock_worker.ip = "192.168.1.153"
        self.mock_worker.camera_status = "connected"
        self.mock_worker.reliability_tracker = ReliabilityTracker()
        self.mock_worker.tracker = MagicMock()
        self.mock_worker.tracker.get_metrics_api_data.return_value = {"status": "ok", "fps": 15.0}

        self.mock_worker.active_benchmark = None
        self.mock_worker.active_reliability_benchmark = None

        self.original_worker = dash.global_worker
        dash.global_worker = self.mock_worker

    def tearDown(self):
        self.dash.global_worker = self.original_worker

    def test_get_reliability_endpoint(self):
        res = self.client.get("/api/reliability")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("camera_status", data)
        self.assertIn("metrics", data)
        self.assertIn("recovery_time_seconds", data)

    def test_benchmark_start_with_json_body_and_scenario_s2(self):
        """Xác minh gửi JSON body {'scenario': 'S2'} được phân tích chuẩn xác (Sửa lỗi Issue #13)."""
        mock_session = MagicMock()
        mock_session.scenario_id = "S2"
        mock_session.get_progress.return_value = {"scenario_id": "S2", "remaining_seconds": 60}
        self.mock_worker.start_benchmark.return_value = mock_session

        res = self.client.post("/api/benchmark/start", json={"scenario": "S2", "duration": 50})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.mock_worker.start_benchmark.assert_called_with("S2", 50.0)

    def test_benchmark_start_conflict_when_already_active(self):
        """Xác minh trả về HTTP 409 khi có phiên benchmark đang chạy dở."""
        self.mock_worker.start_benchmark.side_effect = ValueError(
            "Phiên benchmark 'S1' đang chạy (35s còn lại). Hãy dừng hoặc đợi hoàn tất."
        )

        res = self.client.post("/api/benchmark/start", json={"scenario": "S2"})
        self.assertEqual(res.status_code, 409)
        data = res.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("đang chạy", data["message"])

    def test_status_endpoint_returns_stale_info(self):
        self.mock_worker.get_status.return_value = {
            "camera_status": "reconnecting",
            "is_stale": True,
            "data_age_seconds": 12.5,
            "stale_reason": "Camera mất kết nối 12.5s",
            "fps": 0.0,
            "counts": {"mask": 0, "no_mask": 0, "unknown": 0, "total": 0},
        }

        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["is_stale"])
        self.assertEqual(data["data_age_seconds"], 12.5)


if __name__ == "__main__":
    unittest.main()

