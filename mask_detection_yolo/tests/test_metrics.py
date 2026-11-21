# -*- coding: utf-8 -*-
"""
tests/test_metrics.py
=====================
Unit tests cho src/metrics.py (tính toán bách phân vị, summarize, rolling tracker và benchmark session).
Chạy:
    .\\venv\\Scripts\\python.exe -m unittest tests/test_metrics.py
"""

from pathlib import Path
import sys
import tempfile
import time
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import (
    BENCHMARK_SCENARIOS,
    BenchmarkSession,
    LatencyRecord,
    PerformanceTracker,
    calc_percentile,
    summarize_metrics,
)


class TestMetrics(unittest.TestCase):

    def test_calc_percentile_empty(self):
        """Danh sách rỗng phải trả về 0.0."""
        self.assertEqual(calc_percentile([], 50), 0.0)
        self.assertEqual(calc_percentile([], 95), 0.0)

    def test_calc_percentile_single_value(self):
        """Danh sách 1 phần tử phải trả về chính phần tử đó."""
        self.assertEqual(calc_percentile([42.5], 50), 42.5)
        self.assertEqual(calc_percentile([42.5], 95), 42.5)

    def test_calc_percentile_known_distribution(self):
        """Kiểm tra P50 và P95 với chuỗi số từ 1 đến 100."""
        data = list(range(1, 101))
        p50 = calc_percentile(data, 50.0)
        p95 = calc_percentile(data, 95.0)
        self.assertAlmostEqual(p50, 50.5, places=1)
        self.assertAlmostEqual(p95, 95.05, places=1)

    def test_summarize_metrics(self):
        """Kiểm tra tính đúng đắn khi tổng hợp danh sách LatencyRecord."""
        records = [
            LatencyRecord(
                frame_index=1,
                timestamp=time.time(),
                success=True,
                capture_http_ms=30.0,
                decode_ms=5.0,
                inference_ms=15.0,
                render_ms=2.0,
                encode_ms=8.0,
                server_processing_ms=30.0,
                total_loop_ms=60.0,
                fps=16.6,
                mask_count=1,
                total_faces=1,
            ),
            LatencyRecord(
                frame_index=2,
                timestamp=time.time(),
                success=True,
                capture_http_ms=40.0,
                decode_ms=6.0,
                inference_ms=16.0,
                render_ms=3.0,
                encode_ms=9.0,
                server_processing_ms=34.0,
                total_loop_ms=74.0,
                fps=13.5,
                mask_count=2,
                total_faces=2,
            ),
            LatencyRecord(
                frame_index=3,
                timestamp=time.time(),
                success=False,
                error_msg="Connection timeout",
            ),
        ]

        summary = summarize_metrics(records)
        self.assertEqual(summary["total_frames"], 3)
        self.assertEqual(summary["success_frames"], 2)
        self.assertEqual(summary["failed_frames"], 1)
        self.assertAlmostEqual(summary["success_rate_pct"], 66.7, places=1)

        # Kiểm tra inference latency
        infer_stat = summary["latency_ms"]["inference_ms"]
        self.assertEqual(infer_stat["min"], 15.0)
        self.assertEqual(infer_stat["max"], 16.0)
        self.assertEqual(infer_stat["avg"], 15.5)

    def test_performance_tracker_rolling_window(self):
        """Kiểm tra PerformanceTracker duy trì đúng kích thước cửa sổ trượt."""
        tracker = PerformanceTracker(window_size=5)

        for i in range(10):
            rec = LatencyRecord(
                frame_index=i + 1,
                timestamp=time.time(),
                success=(i % 2 == 0),
                total_loop_ms=50.0,
                fps=20.0,
            )
            tracker.record_frame(rec)

        self.assertEqual(tracker.total_processed_frames, 10)
        self.assertEqual(tracker.total_failed_frames, 5)
        # Bộ đệm rolling chỉ giữ 5 bản ghi mới nhất
        self.assertEqual(len(tracker.rolling_records), 5)

        api_data = tracker.get_metrics_api_data("connected")
        self.assertEqual(api_data["camera_status"], "connected")
        self.assertEqual(api_data["total_processed_frames"], 10)
        self.assertEqual(api_data["window_sample_size"], 5)
        self.assertIn("latency_breakdown_ms", api_data)

    def test_benchmark_session_lifecycle_and_export(self):
        """Kiểm tra vòng đời BenchmarkSession và khả năng xuất file CSV/Markdown."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            session = BenchmarkSession(
                scenario_id="S1",
                duration_seconds=0.1,  # Rất ngắn để test kết thúc tự động
                output_dir=tmp_path,
            )
            session.start()
            self.assertTrue(session.is_active)
            self.assertFalse(session.completed)

            # Thêm record
            rec = LatencyRecord(
                frame_index=1,
                timestamp=time.time(),
                success=True,
                capture_http_ms=25.0,
                decode_ms=4.0,
                inference_ms=12.0,
                render_ms=2.0,
                encode_ms=6.0,
                server_processing_ms=24.0,
                total_loop_ms=49.0,
                fps=20.4,
                no_mask_count=1,
                total_faces=1,
            )

            # Chờ hết thời gian
            time.sleep(0.15)
            finished = session.add_record(rec)
            self.assertTrue(finished)
            self.assertTrue(session.completed)
            self.assertFalse(session.is_active)

            # Kiểm tra file đã được tạo
            self.assertIsNotNone(session.csv_path)
            self.assertTrue(session.csv_path.exists())
            self.assertIsNotNone(session.report_path)
            self.assertTrue(session.report_path.exists())

            # Kiểm tra nội dung CSV
            with open(session.csv_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("frame_index", content)
                self.assertIn("inference_ms", content)

            # Kiểm tra nội dung Markdown report
            with open(session.report_path, "r", encoding="utf-8") as f:
                rep_content = f.read()
                self.assertIn("S1", rep_content)
                self.assertIn("YOLO11n Inference", rep_content)


if __name__ == "__main__":
    unittest.main()
