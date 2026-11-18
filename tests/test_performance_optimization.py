# -*- coding: utf-8 -*-
"""
tests/test_performance_optimization.py  —  Unit Test cho Tối Ưu Hiệu Năng & Before/After Benchmark
"""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import LatencyRecord, calc_percentile, summarize_metrics
from scripts.benchmark_performance import parse_benchmark_csv


class TestPerformanceOptimization(unittest.TestCase):
    def test_calc_percentile(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        p50 = calc_percentile(vals, 50.0)
        self.assertAlmostEqual(p50, 30.0, places=2)
        p95 = calc_percentile(vals, 95.0)
        self.assertGreater(p95, 45.0)

    def test_summarize_metrics_throughput_fps(self):
        records = [
            LatencyRecord(
                frame_index=1,
                timestamp=100.0,
                success=True,
                capture_http_ms=100.0,
                decode_ms=1.0,
                inference_ms=10.0,
                render_ms=0.5,
                encode_ms=0.5,
                server_processing_ms=12.0,
                total_loop_ms=112.0,
                fps=8.9,
            ),
            LatencyRecord(
                frame_index=2,
                timestamp=100.1,
                success=True,
                capture_http_ms=90.0,
                decode_ms=1.0,
                inference_ms=10.0,
                render_ms=0.5,
                encode_ms=0.5,
                server_processing_ms=12.0,
                total_loop_ms=102.0,
                fps=9.8,
            ),
        ]
        # 2 frames in 0.5s -> Throughput FPS = 2 / 0.5 = 4.0
        summary = summarize_metrics(records, elapsed_seconds=0.5)
        self.assertEqual(summary["total_frames"], 2)
        self.assertEqual(summary["success_frames"], 2)
        self.assertEqual(summary["failed_frames"], 0)
        self.assertAlmostEqual(summary["throughput_fps"], 4.0, places=2)
        self.assertGreater(summary["average_fps"], 8.0)

    def test_parse_baseline_csv(self):
        s0_csv = ROOT / "results" / "performance" / "benchmark_S0_20261017_161457.csv"
        if not s0_csv.exists():
            self.skipTest("Baseline S0 CSV không tồn tại")

        res = parse_benchmark_csv(s0_csv)
        self.assertEqual(res["total_frames"], 106)
        self.assertEqual(res["success_frames"], 106)
        self.assertEqual(res["failed_frames"], 0)
        self.assertGreater(res["throughput_fps"], 1.5)
        self.assertAlmostEqual(res["http_capture"]["avg"], 556.83, delta=1.0)
        self.assertAlmostEqual(res["inference"]["avg"], 13.91, delta=0.5)

    def test_replay_mode_worker_initialization(self):
        from scripts.esp32_dashboard import ESP32PipelineWorker

        replay_dir = ROOT / "dataset_raw" / "own_photos" / "session01_mask"
        worker = ESP32PipelineWorker(
            ip="192.168.1.153",
            model_path=ROOT / "runs" / "detect" / "results" / "train" / "antigravity_gpu_v1" / "weights" / "best.pt",
            replay_dir=replay_dir,
        )
        self.assertIsNotNone(worker)
        if replay_dir.exists():
            self.assertGreater(len(worker.replay_images), 0)


if __name__ == "__main__":
    unittest.main()
