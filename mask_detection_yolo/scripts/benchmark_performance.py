# -*- coding: utf-8 -*-
"""
scripts/benchmark_performance.py  —  Công Cụ Chạy Benchmark Đánh Giá Thực Nghiệm (CLI)
======================================================================================
Kích hoạt và theo dõi phiên benchmark thực nghiệm theo kịch bản chuẩn hóa (S0 - S4).
Tái sử dụng pipeline worker duy nhất đang chạy trên FastAPI Web Dashboard để tránh
khởi tạo thêm model YOLO hay gửi trùng request tới ESP32-CAM.

Các tính năng Giai Đoạn 9:
  - Hỗ trợ lặp lại kịch bản nhiều lần (--repeat N, ví dụ: 3 lần) để lấy trung bình thống kê.
  - Phân biệt rõ Throughput FPS (Completed / Elapsed) và Rolling Loop FPS.
  - Tự động so sánh Before/After với Baseline S0/S1 gốc trong results/performance/.
  - Xuất bảng so sánh chi tiết và file Before/After CSV.

Cách sử dụng trên Windows PowerShell:
    # 1. Khởi động Web Dashboard trước:
    python scripts/esp32_dashboard.py

    # 2. Chạy benchmark kịch bản S1 trong 60 giây (1 lần):
    python scripts/benchmark_performance.py --scenario S1 --duration 60

    # 3. Chạy benchmark kịch bản S0 lặp lại 3 lần:
    python scripts/benchmark_performance.py --scenario S0 --duration 60 --repeat 3
"""

import argparse
import csv
import io
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import requests

# Đảm bảo hiển thị UTF-8 trên Windows console
if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import BENCHMARK_SCENARIOS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chạy Benchmark Đánh Giá Hiệu Năng YOLO11n + ESP32-CAM (Giai Đoạn 9)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="S1",
        choices=list(BENCHMARK_SCENARIOS.keys()),
        help="Mã kịch bản kiểm thử: S0, S1, S2, S3, S4",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Thời lượng chạy mỗi phiên benchmark (giây)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Số lần lặp lại kịch bản để tính trung bình và độ lệch chuẩn (ví dụ: 3)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Địa chỉ máy chủ Web Dashboard",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        default=True,
        help="Tự động đối chiếu kết quả với baseline gốc trong results/performance/",
    )
    return parser.parse_args()


def render_progress_bar(current: float, total: float, width: int = 30) -> str:
    """Tạo thanh tiến độ văn bản ASCII."""
    ratio = min(1.0, max(0.0, current / total)) if total > 0 else 0.0
    filled = int(width * ratio)
    bar = "=" * filled + (">" if filled < width else "") + " " * (width - filled - 1 if filled < width else 0)
    return f"[{bar}] {ratio * 100.0:5.1f}%"


def parse_benchmark_csv(csv_path: Path) -> Dict[str, Any]:
    """Phân tích dữ liệu thô từ file CSV benchmark để tính các chỉ số thống kê."""
    if not csv_path.exists():
        return {}
    records = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    if not records:
        return {}

    total = len(records)
    success = [r for r in records if r.get("success", "").lower() == "true"]
    failed = total - len(success)

    timestamps = [float(r["timestamp"]) for r in records if r.get("timestamp")]
    elapsed = (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0.0
    throughput_fps = len(success) / elapsed if elapsed > 0 else 0.0

    def calc_percentile(sorted_list: List[float], p: float) -> float:
        if not sorted_list:
            return 0.0
        k = (len(sorted_list) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_list[int(k)]
        d0 = sorted_list[int(f)] * (c - k)
        d1 = sorted_list[int(c)] * (k - f)
        return d0 + d1

    def stats(key: str, positive_only: bool = False) -> Dict[str, float]:
        vals = [float(r[key]) for r in success if r.get(key) not in (None, "")]
        if positive_only:
            vals = [v for v in vals if v > 0.0]
        if not vals:
            return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "count": 0}
        vals.sort()
        n = len(vals)
        avg = sum(vals) / n
        p50 = calc_percentile(vals, 50.0)
        p95 = calc_percentile(vals, 95.0)
        return {"avg": round(avg, 2), "p50": round(p50, 2), "p95": round(p95, 2), "count": n}

    fps_vals = [float(r["fps"]) for r in success if r.get("fps")]
    avg_loop_fps = sum(fps_vals) / len(fps_vals) if fps_vals else 0.0
    has_cloud = any((r.get("inference_mode") or "").strip().lower() == "cloud" for r in records)

    return {
        "csv_path": str(csv_path),
        "inference_mode": "cloud" if has_cloud else "local",
        "total_frames": total,
        "success_frames": len(success),
        "failed_frames": failed,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_fps": round(throughput_fps, 2),
        "average_loop_fps": round(avg_loop_fps, 2),
        "http_capture": stats("capture_http_ms"),
        "jpeg_decode": stats("decode_ms"),
        "inference": stats("inference_ms"),
        "render": stats("render_ms"),
        "jpeg_encode": stats("encode_ms"),
        "server_processing": stats("server_processing_ms"),
        "total_loop": stats("total_loop_ms"),
        "cloud_inference": stats("cloud_inference_ms", positive_only=True),
        "cloud_server": stats("cloud_server_ms", positive_only=True),
        "cloud_rtt": stats("cloud_request_response_ms", positive_only=True),
        "cloud_transport": stats("cloud_transport_overhead_est_ms"),
        "e2e_display": stats("e2e_display_ms", positive_only=True),
    }


def find_baseline_csv(scenario: str) -> Optional[Path]:
    """Tìm file benchmark gốc trước tối ưu (ưu tiên file tạo ngày 2026-10-17)."""
    perf_dir = PROJECT_ROOT / "results" / "performance"
    if not perf_dir.exists():
        return None

    # Tìm file có tiền tố benchmark_{scenario}_20261017
    candidates = sorted(list(perf_dir.glob(f"benchmark_{scenario}_20261017*.csv")))
    if candidates:
        return candidates[0]

    # Dự phòng: tìm file CSV bất kỳ của scenario không phải phiên vừa chạy
    all_csvs = sorted(list(perf_dir.glob(f"benchmark_{scenario}_*.csv")))
    return all_csvs[0] if all_csvs else None


def run_single_benchmark(
    base_url: str,
    scenario: str,
    duration: float,
    run_idx: int,
    total_runs: int,
) -> Optional[Dict[str, Any]]:
    """Chạy một phiên benchmark đơn lẻ qua API."""
    prefix = f"[Lần {run_idx}/{total_runs}]" if total_runs > 1 else "[*]"
    print(f"\n{prefix} Khởi động phiên benchmark kịch bản {scenario} ({duration:.1f}s)...")

    start_url = f"{base_url}/api/benchmark/start"
    params = {"scenario": scenario, "duration": duration}
    try:
        start_res = requests.post(start_url, params=params, timeout=5.0)
        if start_res.status_code != 200:
            print(f"[!] Không thể bắt đầu benchmark: {start_res.text}")
            return None
    except Exception as e:
        print("[!] Lỗi khi gọi API bắt đầu benchmark:", e)
        return None

    status_url = f"{base_url}/api/benchmark/status"
    try:
        while True:
            time.sleep(1.0)
            try:
                st = requests.get(status_url, timeout=3.0).json()
            except Exception:
                continue

            elapsed = st.get("elapsed_seconds", 0.0)
            frames = st.get("recorded_frames", 0)
            rem = st.get("remaining_seconds", 0.0)

            pbar = render_progress_bar(elapsed, duration, width=24)
            print(
                f"\r  {prefix} {pbar} | {elapsed:4.1f}s / {duration:4.1f}s | Frames: {frames:4d} | Còn: {rem:4.1f}s",
                end="",
                flush=True,
            )

            if st.get("completed") or not st.get("is_active"):
                print()
                break

    except KeyboardInterrupt:
        print("\n\n[*] Đã nhận lệnh dừng từ bàn phím. Đang lưu kết quả...")
        try:
            requests.post(f"{base_url}/api/benchmark/stop", timeout=3.0)
        except Exception:
            pass

    try:
        final_status = requests.get(status_url, timeout=3.0).json()
        csv_file = final_status.get("csv_path")
        if csv_file and Path(csv_file).exists():
            return parse_benchmark_csv(Path(csv_file))
        return None
    except Exception as e:
        print("[!] Lỗi khi lấy kết quả cuối:", e)
        return None


def print_run_summary(run_stat: Dict[str, Any], scenario: str):
    """In bảng tóm tắt độ trễ (bao gồm Cloud Inference, Local <-> Cloud RTT và E2E Display)."""
    mode = run_stat.get("inference_mode", "local").upper()
    print("\n" + "=" * 74)
    print(f"  KẾT QUẢ ĐO ĐỘ TRỄ CHI TIẾT (Kịch bản {scenario} | Chế độ: {mode})")
    print("=" * 74)
    print(
        f"  Frames: {run_stat['success_frames']}/{run_stat['total_frames']} OK "
        f"| Failed: {run_stat['failed_frames']} "
        f"| Throughput: {run_stat['throughput_fps']:.2f} FPS "
        f"| Loop FPS: {run_stat['average_loop_fps']:.2f} FPS"
    )
    print("  " + "-" * 72)
    print(f"  {'Chỉ số độ trễ':<38} {'Avg (ms)':>10} {'P50 (ms)':>10} {'P95 (ms)':>10}")
    print("  " + "-" * 72)

    def _row(label: str, st: Dict[str, float]):
        print(f"  {label:<38} {st.get('avg', 0.0):>10.2f} {st.get('p50', 0.0):>10.2f} {st.get('p95', 0.0):>10.2f}")

    if mode == "CLOUD":
        _row("a) Cloud Inference (YOLO trên Cloud)", run_stat.get("cloud_inference", {}))
        _row("   Cloud Server Total (Decode+Infer)", run_stat.get("cloud_server", {}))
        _row("b) Local <-> Cloud Response (RTT)", run_stat.get("cloud_rtt", {}))
        _row("   Transport Overhead (Ước lượng)", run_stat.get("cloud_transport", {}))
    else:
        _row("Local YOLO Inference", run_stat.get("inference", {}))

    _row("   HTTP Capture (ESP32-CAM)", run_stat.get("http_capture", {}))
    _row("c) Total Pipeline Loop (Gateway)", run_stat.get("total_loop", {}))

    e2e = run_stat.get("e2e_display", {})
    if e2e.get("count", 0) > 0:
        _row(f"c) End-to-End Display (ACK, n={e2e['count']})", e2e)
    else:
        print("  c) End-to-End Display (Browser ACK)    Chưa có mẫu (hãy mở Dashboard trên trình duyệt)")

    print("=" * 74)


def print_comparison_table(baseline: Dict[str, Any], current: Dict[str, Any], scenario: str):
    """In bảng so sánh Before vs After với phần trăm cải thiện."""
    print("\n" + "=" * 74)
    print(f"  SO SÁNH HIỆU NĂNG BEFORE / AFTER (Kịch bản {scenario})")
    print("=" * 74)
    print(f"  Baseline (Before) : {Path(baseline.get('csv_path', '')).name}")
    print(f"  Current (After)   : {Path(current.get('csv_path', '')).name}")
    print("  ──────────────────────────────────────────────────────────────────────────")
    print("  {:<26} {:>12} {:>12} {:>10} {:>10}".format(
        "Chỉ số đánh giá", "Before (Gốc)", "After (Tối ưu)", "Delta", "Cải thiện"
    ))
    print("  " + "-" * 72)

    def row(label: str, v_before: float, v_after: float, higher_is_better: bool = False, unit: str = "ms"):
        delta = v_after - v_before
        if v_before > 0:
            if higher_is_better:
                pct = ((v_after - v_before) / v_before) * 100.0
            else:
                pct = ((v_before - v_after) / v_before) * 100.0
            pct_str = f"{pct:+.1f}%"
        else:
            pct_str = "N/A"

        delta_sign = "+" if delta > 0 else ""
        unit_str = f" {unit}" if unit else ""
        print(f"  {label:<26} {v_before:>10.2f}{unit_str} {v_after:>10.2f}{unit_str} {delta_sign}{delta:>8.2f}{unit_str} {pct_str:>10}")

    # FPS
    row("Throughput FPS", baseline.get("throughput_fps", 0), current.get("throughput_fps", 0), higher_is_better=True, unit="FPS")
    row("Rolling Loop FPS", baseline.get("average_loop_fps", 0), current.get("average_loop_fps", 0), higher_is_better=True, unit="FPS")
    print("  " + "-" * 72)

    # Latencies
    metrics_list = [
        ("HTTP Capture (Avg)", baseline["http_capture"]["avg"], current["http_capture"]["avg"]),
        ("HTTP Capture (P50)", baseline["http_capture"]["p50"], current["http_capture"]["p50"]),
        ("HTTP Capture (P95)", baseline["http_capture"]["p95"], current["http_capture"]["p95"]),
        ("YOLO Inference (Avg)", baseline["inference"]["avg"], current["inference"]["avg"]),
        ("YOLO Inference (P50)", baseline["inference"]["p50"], current["inference"]["p50"]),
        ("YOLO Inference (P95)", baseline["inference"]["p95"], current["inference"]["p95"]),
        ("Server Process (Avg)", baseline["server_processing"]["avg"], current["server_processing"]["avg"]),
        ("Server Process (P50)", baseline["server_processing"]["p50"], current["server_processing"]["p50"]),
        ("Server Process (P95)", baseline["server_processing"]["p95"], current["server_processing"]["p95"]),
        ("Total Loop (Avg)", baseline["total_loop"]["avg"], current["total_loop"]["avg"]),
        ("Total Loop (P50)", baseline["total_loop"]["p50"], current["total_loop"]["p50"]),
        ("Total Loop (P95)", baseline["total_loop"]["p95"], current["total_loop"]["p95"]),
    ]

    for label, b_val, a_val in metrics_list:
        row(label, b_val, a_val, higher_is_better=False, unit="ms")

    print("  " + "-" * 72)
    b_fails = baseline.get("failed_frames", 0)
    a_fails = current.get("failed_frames", 0)
    print(f"  Failed Frames              {b_fails:>10d}      {a_fails:>10d}      {a_fails - b_fails:>+8d}        N/A")
    print("=" * 74 + "\n")


def main():
    if sys.platform == "win32":
        os.system("color")

    args = parse_args()
    base_url = args.api_url.rstrip("/")
    scenario_desc = BENCHMARK_SCENARIOS.get(args.scenario, "Không xác định")

    print("\n" + "=" * 70)
    print("  ESP32-CAM YOLO11n — Giai Đoạn 9: Optimization & Benchmark")
    print("=" * 70)
    print(f"  Kịch bản     : {args.scenario} — {scenario_desc}")
    print(f"  Thời lượng   : {args.duration:.1f} giây / lần")
    print(f"  Số lần lặp   : {args.repeat} lần")
    print(f"  API Endpoint : {base_url}")
    print("=" * 70 + "\n")

    # 1. Kiểm tra kết nối tới Web Dashboard
    try:
        res = requests.get(f"{base_url}/api/status", timeout=3.0)
        if res.status_code != 200:
            print(f"[!] Lỗi: Máy chủ Web trả về HTTP {res.status_code}")
            sys.exit(1)
        status_data = res.json()
    except Exception as e:
        print("[!] Không thể kết nối tới Web Dashboard tại:", base_url)
        print("    Lý do:", e)
        print("\n    HƯỚNG DẪN:")
        print("    1. Hãy khởi động Web Dashboard trước trong một cửa sổ terminal khác:")
        print("       python scripts/esp32_dashboard.py")
        print("    2. Đảm bảo Dashboard đang hoạt động tại http://127.0.0.1:8000/\n")
        sys.exit(1)

    camera_status = status_data.get("camera_status", "unknown")
    print(f"[*] Trạng thái camera: {camera_status.upper()} (Nguồn: {status_data.get('camera_ip')})")

    # 2. Thực hiện các phiên benchmark
    all_runs_stats = []
    for i in range(1, args.repeat + 1):
        stat = run_single_benchmark(base_url, args.scenario, args.duration, i, args.repeat)
        if stat:
            all_runs_stats.append(stat)
        if i < args.repeat:
            print("[*] Nghỉ 2.0s trước lượt lặp tiếp theo...")
            time.sleep(2.0)

    if not all_runs_stats:
        print("[!] Không thu thập được dữ liệu benchmark nào.")
        sys.exit(1)

    # 3. Tổng hợp kết quả các lần chạy
    latest_run = all_runs_stats[-1]
    print_run_summary(latest_run, args.scenario)

    if len(all_runs_stats) > 1:
        print("\n" + "=" * 70)
        print(f"  TỔNG HỢP {len(all_runs_stats)} LẦN CHẠY ({args.scenario})")
        print("=" * 70)
        t_fps_list = [r["throughput_fps"] for r in all_runs_stats]
        r_fps_list = [r["average_loop_fps"] for r in all_runs_stats]
        cap_list = [r["http_capture"]["avg"] for r in all_runs_stats]
        inf_list = [r["inference"]["avg"] for r in all_runs_stats]
        tot_list = [r["total_loop"]["avg"] for r in all_runs_stats]

        def mean_std(vals):
            m = sum(vals) / len(vals)
            variance = sum((x - m) ** 2 for x in vals) / len(vals)
            return m, math.sqrt(variance)

        m_tfps, s_tfps = mean_std(t_fps_list)
        m_rfps, s_rfps = mean_std(r_fps_list)
        m_cap, s_cap = mean_std(cap_list)
        m_inf, s_inf = mean_std(inf_list)
        m_tot, s_tot = mean_std(tot_list)

        print(f"  Throughput FPS    : {m_tfps:.2f} ± {s_tfps:.2f} FPS")
        print(f"  Rolling Loop FPS  : {m_rfps:.2f} ± {s_rfps:.2f} FPS")
        print(f"  HTTP Capture Avg  : {m_cap:.2f} ± {s_cap:.2f} ms")
        print(f"  Inference Avg     : {m_inf:.2f} ± {s_inf:.2f} ms")
        print(f"  Total Loop Avg    : {m_tot:.2f} ± {s_tot:.2f} ms")
        if any(r.get("inference_mode") == "cloud" for r in all_runs_stats):
            m_cinf, s_cinf = mean_std([r["cloud_inference"]["avg"] for r in all_runs_stats])
            m_crtt, s_crtt = mean_std([r["cloud_rtt"]["avg"] for r in all_runs_stats])
            print(f"  Cloud Infer Avg   : {m_cinf:.2f} ± {s_cinf:.2f} ms")
            print(f"  Local-Cloud RTT   : {m_crtt:.2f} ± {s_crtt:.2f} ms")
        e2e_runs = [r["e2e_display"]["avg"] for r in all_runs_stats if r.get("e2e_display", {}).get("count", 0) > 0]
        if e2e_runs:
            m_e2e, s_e2e = mean_std(e2e_runs)
            print(f"  E2E Display Avg   : {m_e2e:.2f} ± {s_e2e:.2f} ms")
        print("=" * 70)

    # 4. Đối chiếu với Baseline nếu có
    if args.compare_baseline:
        baseline_file = find_baseline_csv(args.scenario)
        if baseline_file and baseline_file.resolve() != Path(latest_run["csv_path"]).resolve():
            baseline_stats = parse_benchmark_csv(baseline_file)
            if baseline_stats:
                print_comparison_table(baseline_stats, latest_run, args.scenario)
        else:
            print(f"[*] Chưa có file baseline trước đó cho kịch bản {args.scenario} để đối chiếu.")


if __name__ == "__main__":
    main()
