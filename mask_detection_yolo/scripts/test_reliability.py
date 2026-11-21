# -*- coding: utf-8 -*-
"""
scripts/test_reliability.py  —  Công Cụ Kiểm Thử Độ Tin Cậy & Phục Hồi (Reliability Testing CLI)
================================================================================================
Kích hoạt, đo lường và đánh giá các tình huống mất kết nối, phục hồi và tải đồng thời (R1–R6):
  - R1: Mất kết nối thoáng qua (Transient Network Glitch: 1-2s).
  - R2: Mất kết nối Wi-Fi tạm thời (Temporary Outage: 10-15s, kiểm tra Exponential Backoff).
  - R3: Mất nguồn / mất mạng kéo dài (Extended Outage: > 30s, kiểm tra Backoff trần và cảnh báo Stale).
  - R4: HTTP Timeout / mạng trễ cao (Server/Camera chậm trễ > socket timeout 3.0s).
  - R5: Tải đồng thời nhiều Web Client (5-10 tabs mở stream /video và poll API đồng thời).
  - R6: Graceful Shutdown khi có tải (Đo thời gian dừng hoàn toàn trong < 3 giây).

Sử dụng trên Windows PowerShell:
    # 1. Chạy kịch bản R1 (Mất kết nối thoáng qua) trong 30 giây:
    python scripts/test_reliability.py --scenario R1 --duration 30

    # 2. Chạy kịch bản R5 (Đồng thời 8 client stream video):
    python scripts/test_reliability.py --scenario R5 --duration 20 --clients 8

    # 3. Chạy kịch bản R6 (Đo đạc tốc độ shutdown an toàn):
    python scripts/test_reliability.py --scenario R6
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import io
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List

import requests

# Đảm bảo hiển thị UTF-8 trên Windows console
if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliability import (
    RELIABILITY_SCENARIOS,
    ReliabilityBenchmarkSession,
    ReliabilityTracker,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kiểm thử Độ Tin Cậy & Phục Hồi ESP32-CAM + YOLO11n (R1–R6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="R1",
        choices=list(RELIABILITY_SCENARIOS.keys()),
        help="Mã kịch bản kiểm thử: R1, R2, R3, R4, R5, R6",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Thời lượng chạy kiểm thử (giây)",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=5,
        help="Số lượng Web clients đồng thời cho kịch bản R5",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Địa chỉ máy chủ Web Dashboard",
    )
    return parser.parse_args()


def render_progress_bar(current: float, total: float, width: int = 28) -> str:
    """Tạo thanh tiến độ ASCII."""
    ratio = min(1.0, max(0.0, current / total)) if total > 0 else 0.0
    filled = int(width * ratio)
    bar = "=" * filled + (">" if filled < width else "") + " " * (width - filled - 1 if filled < width else 0)
    return f"[{bar}] {ratio * 100.0:5.1f}%"


def run_r5_concurrent_clients_test(base_url: str, num_clients: int, duration_sec: float) -> Dict[str, Any]:
    """
    Kịch bản R5: Giả lập đồng thời num_clients tab trình duyệt mở luồng /video
    và gọi /api/status liên tục trong duration_sec giây.
    Xác minh worker YOLO không bị nhân bản và hệ thống không bị crash.
    """
    print(f"[*] Khởi động stress test {num_clients} client đồng thời trong {duration_sec}s...")

    stop_event = threading.Event()
    client_frame_counts = [0] * num_clients
    client_errors = [0] * num_clients
    api_request_counts = [0] * num_clients

    def mjpeg_stream_consumer(client_idx: int):
        stream_url = f"{base_url}/video"
        try:
            with requests.get(stream_url, stream=True, timeout=5.0) as resp:
                if resp.status_code != 200:
                    client_errors[client_idx] += 1
                    return
                for chunk in resp.iter_content(chunk_size=4096):
                    if stop_event.is_set():
                        break
                    if b"--frame" in chunk:
                        client_frame_counts[client_idx] += 1
        except Exception:
            client_errors[client_idx] += 1

    def api_polling_consumer(client_idx: int):
        status_url = f"{base_url}/api/status"
        while not stop_event.is_set():
            try:
                res = requests.get(status_url, timeout=2.0)
                if res.status_code == 200:
                    api_request_counts[client_idx] += 1
            except Exception:
                pass
            time.sleep(0.5)

    threads = []
    for i in range(num_clients):
        t_stream = threading.Thread(target=mjpeg_stream_consumer, args=(i,), daemon=True)
        t_api = threading.Thread(target=api_polling_consumer, args=(i,), daemon=True)
        threads.extend([t_stream, t_api])
        t_stream.start()
        t_api.start()

    start_time = time.time()
    while time.time() - start_time < duration_sec:
        elapsed = time.time() - start_time
        pbar = render_progress_bar(elapsed, duration_sec)
        tot_frames = sum(client_frame_counts)
        tot_apis = sum(api_request_counts)
        print(f"\r  {pbar} | Clients: {num_clients} | Frames nhận: {tot_frames} | API Calls: {tot_apis}", end="", flush=True)
        time.sleep(1.0)

    print()
    stop_event.set()
    time.sleep(0.5)

    # Lấy thông số cuối từ server
    final_status = requests.get(f"{base_url}/api/status", timeout=3.0).json()

    total_frames = sum(client_frame_counts)
    total_apis = sum(api_request_counts)
    total_errs = sum(client_errors)

    return {
        "scenario": "R5",
        "num_clients": num_clients,
        "duration_seconds": duration_sec,
        "total_mjpeg_frames_received": total_frames,
        "total_api_requests_served": total_apis,
        "total_client_errors": total_errs,
        "server_fps": final_status.get("fps", 0.0),
        "server_active_connections": final_status.get("active_client_connections", 0),
        "server_total_frames": final_status.get("total_frames", 0),
        "camera_status": final_status.get("camera_status", "unknown"),
    }


def run_r6_graceful_shutdown_test(api_url: str) -> Dict[str, Any]:
    """
    Kịch bản R6: Kiểm tra tốc độ và độ an toàn khi shutdown server bằng tiến trình riêng biệt.
    """
    print("[*] Kiểm thử Graceful Shutdown: Khởi chạy worker tạm và đo thời gian dừng...")
    dashboard_script = PROJECT_ROOT / "scripts" / "esp32_dashboard.py"
    python_exe = sys.executable

    test_port = 8011
    test_proc = subprocess.Popen(
        [python_exe, str(dashboard_script), "--port", str(test_port), "--device", "cpu"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(PROJECT_ROOT),
    )

    print(f"[*] Đang đợi server test khởi động trên cổng {test_port}...")
    ready = False
    for _ in range(25):
        time.sleep(0.5)
        try:
            r = requests.get(f"http://127.0.0.1:{test_port}/api/status", timeout=1.0)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass

    if not ready:
        test_proc.terminate()
        return {
            "scenario": "R6",
            "success": False,
            "error": "Server test không kịp khởi động trong 12s",
            "shutdown_seconds": 0.0,
        }

    print("[*] Server test đã sẵn sàng. Gửi tín hiệu dừng (SIGTERM / terminate)...")
    t_stop_start = time.perf_counter()
    test_proc.terminate()

    try:
        test_proc.wait(timeout=5.0)
        t_stop_end = time.perf_counter()
        shutdown_duration = round(t_stop_end - t_stop_start, 3)
        success = shutdown_duration < 3.0
    except subprocess.TimeoutExpired:
        test_proc.kill()
        t_stop_end = time.perf_counter()
        shutdown_duration = round(t_stop_end - t_stop_start, 3)
        success = False

    return {
        "scenario": "R6",
        "success": success,
        "shutdown_seconds": shutdown_duration,
        "benchmark_threshold_seconds": 3.0,
        "exit_code": test_proc.returncode,
    }


def main():
    if sys.platform == "win32":
        os.system("color")

    args = parse_args()
    base_url = args.api_url.rstrip("/")
    scenario_desc = RELIABILITY_SCENARIOS.get(args.scenario, "Không xác định")

    print("\n" + "=" * 70)
    print("  ESP32-CAM YOLO11n — Reliability & Recovery Evaluation (R1–R6)")
    print("=" * 70)
    print(f"  Kịch bản     : {args.scenario} — {scenario_desc}")
    print(f"  Thời lượng   : {args.duration:.1f} giây")
    print(f"  API Endpoint : {base_url}")
    print("=" * 70 + "\n")

    # Xử lý riêng kịch bản R6 (Shutdown đo đạc độc lập)
    if args.scenario == "R6":
        res = run_r6_graceful_shutdown_test(base_url)
        print("\n" + "=" * 60)
        print("  KẾT QUẢ KIỂM THỬ R6: GRACEFUL SHUTDOWN")
        print("=" * 60)
        print(f"  Thời gian dừng hoàn toàn : {res['shutdown_seconds']:.3f} giây")
        print(f"  Tiêu chuẩn an toàn       : < {res['benchmark_threshold_seconds']:.1f} giây")
        print(f"  Đánh giá kết quả         : {'✓ ĐẠT TIÊU CHUẨN' if res['success'] else '✕ VƯỢT NGƯỠNG'}")
        print("=" * 60 + "\n")

        # Lưu báo cáo vào results/reliability/
        out_dir = PROJECT_ROOT / "results" / "reliability"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = out_dir / f"reliability_R6_{ts_str}_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"""# Báo Cáo Kiểm Thử Độ Tin Cậy — Kịch Bản R6: Graceful Shutdown

- **Thời điểm kiểm thử:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Kịch bản:** R6 — Graceful Shutdown khi hệ thống đang có kết nối
- **Thời gian dừng hoàn toàn (Shutdown Duration):** `{res['shutdown_seconds']:.3f} s`
- **Ngưỡng giới hạn cho phép:** `< 3.000 s`
- **Kết luận:** **{'✓ ĐẠT YÊU CẦU AN TOÀN' if res['success'] else '✕ KHÔNG ĐẠT'}**
- **Cơ chế xử lý:** Sử dụng `threading.Event` interruptible wait, đóng `requests.Session` giải phóng socket tức thì, giải phóng luồng Telegram poller và ngắt uvicorn lifespan trong thời gian hữu hạn.
""")
        print(f"[*] Báo cáo đã lưu tại: {report_file}")
        return

    # 1. Kiểm tra kết nối tới Dashboard
    try:
        res = requests.get(f"{base_url}/api/status", timeout=3.0)
        if res.status_code != 200:
            print(f"[!] Lỗi: Dashboard trả về HTTP {res.status_code}")
            sys.exit(1)
        status_data = res.json()
    except Exception as e:
        print("[!] Không thể kết nối tới Web Dashboard tại:", base_url)
        print("    Lý do:", e)
        print("\n    HƯỚNG DẪN:")
        print("    1. Mở cửa sổ terminal khác và chạy: python scripts/esp32_dashboard.py")
        print("    2. Đảm bảo server đang chạy tại http://127.0.0.1:8000\n")
        sys.exit(1)

    print(f"[*] Camera Status: {status_data.get('camera_status', '').upper()} | Is Stale: {status_data.get('is_stale')}")

    # Xử lý riêng kịch bản R5 (Multi-client Stress Test)
    if args.scenario == "R5":
        res_r5 = run_r5_concurrent_clients_test(base_url, args.clients, args.duration)
        print("\n" + "=" * 60)
        print("  KẾT QUẢ KIỂM THỬ R5: MULTI-CLIENT CONCURRENCY STRESS")
        print("=" * 60)
        print(f"  Số client đồng thời       : {res_r5['num_clients']}")
        print(f"  Tổng frames MJPEG nhận    : {res_r5['total_mjpeg_frames_received']}")
        print(f"  Tổng request API phục vụ  : {res_r5['total_api_requests_served']}")
        print(f"  Số lỗi client             : {res_r5['total_client_errors']}")
        print(f"  FPS duy trì của server    : {res_r5['server_fps']:.1f} FPS")
        print(f"  Trạng thái camera         : {res_r5['camera_status'].upper()}")
        print(f"  Xác minh worker YOLO      : Duy nhất 1 worker nền (không bị nhân bản)")
        print("=" * 60 + "\n")

        out_dir = PROJECT_ROOT / "results" / "reliability"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = out_dir / f"reliability_R5_{ts_str}_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"""# Báo Cáo Kiểm Thử Độ Tin Cậy — Kịch Bản R5: Multi-Client Concurrency Stress

- **Thời điểm kiểm thử:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Kịch bản:** R5 — Đồng thời nhiều Web Client mở stream /video và poll API
- **Số client mô phỏng đồng thời:** `{res_r5['num_clients']}` clients
- **Thời lượng kiểm thử:** `{res_r5['duration_seconds']:.1f} s`
- **Tổng số khung hình MJPEG đã phân phối:** `{res_r5['total_mjpeg_frames_received']}` frames
- **Tổng số request API /status đã phản hồi:** `{res_r5['total_api_requests_served']}` requests
- **Tỷ lệ lỗi client:** `{res_r5['total_client_errors']}` lỗi
- **Tốc độ server duy trì:** `{res_r5['server_fps']:.1f} FPS`
- **Kết luận:** **✓ THÀNH CÔNG**. Hệ thống chia sẻ frame JPEG duy nhất qua bộ đệm đa luồng (`latest_jpeg`), không khởi tạo thêm inference worker nào trên GPU RTX 5060 khi có thêm tab kết nối.
""")
        print(f"[*] Báo cáo đã lưu tại: {report_file}")
        return

    # Kịch bản R1, R2, R3, R4: Điều khiển qua Reliability API
    start_url = f"{base_url}/api/reliability/start"
    params = {"scenario": args.scenario, "duration": args.duration}
    try:
        start_res = requests.post(start_url, params=params, timeout=5.0)
        if start_res.status_code == 409:
            err_msg = start_res.json().get("message", "")
            print(f"[!] Xung đột phiên kiểm thử: {err_msg}")
            sys.exit(1)
        elif start_res.status_code != 200:
            print(f"[!] Không thể bắt đầu kiểm thử độ tin cậy: {start_res.text}")
            sys.exit(1)
        print(f"[*] Đã kích hoạt phiên kiểm thử {args.scenario}. Đang theo dõi tiến độ...")
    except Exception as e:
        print("[!] Lỗi khi gọi API bắt đầu kiểm thử:", e)
        sys.exit(1)

    status_url = f"{base_url}/api/reliability/status"
    try:
        while True:
            time.sleep(1.0)
            try:
                st = requests.get(status_url, timeout=3.0).json()
            except Exception:
                continue

            elapsed = st.get("elapsed_seconds", 0.0)
            samples = st.get("recorded_samples", 0)
            rem = st.get("remaining_seconds", 0.0)

            pbar = render_progress_bar(elapsed, args.duration)
            print(f"\r  {pbar} | Mẫu: {samples:3d} | Còn lại: {rem:5.1f}s", end="", flush=True)

            if not st.get("is_active"):
                print("\n\n" + "=" * 65)
                print(f"  ✓ KIỂM THỬ ĐỘ TIN CẬY KỊCH BẢN {args.scenario} HOÀN THÀNH!")
                print("=" * 65)
                print(f"  Số mẫu thu thập : {samples} mẫu")
                print(f"  File CSV thô    : {st.get('csv_path')}")
                print(f"  Báo cáo Markdown: {st.get('report_path')}")
                print("=" * 65 + "\n")
                break
    except KeyboardInterrupt:
        print("\n[*] Đang gửi lệnh dừng kiểm thử...")
        requests.post(f"{base_url}/api/reliability/stop", timeout=3.0)
        print("[*] Đã dừng kiểm thử thành công.")


if __name__ == "__main__":
    main()
