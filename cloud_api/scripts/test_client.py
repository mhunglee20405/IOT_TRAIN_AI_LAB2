# -*- coding: utf-8 -*-
"""
cloud_api/scripts/test_client.py — Smoke Test Client cho Cloud Inference API.
=============================================================================
Kiểm thử tự động các chức năng của Cloud API:
  1. Kiểm tra GET /health
  2. Gửi ảnh mẫu tới POST /v1/infer
  3. Đo lường:
     - cloud_response_ms (Client round-trip time)
     - cloud_inference_ms (Thời gian YOLO trên server)
     - cloud_decode_ms, cloud_postprocess_ms
     - Network & Serialization overhead = cloud_response_ms - cloud_total_ms
  4. In bảng kết quả chi tiết.

Sử dụng:
  # Chạy test với ảnh mặc định cục bộ:
  python cloud_api/scripts/test_client.py

  # Test qua SSH Tunnel hoặc Cloud IP với API Key:
  python cloud_api/scripts/test_client.py --url http://127.0.0.1:8000 --api-key YOUR_KEY --iterations 5
"""

import argparse
import io
from pathlib import Path
import sys
import time
from typing import Optional

import requests

# Đảm bảo hiển thị UTF-8 trên Windows console
if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]


def find_default_image() -> Optional[Path]:
    """Tìm một ảnh mẫu có sẵn trong repo."""
    candidates = [
        ROOT / "dataset_raw" / "own_photos" / "session01_mask" / "esp32_20261017_132116_883_1.jpg",
        ROOT / "dataset" / "images" / "val",
    ]
    for c in candidates:
        if c.is_file():
            return c
        elif c.is_dir():
            for f in c.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    return f
    return None


def run_smoke_test(
    api_url: str,
    image_path: Path,
    api_key: Optional[str] = None,
    iterations: int = 3,
):
    print("=" * 70)
    print("  YOLO11n Cloud Inference Server — Smoke Test Client")
    print("=" * 70)
    print(f"  Target Server : {api_url}")
    print(f"  Test Image    : {image_path} ({image_path.stat().st_size} bytes)")
    print(f"  API Key Auth  : {'BẬT (Masked)' if api_key else 'TẮT / Chưa cấu hình'}")
    print(f"  Iterations    : {iterations}")
    print("=" * 70 + "\n")

    base_url = api_url.rstrip("/")
    session = requests.Session()
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    # 1. Health check
    print("[1/3] Kiểm tra GET /health...")
    t_h_start = time.perf_counter()
    try:
        h_res = session.get(f"{base_url}/health", timeout=5.0)
    except Exception as e:
        print(f"[!] Lỗi kết nối tới server: {e}")
        return False

    h_ms = (time.perf_counter() - t_h_start) * 1000.0
    if h_res.status_code != 200:
        print(f"[!] Health check thất bại: HTTP {h_res.status_code} - {h_res.text}")
        return False

    h_data = h_res.json()
    print(f"    -> OK! HTTP 200 ({h_ms:.1f}ms)")
    print(f"    -> Model: {h_data.get('model_version')} | Device: {h_data.get('device')} | Loaded: {h_data.get('model_loaded')}")
    print(f"    -> Conf Floor: {h_data.get('conf_floor')} | Accept Thresh: {h_data.get('accept_thresh')}\n")

    # 2. Đọc ảnh
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # 3. Chạy iterations suy luận
    print(f"[2/3] Gửi request POST /v1/infer ({iterations} lần)...")
    round_trip_times = []
    cloud_infer_times = []
    cloud_total_times = []

    last_data = None
    for i in range(1, iterations + 1):
        frame_id = f"smoke_frame_{i}_{int(time.time())}"
        files = {"file": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        data = {"frame_id": frame_id}

        t_req_start = time.perf_counter()
        res = session.post(
            f"{base_url}/v1/infer",
            files=files,
            data=data,
            headers=headers,
            timeout=10.0,
        )
        t_req_end = time.perf_counter()

        rt_ms = (t_req_end - t_req_start) * 1000.0
        if res.status_code != 200:
            print(f"[!] Lần {i} thất bại: HTTP {res.status_code} - {res.text}")
            return False

        res_json = res.json()
        last_data = res_json
        timing = res_json.get("timing_ms", {})
        c_infer = timing.get("cloud_inference_ms", 0.0)
        c_tot = timing.get("cloud_total_ms", 0.0)

        round_trip_times.append(rt_ms)
        cloud_infer_times.append(c_infer)
        cloud_total_times.append(c_tot)

        net_overhead = max(0.0, rt_ms - c_tot)
        print(
            f"    Iter #{i}: Round-trip: {rt_ms:6.1f}ms | "
            f"Cloud Infer: {c_infer:5.1f}ms | Cloud Total: {c_tot:5.1f}ms | "
            f"Net Overhead: {net_overhead:5.1f}ms | "
            f"Faces: {res_json.get('counts', {}).get('total', 0)}"
        )

    # 4. Tổng kết số liệu
    print("\n" + "=" * 70)
    print("  KẾT QUẢ KIỂM THỬ SUY LUẬN (SMOKE TEST SUMMARY)")
    print("=" * 70)
    avg_rt = sum(round_trip_times) / len(round_trip_times)
    avg_inf = sum(cloud_infer_times) / len(cloud_infer_times)
    avg_tot = sum(cloud_total_times) / len(cloud_total_times)
    avg_net = max(0.0, avg_rt - avg_tot)

    print(f"  • Response Time (Client Round-trip) : {avg_rt:.2f} ms (Min: {min(round_trip_times):.1f}, Max: {max(round_trip_times):.1f})")
    print(f"  • Cloud Server Processing Total     : {avg_tot:.2f} ms")
    print(f"  • Cloud YOLO Inference Time         : {avg_inf:.2f} ms")
    print(f"  • Network & Serialization Overhead  : {avg_net:.2f} ms ({(avg_net / avg_rt * 100):.1f}% tổng thời gian)")
    print(f"  • Image Dimensions                  : {last_data['image']['width']}x{last_data['image']['height']} px")
    print(f"  • Detections Count                  : {last_data['counts']}")

    if last_data.get("detections"):
        print("  • Chi tiết bounding boxes:")
        for idx, det in enumerate(last_data["detections"], 1):
            print(
                f"    [{idx}] Label: {det['label']:<8} | Conf: {det['confidence']:.3f} | "
                f"Box: {det['bbox_xyxy']} | Raw: {det['raw_class_name']} | Unknown: {det['is_unknown']}"
            )
    else:
        print("  • Không phát hiện khuôn mặt nào trong ảnh test.")

    print("=" * 70)
    print("[*] SMOKE TEST THÀNH CÔNG HOÀN TOÀN!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke test client cho Cloud Inference API")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="URL server (ví dụ: http://localhost:8000)")
    parser.add_argument("--image", default=None, help="Đường dẫn ảnh kiểm thử (.jpg/.png)")
    parser.add_argument("--api-key", default=None, help="API Key nếu server yêu cầu X-API-Key")
    parser.add_argument("--iterations", type=int, default=3, help="Số lần gửi request lặp lại")
    args = parser.parse_args()

    img_path = Path(args.image) if args.image else find_default_image()
    if not img_path or not img_path.exists():
        print(f"[!] Không tìm thấy ảnh kiểm thử tại: {img_path}")
        sys.exit(1)

    success = run_smoke_test(
        api_url=args.url,
        image_path=img_path,
        api_key=args.api_key,
        iterations=args.iterations,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
