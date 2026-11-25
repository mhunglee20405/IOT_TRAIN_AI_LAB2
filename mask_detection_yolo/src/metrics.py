# -*- coding: utf-8 -*-
"""
src/metrics.py  —  Module Đo Lường Hiệu Năng & Đánh Giá Thực Nghiệm (Performance & Benchmark)
===========================================================================================
Cung cấp các công cụ:
  - Đo thời gian chi tiết từng công đoạn bằng time.perf_counter():
      + capture_http_ms: Thời gian request & tải ảnh JPEG từ ESP32-CAM qua mạng HTTP.
      + decode_ms: Thời gian giải mã JPEG thành mảng numpy BGR.
      + inference_ms: Thời gian mô hình YOLO11n suy luận (đồng bộ CUDA nếu dùng GPU).
      + render_ms: Thời gian vẽ bounding box, nhãn và thanh thông tin.
      + encode_ms: Thời gian nén frame thành JPEG cho luồng MJPEG.
      + server_processing_ms: Tổng thời gian máy chủ xử lý (decode + inference + render + encode).
      + total_loop_ms: Tổng thời gian một chu kỳ (capture_http + server_processing).
  - Tính toán các chỉ số thống kê thực tế: Average, P50 (Median), P95, Min, Max, FPS.
  - Quản lý phiên benchmark theo kịch bản (S0 - S4) trong thời lượng xác định (ví dụ 60s).
  - Xuất dữ liệu chi tiết ra CSV và tạo báo cáo markdown tổng hợp tại results/performance/.
"""

from collections import deque
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional


# Danh mục các kịch bản thực nghiệm chuẩn hóa
BENCHMARK_SCENARIOS = {
    "S0": "Không có người (Khung cảnh trống)",
    "S1": "Một người không đeo khẩu trang",
    "S2": "Một người đeo khẩu trang đúng cách",
    "S3": "Nhiều người, trạng thái khẩu trang khác nhau",
    "S4": "Mặt nhỏ, quay nghiêng hoặc góc quay thiếu sáng",
}


@dataclass
class LatencyRecord:
    """Bản ghi đo lường chi tiết cho một khung hình."""
    frame_index: int
    timestamp: float
    success: bool
    capture_http_ms: float = 0.0
    decode_ms: float = 0.0
    inference_ms: float = 0.0
    render_ms: float = 0.0
    encode_ms: float = 0.0
    server_processing_ms: float = 0.0
    total_loop_ms: float = 0.0
    fps: float = 0.0
    mask_count: int = 0
    no_mask_count: int = 0
    unknown_count: int = 0
    total_faces: int = 0
    inference_mode: str = "local"
    cloud_inference_ms: float = 0.0
    cloud_server_ms: float = 0.0
    cloud_decode_ms: float = 0.0
    cloud_predict_wall_ms: float = 0.0
    cloud_request_response_ms: float = 0.0
    cloud_transport_overhead_est_ms: float = 0.0
    e2e_display_ms: float = 0.0
    error_msg: str = ""


def calc_percentile(values: List[float], percentile: float) -> float:
    """
    Tính bách phân vị (percentile: 0 - 100) theo phương pháp nội suy tuyến tính chuẩn.
    Trả về 0.0 nếu danh sách rỗng.
    """
    if not values:
        return 0.0
    clean_vals = sorted([float(v) for v in values if not math.isnan(v)])
    if not clean_vals:
        return 0.0
    if len(clean_vals) == 1:
        return clean_vals[0]

    k = (len(clean_vals) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return clean_vals[int(k)]
    d0 = clean_vals[f] * (c - k)
    d1 = clean_vals[c] * (k - f)
    return d0 + d1


def summarize_metrics(
    records: List[LatencyRecord], elapsed_seconds: Optional[float] = None
) -> Dict[str, Any]:
    """
    Tổng hợp các chỉ số thống kê (Average, P50, P95, Min, Max) từ danh sách bản ghi.
    Phân biệt rõ:
      - average_fps: Tốc độ FPS trung bình theo chu kỳ lặp (1000 / mean_loop_ms).
      - throughput_fps: Tốc độ thông lượng thực tế (completed frames / elapsed seconds).
    """
    if not records:
        return {
            "total_frames": 0,
            "success_frames": 0,
            "failed_frames": 0,
            "success_rate_pct": 0.0,
            "current_fps": 0.0,
            "average_fps": 0.0,
            "throughput_fps": 0.0,
            "latency_ms": {},
        }

    total_count = len(records)
    success_records = [r for r in records if r.success]
    failed_count = total_count - len(success_records)
    success_rate = (len(success_records) / total_count * 100.0) if total_count > 0 else 0.0

    current_fps = success_records[-1].fps if success_records else 0.0

    # 1. Tính Rolling Loop FPS từ tổng thời gian chu kỳ
    loop_times = [r.total_loop_ms for r in success_records if r.total_loop_ms > 0]
    avg_fps = (1000.0 / (sum(loop_times) / len(loop_times))) if loop_times else 0.0

    # 2. Tính Throughput FPS (Completed Frames / Elapsed Seconds)
    if elapsed_seconds and elapsed_seconds > 0:
        throughput_fps = len(success_records) / elapsed_seconds
    else:
        throughput_fps = avg_fps

    # Các metric cần thống kê
    metric_fields = [
        "capture_http_ms",
        "decode_ms",
        "inference_ms",
        "render_ms",
        "encode_ms",
        "server_processing_ms",
        "total_loop_ms",
        "cloud_inference_ms",
        "cloud_server_ms",
        "cloud_decode_ms",
        "cloud_predict_wall_ms",
        "cloud_request_response_ms",
        "cloud_transport_overhead_est_ms",
        "e2e_display_ms",
    ]

    latency_summary: Dict[str, Dict[str, float]] = {}

    for field in metric_fields:
        if field == "e2e_display_ms":
            # Chỉ thống kê các frame đã nhận được Browser ACK (> 0) để tránh giá trị 0.0 làm sai lệch trung bình
            vals = [getattr(r, field) for r in success_records if getattr(r, field, 0.0) > 0.0]
        else:
            vals = [getattr(r, field) for r in success_records]
        if vals:
            latency_summary[field] = {
                "avg": round(sum(vals) / len(vals), 2),
                "p50": round(calc_percentile(vals, 50.0), 2),
                "p95": round(calc_percentile(vals, 95.0), 2),
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
            }
        else:
            latency_summary[field] = {"avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}

    return {
        "total_frames": total_count,
        "success_frames": len(success_records),
        "failed_frames": failed_count,
        "success_rate_pct": round(success_rate, 1),
        "current_fps": round(current_fps, 1),
        "average_fps": round(avg_fps, 1),
        "throughput_fps": round(throughput_fps, 2),
        "latency_ms": latency_summary,
    }


class PerformanceTracker:
    """
    Bộ giám sát hiệu năng theo thời gian thực (Rolling window) phục vụ /api/metrics.
    Thread-safe, lưu tối đa `window_size` mẫu gần nhất.
    """

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.lock = threading.Lock()
        self.rolling_records: deque = deque(maxlen=window_size)
        self.total_processed_frames = 0
        self.total_failed_frames = 0
        self.start_timestamp = time.time()
        self.latest_record: Optional[LatencyRecord] = None

    def record_frame(self, record: LatencyRecord) -> None:
        """Ghi nhận một mẫu khung hình mới vào bộ đệm."""
        with self.lock:
            self.total_processed_frames += 1
            if not record.success:
                self.total_failed_frames += 1
            self.rolling_records.append(record)
            self.latest_record = record

    def get_metrics_api_data(
        self,
        camera_status: str,
        reliability_summary: Optional[Dict[str, Any]] = None,
        inference_mode: str = "local",
    ) -> Dict[str, Any]:
        """Tạo dữ liệu trả về cho endpoint GET /api/metrics."""
        with self.lock:
            records_snapshot = list(self.rolling_records)
            total_proc = self.total_processed_frames
            total_fail = self.total_failed_frames
            uptime_sec = time.time() - self.start_timestamp

        summary = summarize_metrics(records_snapshot)

        # Cung cấp cấu trúc phản hồi trực quan, phân định rõ ràng các loại latency
        res = {
            "status": "ok",
            "camera_status": camera_status,
            "inference_mode": inference_mode,
            "uptime_seconds": round(uptime_sec, 1),
            "total_processed_frames": total_proc,
            "total_failed_frames": total_fail,
            "window_sample_size": len(records_snapshot),
            "fps": {
                "current": summary["current_fps"],
                "average": summary["average_fps"],
            },
            "latency_breakdown_ms": {
                "http_capture": summary["latency_ms"].get("capture_http_ms", {}),
                "jpeg_decode": summary["latency_ms"].get("decode_ms", {}),
                "yolo_inference": summary["latency_ms"].get("inference_ms", {}),
                "box_rendering": summary["latency_ms"].get("render_ms", {}),
                "jpeg_encode": summary["latency_ms"].get("encode_ms", {}),
            },
            "latency_totals_ms": {
                "server_processing": summary["latency_ms"].get("server_processing_ms", {}),
                "total_loop": summary["latency_ms"].get("total_loop_ms", {}),
                "e2e_display": summary["latency_ms"].get("e2e_display_ms", {}),
            },
            "cloud_latency_ms": {
                "cloud_inference": summary["latency_ms"].get("cloud_inference_ms", {}),
                "cloud_server": summary["latency_ms"].get("cloud_server_ms", {}),
                "cloud_request_response": summary["latency_ms"].get("cloud_request_response_ms", {}),
                "cloud_transport_overhead_est": summary["latency_ms"].get("cloud_transport_overhead_est_ms", {}),
            },
            "display_latency_status": {
                "measured": summary["latency_ms"].get("e2e_display_ms", {}).get("avg", 0.0) > 0.0,
                "type": "render_ack_latency",
                "method": "Browser ACK (requestAnimationFrame + HTTP ACK)",
                "stats_ms": summary["latency_ms"].get("e2e_display_ms", {}),
                "note": "Ước lượng đo từ khi Gateway bắt đầu lấy frame ESP32-CAM (/capture) đến khi server nhận ACK xác nhận hiển thị từ trình duyệt sau khi giải mã và đưa vào chu kỳ hiển thị requestAnimationFrame (bao gồm overhead gửi HTTP ACK, không phải thời điểm xuất hiện vật lý photon trên màn hình).",
            },
            "definitions": {
                "http_capture": "Thời gian mạng HTTP request/response tải ảnh JPEG từ ESP32-CAM (không phải shutter/sensor time).",
                "yolo_inference": "Thời gian mô hình YOLO11n suy luận trên GPU RTX 5060 (khi ở Local Mode).",
                "cloud_inference": "Thời gian mô hình YOLO11n suy luận trên CPU của EC2 (lấy từ timings_ms của Cloud).",
                "cloud_server": "Tổng thời gian server EC2 xử lý request (decode + infer + postprocess).",
                "cloud_request_response": "Thời gian vòng lặp từ lúc Laptop bắt đầu gửi POST lên EC2 đến khi nhận xong response.",
                "cloud_transport_overhead_est": "Ước lượng overhead vận chuyển = max(0, cloud_request_response - cloud_server). Bao gồm network transmission, serialization, TLS/SSH, không phải độ trễ mạng thuần túy.",
                "end_to_end_display": "Ước lượng Render-ACK Latency từ khi Gateway bắt đầu GET /capture đến khi trình duyệt gửi ACK sau khi decode và render qua requestAnimationFrame.",
                "server_processing": "Tổng thời gian máy tính xử lý khung hình (Decode + Inference/Cloud RTT + Rendering + Encode).",
                "total_loop": "Thời gian hoàn thành một chu kỳ pipeline (HTTP Capture + Server Processing).",
            },
        }

        if reliability_summary:
            res["is_stale"] = reliability_summary.get("is_stale", False)
            res["data_age_seconds"] = reliability_summary.get("data_age_seconds", 0.0)
            res["stale_reason"] = reliability_summary.get("stale_reason", "")
            res["reliability"] = reliability_summary

        return res



class BenchmarkSession:
    """
    Quản lý phiên chạy benchmark thực nghiệm trong thời lượng xác định (ví dụ 60s).
    """

    def __init__(
        self,
        scenario_id: str,
        duration_seconds: float = 60.0,
        output_dir: Optional[Path] = None,
    ):
        clean_id = scenario_id.strip().upper()
        if clean_id not in BENCHMARK_SCENARIOS:
            raise ValueError(f"Kịch bản không hợp lệ: '{scenario_id}'. Chọn một trong: {list(BENCHMARK_SCENARIOS.keys())}")

        self.scenario_id = clean_id
        self.scenario_desc = BENCHMARK_SCENARIOS[clean_id]
        self.duration_seconds = float(duration_seconds)
        self.output_dir = output_dir or (Path(__file__).resolve().parents[1] / "results" / "performance")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.is_active = False
        self.start_time = 0.0
        self.end_time = 0.0
        self.records: List[LatencyRecord] = []
        self.lock = threading.Lock()
        self.completed = False
        self.csv_path: Optional[Path] = None
        self.report_path: Optional[Path] = None

    def start(self) -> None:
        """Bắt đầu phiên benchmark."""
        with self.lock:
            self.records.clear()
            self.is_active = True
            self.completed = False
            self.start_time = time.time()
            self.end_time = self.start_time + self.duration_seconds

    def add_record(self, record: LatencyRecord) -> bool:
        """
        Thêm bản ghi nếu phiên đang chạy.
        Tự động dừng khi vượt quá thời lượng benchmark.
        Trả về True nếu phiên vừa hoàn tất sau khung hình này.
        """
        with self.lock:
            if not self.is_active:
                return False

            self.records.append(record)

            # Kiểm tra thời gian kết thúc
            if time.time() >= self.end_time:
                self.is_active = False
                self.completed = True
                self._save_results()
                return True

        return False

    def stop(self) -> None:
        """Dừng benchmark trước hạn và lưu kết quả."""
        with self.lock:
            if self.is_active:
                self.is_active = False
                self.completed = True
                self._save_results()

    def get_progress(self) -> Dict[str, Any]:
        """Lấy tiến độ phiên benchmark."""
        with self.lock:
            now = time.time()
            elapsed = (now - self.start_time) if self.start_time > 0 else 0.0
            if self.is_active:
                progress_pct = min(100.0, (elapsed / self.duration_seconds) * 100.0)
                remaining = max(0.0, self.duration_seconds - elapsed)
            else:
                progress_pct = 100.0 if self.completed else 0.0
                remaining = 0.0

            records_count = len(self.records)

        return {
            "is_active": self.is_active,
            "completed": self.completed,
            "scenario_id": self.scenario_id,
            "scenario_desc": self.scenario_desc,
            "duration_seconds": self.duration_seconds,
            "elapsed_seconds": round(elapsed, 1),
            "remaining_seconds": round(remaining, 1),
            "progress_pct": round(progress_pct, 1),
            "recorded_frames": records_count,
            "csv_path": str(self.csv_path) if self.csv_path else None,
            "report_path": str(self.report_path) if self.report_path else None,
        }

    def _save_results(self) -> None:
        """Ghi kết quả ra CSV và tạo báo cáo markdown."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"benchmark_{self.scenario_id}_{ts_str}"

        self.csv_path = self.output_dir / f"{prefix}.csv"
        self.report_path = self.output_dir / f"{prefix}_report.md"

        # 1. Xuất file CSV
        fieldnames = [
            "frame_index",
            "timestamp",
            "success",
            "inference_mode",
            "capture_http_ms",
            "decode_ms",
            "inference_ms",
            "render_ms",
            "encode_ms",
            "server_processing_ms",
            "total_loop_ms",
            "fps",
            "mask_count",
            "no_mask_count",
            "unknown_count",
            "total_faces",
            "cloud_inference_ms",
            "cloud_server_ms",
            "cloud_decode_ms",
            "cloud_predict_wall_ms",
            "cloud_request_response_ms",
            "cloud_transport_overhead_est_ms",
            "e2e_display_ms",
            "error_msg",
        ]

        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))

        # 2. Tạo báo cáo Markdown tổng hợp
        actual_elapsed = (time.time() - self.start_time) if self.start_time > 0 else self.duration_seconds
        summary = summarize_metrics(self.records, elapsed_seconds=actual_elapsed)
        lat = summary["latency_ms"]

        # Kiểm tra xem phiên benchmark có chạy Cloud mode hay không
        has_cloud_records = any(getattr(r, "inference_mode", "local") == "cloud" for r in self.records)

        cloud_table_md = ""
        if has_cloud_records:
            cloud_table_md = f"""
### 2.1. Phân Tích Độ Trễ AWS Cloud Inference (CPU EC2)

*Lưu ý: Phân định rõ ràng giữa CPU Cloud và GPU Local. Không so sánh trực tiếp hiệu năng phần cứng thuần túy.*
- **Cloud Inference Latency:** Thời gian thực thi mạng nơ-ron trên CPU của EC2 (lấy từ timings_ms của Cloud Server).
- **Local-Cloud Request-Response:** Thời gian vòng lặp HTTP POST từ Laptop đến EC2 và nhận đầy đủ response.
- **Transport Overhead (Est):** max(0, Request-Response - Server Processing). Ước lượng tổng overhead vận chuyển (bao gồm network, serialization, HTTP handshake, TLS/SSH tunnel), không phải độ trễ mạng thuần túy.

| Chỉ số Cloud | Average (ms) | P50 / Median (ms) | P95 (ms) | Min (ms) | Max (ms) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Cloud YOLO Inference (CPU)** | **{lat.get('cloud_inference_ms', {}).get('avg', 0)}** | **{lat.get('cloud_inference_ms', {}).get('p50', 0)}** | **{lat.get('cloud_inference_ms', {}).get('p95', 0)}** | {lat.get('cloud_inference_ms', {}).get('min', 0)} | {lat.get('cloud_inference_ms', {}).get('max', 0)} |
| **Cloud Server Processing** | {lat.get('cloud_server_ms', {}).get('avg', 0)} | {lat.get('cloud_server_ms', {}).get('p50', 0)} | {lat.get('cloud_server_ms', {}).get('p95', 0)} | {lat.get('cloud_server_ms', {}).get('min', 0)} | {lat.get('cloud_server_ms', {}).get('max', 0)} |
| **Local-Cloud Request-Response (RTT)** | **{lat.get('cloud_request_response_ms', {}).get('avg', 0)}** | **{lat.get('cloud_request_response_ms', {}).get('p50', 0)}** | **{lat.get('cloud_request_response_ms', {}).get('p95', 0)}** | {lat.get('cloud_request_response_ms', {}).get('min', 0)} | {lat.get('cloud_request_response_ms', {}).get('max', 0)} |
| **Transport Overhead Est.** | {lat.get('cloud_transport_overhead_est_ms', {}).get('avg', 0)} | {lat.get('cloud_transport_overhead_est_ms', {}).get('p50', 0)} | {lat.get('cloud_transport_overhead_est_ms', {}).get('p95', 0)} | {lat.get('cloud_transport_overhead_est_ms', {}).get('min', 0)} | {lat.get('cloud_transport_overhead_est_ms', {}).get('max', 0)} |
"""

        e2e_lat = lat.get("e2e_display_ms", {})
        e2e_sample_count = sum(
            1 for r in self.records if r.success and getattr(r, "e2e_display_ms", 0.0) > 0.0
        )
        if e2e_sample_count > 0:
            e2e_table_md = f"""
| Chỉ số Độ trễ Toàn trình | Số mẫu | Average (ms) | P50 / Median (ms) | P95 (ms) | Min (ms) | Max (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **End-to-End Display Latency (Browser ACK)** | {e2e_sample_count} | **{e2e_lat.get('avg', 0)}** | **{e2e_lat.get('p50', 0)}** | **{e2e_lat.get('p95', 0)}** | {e2e_lat.get('min', 0)} | {e2e_lat.get('max', 0)} |
| **Gateway Total Pipeline Loop** | {summary['success_frames']} | {lat.get('total_loop_ms', {}).get('avg', 0)} | {lat.get('total_loop_ms', {}).get('p50', 0)} | {lat.get('total_loop_ms', {}).get('p95', 0)} | {lat.get('total_loop_ms', {}).get('min', 0)} | {lat.get('total_loop_ms', {}).get('max', 0)} |
"""
        else:
            e2e_table_md = "\n*Chưa ghi nhận mẫu Browser ACK trong phiên benchmark này (cần mở trang Web Dashboard trên trình duyệt trong lúc chạy benchmark để đo `e2e_display_ms`).*\n"

        report_md = f"""# Báo Cáo Thực Nghiệm Hiệu Năng (Benchmark Performance Report)

- **Kịch bản kiểm thử:** `{self.scenario_id}` — {self.scenario_desc}
- **Thời gian thực hiện:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Thời lượng cấu hình:** {self.duration_seconds:.1f} giây (Thực tế: {actual_elapsed:.1f} giây)
- **File CSV chi tiết:** `{self.csv_path.name}`

---

## 1. Tóm tắt kết quả vận hành

| Chỉ số | Giá trị |
|---|---|
| **Tổng số frame ghi nhận** | {summary['total_frames']} |
| **Số frame thành công** | {summary['success_frames']} |
| **Số frame lỗi / timeout** | {summary['failed_frames']} |
| **Tỷ lệ thành công** | {summary['success_rate_pct']}% |
| **Throughput FPS (Completed / Elapsed)** | **{summary['throughput_fps']} FPS** |
| **Tốc độ khung hình chu kỳ (Rolling Loop FPS)** | {summary['average_fps']} FPS |
| **Tốc độ khung hình tức thời (Current FPS)** | {summary['current_fps']} FPS |

*Ghi chú về phân biệt FPS:*
- **Throughput FPS:** Đo lường thông lượng thực tế dựa trên số khung hình hoàn thành chia cho tổng thời gian trôi qua thực tế ($\\text{{Completed Frames}} / \\text{{Elapsed Seconds}}$).
- **Rolling Loop FPS:** Tính toán dựa trên độ trễ trung bình của một chu kỳ lặp ($1000 / \\bar{{t}}_{{\\text{{loop}}}}$).

---

## 2. Chi tiết độ trễ từng công đoạn (Latency Breakdown)

*Ghi chú:*
- **Capture (HTTP):** Thời gian mạng request/response lấy ảnh qua HTTP GET `/capture` từ ESP32-CAM.
- **Inference (YOLO11n):** Thời gian thực thi mạng nơ-ron trên GPU RTX 5060 (đồng bộ CUDA) khi ở Local Mode.
- **Server Processing:** Tổng thời gian máy tính xử lý (Decode + Inference/Cloud RTT + Box Render + Encode).
- **Total Loop:** Tổng thời gian một chu kỳ pipeline từ lúc gửi request capture đến khi hoàn thành frame.

| Công đoạn | Average (ms) | P50 / Median (ms) | P95 (ms) | Min (ms) | Max (ms) |
|---|:---:|:---:|:---:|:---:|:---:|
| **HTTP Capture (Mạng)** | {lat.get('capture_http_ms', {}).get('avg', 0)} | {lat.get('capture_http_ms', {}).get('p50', 0)} | {lat.get('capture_http_ms', {}).get('p95', 0)} | {lat.get('capture_http_ms', {}).get('min', 0)} | {lat.get('capture_http_ms', {}).get('max', 0)} |
| **JPEG Decode** | {lat.get('decode_ms', {}).get('avg', 0)} | {lat.get('decode_ms', {}).get('p50', 0)} | {lat.get('decode_ms', {}).get('p95', 0)} | {lat.get('decode_ms', {}).get('min', 0)} | {lat.get('decode_ms', {}).get('max', 0)} |
| **YOLO11n Inference (Local RTX 5060)** | **{lat.get('inference_ms', {}).get('avg', 0)}** | **{lat.get('inference_ms', {}).get('p50', 0)}** | **{lat.get('inference_ms', {}).get('p95', 0)}** | {lat.get('inference_ms', {}).get('min', 0)} | {lat.get('inference_ms', {}).get('max', 0)} |
| **Render Box & Labels** | {lat.get('render_ms', {}).get('avg', 0)} | {lat.get('render_ms', {}).get('p50', 0)} | {lat.get('render_ms', {}).get('p95', 0)} | {lat.get('render_ms', {}).get('min', 0)} | {lat.get('render_ms', {}).get('max', 0)} |
| **JPEG Encode (MJPEG)** | {lat.get('encode_ms', {}).get('avg', 0)} | {lat.get('encode_ms', {}).get('p50', 0)} | {lat.get('encode_ms', {}).get('p95', 0)} | {lat.get('encode_ms', {}).get('min', 0)} | {lat.get('encode_ms', {}).get('max', 0)} |
| **Server Processing (Tổng xử lý)** | **{lat.get('server_processing_ms', {}).get('avg', 0)}** | **{lat.get('server_processing_ms', {}).get('p50', 0)}** | **{lat.get('server_processing_ms', {}).get('p95', 0)}** | {lat.get('server_processing_ms', {}).get('min', 0)} | {lat.get('server_processing_ms', {}).get('max', 0)} |
| **Total Pipeline Loop** | **{lat.get('total_loop_ms', {}).get('avg', 0)}** | **{lat.get('total_loop_ms', {}).get('p50', 0)}** | **{lat.get('total_loop_ms', {}).get('p95', 0)}** | {lat.get('total_loop_ms', {}).get('min', 0)} | {lat.get('total_loop_ms', {}).get('max', 0)} |
{cloud_table_md}
---

### 3. Đánh Giá End-to-End Display Latency
- Phương pháp: **Browser Acknowledgment (requestAnimationFrame + HTTP ACK)**.
- Định nghĩa: Đo từ lúc Gateway bắt đầu GET `/capture` từ ESP32-CAM đến khi trình duyệt giải mã frame, đưa vào chu kỳ hiển thị qua `requestAnimationFrame`, và gửi HTTP ACK về server.
- Ghi chú kỹ thuật: Đây là ước lượng render-ack latency (bao gồm overhead gửi ACK trên mạng), không tuyên bố là thời điểm xuất hiện photon vật lý trên tấm nền màn hình. Không sử dụng Gateway Pipeline Latency để thay thế cho Display Latency.
{e2e_table_md}
---
*Báo cáo được sinh tự động bởi hệ thống Multi-Person Face Mask Detection Performance Tracker.*
"""

        with open(self.report_path, mode="w", encoding="utf-8") as f:
            f.write(report_md)
