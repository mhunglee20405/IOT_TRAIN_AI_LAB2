# -*- coding: utf-8 -*-
"""
src/reliability.py  —  Hệ Thống Giám Sát Độ Tin Cậy & Phục Hồi (Reliability & Recovery)
====================================================================================
Quản lý cơ chế xử lý lỗi mạng, retry với Exponential Backoff, tính toán thời gian phục hồi
(Recovery Time), phát hiện dữ liệu cũ (Stale Data), và thực hiện các kịch bản kiểm thử R1–R6.
"""

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
import io
import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from src.metrics import calc_percentile

# Danh mục kịch bản kiểm thử độ tin cậy chuẩn hóa
RELIABILITY_SCENARIOS = {
    "R1": "Mất kết nối thoáng qua (Transient Network Glitch: 1-2 request lỗi/timeout rồi hồi phục ngay)",
    "R2": "Mất kết nối Wi-Fi tạm thời (Temporary Outage: mất mạng 10-15s, kiểm tra backoff và phục hồi)",
    "R3": "Mất nguồn/mất kết nối kéo dài (Extended Outage: > 30s, kiểm tra backoff trần và cảnh báo stale)",
    "R4": "Mạng chập chờn / HTTP Timeout kéo dài (Phản hồi chậm vượt ngưỡng socket timeout 3.0s)",
    "R5": "Tải đồng thời nhiều Web Client (5-10 tabs mở stream /video và poll API đồng thời)",
    "R6": "Graceful Shutdown khi có tải (Tắt server an toàn trong < 3 giây khi đang stream và polling)",
}


@dataclass
class RecoveryEvent:
    """Bản ghi một sự kiện phục hồi kết nối thành công."""
    event_index: int
    disconnect_timestamp: float
    recovery_timestamp: float
    recovery_duration_seconds: float
    consecutive_failures_before_recovery: int
    failure_reasons: List[str]


@dataclass
class ReliabilityRecord:
    """Bản ghi trạng thái định kỳ phục vụ xuất CSV / báo cáo."""
    timestamp: float
    elapsed_seconds: float
    camera_status: str
    is_stale: bool
    data_age_seconds: float
    consecutive_failures: int
    total_failures: int
    total_timeouts: int
    total_recoveries: int
    last_recovery_time_seconds: float
    current_backoff_seconds: float
    active_connections: int = 1
    event_note: str = ""


class ReliabilityTracker:
    """
    Theo dõi và quản lý độ tin cậy kết nối camera ESP32-CAM:
      - Exponential Backoff có trần (max_backoff).
      - Đo đạc chính xác Recovery Time khi kết nối lại thành công.
      - Thống kê Total Failures, Timeouts, Recoveries, Success Rate.
      - Đánh giá tính tươi mới của dữ liệu (Stale Data Detection).
    """

    def __init__(
        self,
        initial_backoff: float = 0.5,
        max_backoff: float = 5.0,
        backoff_factor: float = 1.5,
        stale_threshold_seconds: float = 4.0,
        max_history: int = 100,
    ):
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_factor = backoff_factor
        self.stale_threshold_seconds = stale_threshold_seconds
        self.max_history = max_history

        self.lock = threading.Lock()

        # Trạng thái hiện tại
        self.camera_status = "connecting"  # "connected", "reconnecting", "error"
        self.current_backoff = 0.0
        self.consecutive_failures = 0
        self.consecutive_successes = 0

        # Mốc thời gian
        self.last_successful_frame_time: float = 0.0
        self.disconnect_start_time: Optional[float] = None
        self.current_failure_reasons: List[str] = []

        # Số liệu tích lũy
        self.total_connection_attempts: int = 0
        self.total_connection_failures: int = 0
        self.total_timeouts: int = 0
        self.total_recoveries: int = 0

        # Lịch sử các đợt phục hồi
        self.recovery_events: List[RecoveryEvent] = []
        self.recovery_durations: List[float] = []

        # Lịch sử mẫu gần nhất
        self.history_records: deque = deque(maxlen=max_history)
        self.start_timestamp = time.time()

    def record_attempt(self) -> None:
        """Ghi nhận một lượt gửi HTTP request tới camera."""
        with self.lock:
            self.total_connection_attempts += 1

    def record_failure(self, error_msg: str, is_timeout: bool = False) -> float:
        """
        Ghi nhận khi một frame bị lỗi hoặc rớt mạng.
        Tính toán và trả về thời gian ngủ (backoff delay) tiếp theo.
        """
        now = time.time()
        with self.lock:
            self.total_connection_failures += 1
            if is_timeout:
                self.total_timeouts += 1

            self.consecutive_failures += 1
            self.consecutive_successes = 0

            # Nếu đây là lỗi đầu tiên sau một đợt hoạt động bình thường
            if self.disconnect_start_time is None:
                # Nếu đã từng nhận frame, tính từ frame cuối; nếu chưa, tính từ lúc này
                self.disconnect_start_time = self.last_successful_frame_time if self.last_successful_frame_time > 0 else now

            self.current_failure_reasons.append(error_msg[:120])
            if len(self.current_failure_reasons) > 10:
                self.current_failure_reasons.pop(0)

            # Tính Exponential Backoff có trần
            exponent = max(0, self.consecutive_failures - 1)
            calculated_backoff = self.initial_backoff * (self.backoff_factor ** exponent)
            self.current_backoff = min(self.max_backoff, round(calculated_backoff, 2))
            self.camera_status = "reconnecting"

            backoff_to_sleep = self.current_backoff

        return backoff_to_sleep

    def record_success(self) -> Optional[float]:
        """
        Ghi nhận khi một frame tải về và giải mã thành công.
        Nếu hệ thống vừa phục hồi sau khi mất kết nối, tính toán và ghi nhận recovery time.
        Trả về: recovery_duration_seconds nếu có sự kiện phục hồi, ngược lại None.
        """
        now = time.time()
        recovery_duration: Optional[float] = None

        with self.lock:
            self.consecutive_successes += 1
            self.last_successful_frame_time = now

            # Phát hiện phục hồi (vừa có sự cố trước đó)
            if self.disconnect_start_time is not None:
                recovery_duration = round(max(0.001, now - self.disconnect_start_time), 3)
                self.total_recoveries += 1
                self.recovery_durations.append(recovery_duration)

                event = RecoveryEvent(
                    event_index=self.total_recoveries,
                    disconnect_timestamp=self.disconnect_start_time,
                    recovery_timestamp=now,
                    recovery_duration_seconds=recovery_duration,
                    consecutive_failures_before_recovery=self.consecutive_failures,
                    failure_reasons=list(self.current_failure_reasons),
                )
                self.recovery_events.append(event)

                # Reset cờ mất kết nối
                self.disconnect_start_time = None
                self.current_failure_reasons.clear()

            self.consecutive_failures = 0
            self.current_backoff = 0.0
            self.camera_status = "connected"

        return recovery_duration

    def evaluate_staleness(self) -> Tuple[bool, float, str]:
        """
        Đánh giá xem dữ liệu hiện tại có bị cũ (stale) hay không.
        Trả về: (is_stale, data_age_seconds, reason)
        """
        now = time.time()
        with self.lock:
            if self.last_successful_frame_time <= 0:
                return True, 0.0, "Chưa từng nhận được khung hình hợp lệ từ ESP32-CAM"

            age = round(now - self.last_successful_frame_time, 2)

            if self.camera_status != "connected":
                return True, age, f"Camera đang ở trạng thái '{self.camera_status}' (mất kết nối {age:.1f}s)"

            if age > self.stale_threshold_seconds:
                return True, age, f"Dữ liệu cũ (cập nhật cách đây {age:.1f}s, vượt ngưỡng {self.stale_threshold_seconds}s)"

            return False, age, "Dữ liệu hợp lệ và thời gian thực"

    def get_summary(self) -> Dict[str, Any]:
        """Tổng hợp các chỉ số độ tin cậy và phục hồi."""
        with self.lock:
            durations = list(self.recovery_durations)
            status = self.camera_status
            total_attempts = self.total_connection_attempts
            total_failures = self.total_connection_failures
            total_timeouts = self.total_timeouts
            total_recoveries = self.total_recoveries
            consec_fails = self.consecutive_failures
            current_backoff = self.current_backoff
            last_frame_ts = self.last_successful_frame_time
            threshold = self.stale_threshold_seconds

        is_stale, age, stale_reason = self.evaluate_staleness()

        # Tính toán thống kê recovery time
        avg_recovery = round(sum(durations) / len(durations), 3) if durations else 0.0
        p50_recovery = round(calc_percentile(durations, 50.0), 3) if durations else 0.0
        p95_recovery = round(calc_percentile(durations, 95.0), 3) if durations else 0.0
        min_recovery = round(min(durations), 3) if durations else 0.0
        max_recovery = round(max(durations), 3) if durations else 0.0

        # Tỷ lệ phục hồi (Số lần hồi phục / Số đợt mất kết nối đã ghi nhận)
        # Một đợt mất kết nối được tính khi có recovery hoặc đang bị mất
        disconnection_episodes = total_recoveries + (1 if self.disconnect_start_time is not None else 0)
        recovery_success_rate = round((total_recoveries / disconnection_episodes * 100.0), 1) if disconnection_episodes > 0 else 100.0

        return {
            "camera_status": status,
            "is_stale": is_stale,
            "data_age_seconds": age,
            "stale_reason": stale_reason,
            "stale_threshold_seconds": threshold,
            "last_successful_frame_timestamp": last_frame_ts,
            "consecutive_failures": consec_fails,
            "current_backoff_seconds": current_backoff,
            "metrics": {
                "total_connection_attempts": total_attempts,
                "total_connection_failures": total_failures,
                "total_timeouts": total_timeouts,
                "total_recoveries": total_recoveries,
                "recovery_success_rate_pct": recovery_success_rate,
            },
            "recovery_time_seconds": {
                "sample_count": len(durations),
                "avg": avg_recovery,
                "p50": p50_recovery,
                "p95": p95_recovery,
                "min": min_recovery,
                "max": max_recovery,
                "last": durations[-1] if durations else 0.0,
            },
        }


class ReliabilityBenchmarkSession:
    """
    Quản lý một phiên kiểm thử độ tin cậy thực nghiệm (R1–R6) và xuất kết quả ra thư mục results/reliability/.
    """

    def __init__(
        self,
        scenario_id: str,
        duration_seconds: float = 60.0,
        output_dir: Optional[Path] = None,
        tracker: Optional[ReliabilityTracker] = None,
    ):
        clean_id = scenario_id.strip().upper()
        if clean_id not in RELIABILITY_SCENARIOS:
            raise ValueError(f"Kịch bản không hợp lệ: '{scenario_id}'. Chọn một trong: {list(RELIABILITY_SCENARIOS.keys())}")

        self.scenario_id = clean_id
        self.scenario_desc = RELIABILITY_SCENARIOS[clean_id]
        self.duration_seconds = float(duration_seconds)
        self.output_dir = output_dir or (Path(__file__).resolve().parents[1] / "results" / "reliability")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = tracker or ReliabilityTracker()
        self.is_active = False
        self.completed = False
        self.start_time = 0.0
        self.end_time = 0.0
        self.records: List[ReliabilityRecord] = []
        self.lock = threading.Lock()
        self.csv_path: Optional[Path] = None
        self.report_path: Optional[Path] = None

    def start(self) -> None:
        """Khởi động phiên kiểm thử độ tin cậy."""
        with self.lock:
            self.records.clear()
            self.is_active = True
            self.completed = False
            self.start_time = time.time()
            self.end_time = self.start_time + self.duration_seconds

    def record_sample(
        self,
        camera_status: str,
        is_stale: bool,
        data_age_seconds: float,
        active_connections: int = 1,
        event_note: str = "",
    ) -> bool:
        """
        Ghi nhận một mẫu trạng thái theo chu kỳ.
        Trả về True nếu phiên vừa hoàn thành sau mẫu này.
        """
        with self.lock:
            if not self.is_active:
                return False

            now = time.time()
            elapsed = round(now - self.start_time, 2)
            summary = self.tracker.get_summary()

            rec = ReliabilityRecord(
                timestamp=now,
                elapsed_seconds=elapsed,
                camera_status=camera_status,
                is_stale=is_stale,
                data_age_seconds=data_age_seconds,
                consecutive_failures=summary.get("consecutive_failures", 0),
                total_failures=summary["metrics"]["total_connection_failures"],
                total_timeouts=summary["metrics"]["total_timeouts"],
                total_recoveries=summary["metrics"]["total_recoveries"],
                last_recovery_time_seconds=summary["recovery_time_seconds"]["last"],
                current_backoff_seconds=summary.get("current_backoff_seconds", 0.0),
                active_connections=active_connections,
                event_note=event_note,
            )
            self.records.append(rec)

            if now >= self.end_time:
                self.is_active = False
                self.completed = True
                self._save_results()
                return True

        return False

    def stop(self) -> None:
        """Dừng kiểm thử trước hạn."""
        with self.lock:
            if self.is_active:
                self.is_active = False
                self.completed = True
                self._save_results()

    def get_progress(self) -> Dict[str, Any]:
        """Lấy tiến độ phiên kiểm thử."""
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
            "recorded_samples": records_count,
            "csv_path": str(self.csv_path) if self.csv_path else None,
            "report_path": str(self.report_path) if self.report_path else None,
        }

    def _save_results(self) -> None:
        """Xuất file CSV và Markdown báo cáo độ tin cậy."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"reliability_{self.scenario_id}_{ts_str}"

        self.csv_path = self.output_dir / f"{prefix}.csv"
        self.report_path = self.output_dir / f"{prefix}_report.md"

        # 1. Xuất file CSV
        fieldnames = [
            "timestamp",
            "elapsed_seconds",
            "camera_status",
            "is_stale",
            "data_age_seconds",
            "consecutive_failures",
            "total_failures",
            "total_timeouts",
            "total_recoveries",
            "last_recovery_time_seconds",
            "current_backoff_seconds",
            "active_connections",
            "event_note",
        ]

        import csv
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))

        # 2. Tạo Markdown Report
        summary = self.tracker.get_summary()
        rec_m = summary["metrics"]
        rec_t = summary["recovery_time_seconds"]

        md_content = f"""# Báo Cáo Kiểm Thử Độ Tin Cậy & Phục Hồi — Kịch Bản {self.scenario_id}

- **Thời gian thực hiện:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Kịch bản:** `{self.scenario_id}` — {self.scenario_desc}
- **Thời lượng dự kiến:** {self.duration_seconds:.1f} giây
- **Số mẫu thu thập:** {len(self.records)} mẫu
- **File dữ liệu thô CSV:** `{self.csv_path.name}`

---

## 1. Tóm Tắt Kết Quả Độ Tin Cậy & Phục Hồi

| Chỉ số kiểm thử | Giá trị đo được | Ý nghĩa kỹ thuật |
| :--- | :--- | :--- |
| **Tổng số lần gửi request** | `{rec_m['total_connection_attempts']}` | Tổng số lần worker cố gắng lấy ảnh từ camera |
| **Số lần mất kết nối / lỗi** | `{rec_m['total_connection_failures']}` | Số lần request gặp lỗi HTTP, mạng hoặc giải mã JPEG |
| **Số lần HTTP Timeout** | `{rec_m['total_timeouts']}` | Số lần camera phản hồi vượt quá socket timeout |
| **Số lần phục hồi thành công** | `{rec_m['total_recoveries']}` | Số chu kỳ kết nối lại và suy luận tiếp sau khi mất mạng |
| **Tỷ lệ phục hồi (Success Rate)** | **`{rec_m['recovery_success_rate_pct']}%`** | Tỷ lệ khôi phục thành công các sự cố kết nối |
| **Recovery Time Trung Bình** | **`{rec_t['avg']} s`** | Thời gian trung bình từ lúc mất mạng tới khi frame đầu tiên được phục hồi |
| **Recovery Time P50 (Trung vị)** | **`{rec_t['p50']} s`** | 50% các lần phục hồi diễn ra trong khoảng thời gian này |
| **Recovery Time P95** | **`{rec_t['p95']} s`** | 95% các lần phục hồi diễn ra trong khoảng thời gian này |
| **Recovery Time Ngắn nhất / Dài nhất** | `{rec_t['min']} s` / `{rec_t['max']} s` | Biên độ dao động thời gian phục hồi |

---

## 2. Đánh Giá Trạng Thái Stale Data & Cơ Chế Backoff

- **Ngưỡng tươi mới cấu hình (Stale Threshold):** `{summary['stale_threshold_seconds']} giây`
- **Trạng thái camera kết thúc bài đo:** `{summary['camera_status'].upper()}`
- **Dữ liệu có bị cũ (Is Stale):** `{summary['is_stale']}`
- **Thời gian trễ dữ liệu cuối (Data Age):** `{summary['data_age_seconds']} giây`
- **Cơ chế Backoff:** Exponential Backoff tăng dần theo hệ số 1.5x từ 0.5s đến tối đa 5.0s, tránh bão request khi ESP32 khởi động lại.

---

## 3. Lịch Sử Các Đợt Phục Hồi Kết Nối

"""
        if self.tracker.recovery_events:
            md_content += "| Sự kiện # | Thời điểm rớt mạng | Thời điểm phục hồi | Thời gian phục hồi | Số frame lỗi trước đó |\n"
            md_content += "| :--- | :--- | :--- | :--- | :--- |\n"
            for ev in self.tracker.recovery_events:
                t_disc = datetime.fromtimestamp(ev.disconnect_timestamp).strftime('%H:%M:%S')
                t_rec = datetime.fromtimestamp(ev.recovery_timestamp).strftime('%H:%M:%S')
                md_content += f"| #{ev.event_index} | {t_disc} | {t_rec} | **{ev.recovery_duration_seconds:.2f} s** | {ev.consecutive_failures_before_recovery} |\n"
        else:
            md_content += "> *Không phát hiện sự cố mất kết nối nào trong phiên đo (kết nối liên tục 100%).*\n"

        with open(self.report_path, mode="w", encoding="utf-8") as f:
            f.write(md_content)
