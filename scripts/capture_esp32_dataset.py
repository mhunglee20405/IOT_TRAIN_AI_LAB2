# -*- coding: utf-8 -*-
"""
scripts/capture_esp32_dataset.py  —  Thu thập ảnh từ ESP32-CAM
==============================================================
Gọi HTTP GET /capture trên ESP32-CAM và lưu ảnh JPEG vào:
    dataset_raw/own_photos/<session>/

Lưu ý:
  - Script KHÔNG tự đưa ảnh vào train/val/test.
  - Script KHÔNG sửa firmware ESP32-CAM.
  - Script KHÔNG can thiệp weights hay model YOLO.

Cách sử dụng trên Windows (PowerShell / Command Prompt):
    # Kích hoạt venv
    .\\venv\\Scripts\\activate

    # Chụp 20 ảnh mặc định từ ESP32-CAM (tự động tạo session theo ngày giờ):
    python scripts/capture_esp32_dataset.py --ip 192.168.1.50

    # Tùy chỉnh session, số lượng ảnh và khoảng cách thời gian giữa các lần chụp:
    python scripts/capture_esp32_dataset.py --ip 192.168.1.50 --count 50 --interval 1.5 --session session_office_mask

    # Xem trợ giúp:
    python scripts/capture_esp32_dataset.py --help
"""

import argparse
import io
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

# Đảm bảo in tiếng Việt / ký tự đặc biệt ổn định trên console Windows
if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_BASE_DIR = PROJECT_ROOT / "dataset_raw" / "own_photos"

# Magic bytes nhận diện định dạng JPEG
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"

DEFAULT_COUNT = 20
DEFAULT_INTERVAL = 2.0
DEFAULT_TIMEOUT = 8

# Mã màu ANSI
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET}    {msg}")


def err(msg: str) -> None:
    print(f"  {RED}[ERR]{RESET}   {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"  {CYAN}[INFO]{RESET}  {msg}")


def is_valid_jpeg(data: bytes) -> tuple[bool, str]:
    """
    Kiểm tra tính hợp lệ cơ bản của file ảnh JPEG:
    - Kích thước tối thiểu > 100 bytes
    - Bắt đầu với SOI marker (0xFFD8)
    - Kết thúc với EOI marker (0xFFD9)
    """
    if len(data) < 100:
        return False, f"Kích thước dữ liệu quá nhỏ ({len(data)} bytes)"
    if not data.startswith(JPEG_SOI):
        return False, f"Thiếu SOI marker (0xFFD8). Bắt đầu bởi: {data[:4].hex()}"
    if not data.endswith(JPEG_EOI):
        return False, f"Thiếu EOI marker (0xFFD9). Kết thúc bởi: {data[-4:].hex()}"
    return True, "Hợp lệ"


def fetch_capture(ip: str, timeout: int) -> tuple[bytes | None, str]:
    """
    Gửi request HTTP GET /capture tới ESP32-CAM và nhận nội dung byte.
    Hỗ trợ format ip dạng: 192.168.1.50 hoặc 192.168.1.50:80 hoặc http://...
    """
    clean_ip = ip.strip()
    if clean_ip.startswith("http://") or clean_ip.startswith("https://"):
        base_url = clean_ip.rstrip("/")
    else:
        base_url = f"http://{clean_ip.rstrip('/')}"

    url = f"{base_url}/capture"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ESP32-Capture-Client/1.0",
            "Accept": "image/jpeg, image/*",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            if status != 200:
                return None, f"HTTP Status {status}"

            content_type = response.headers.get("Content-Type", "")
            if content_type and ("image" not in content_type.lower() and "jpeg" not in content_type.lower()):
                warn(f"Content-Type từ server là '{content_type}', tiếp tục thẩm định JPEG marker.")

            data = response.read()
            return data, ""

    except urllib.error.HTTPError as e:
        return None, f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        reason = str(e.reason) if hasattr(e, "reason") else str(e)
        return None, f"Lỗi kết nối ({reason})"
    except TimeoutError:
        return None, f"Quá thời gian chờ phản hồi ({timeout}s)"
    except Exception as e:
        return None, f"Ngoại lệ: {e}"


def get_unique_filepath(target_dir: Path, prefix: str = "img") -> Path:
    """
    Sinh đường dẫn file không bị ghi đè.
    Định dạng: <prefix>_YYYYMMDD_HHMMSS_<seq>.jpg
    """
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    counter = 1
    while True:
        filename = f"{prefix}_{now_str}_{counter:04d}.jpg"
        filepath = target_dir / filename
        if not filepath.exists():
            return filepath
        counter += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thu thập ảnh dataset từ ESP32-CAM qua HTTP GET /capture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ip",
        type=str,
        required=True,
        help="Địa chỉ IP (hoặc host:port) của ESP32-CAM (vd: 192.168.1.50 hoặc 192.168.1.50:81)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Số lượng ảnh cần chụp",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help="Khoảng cách thời gian (giây) giữa các lần chụp liên tiếp",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Tên session lưu ảnh (nếu không cung cấp, tự sinh session_<YYYYMMDD_HHMMSS>)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Thời gian timeout (giây) cho mỗi HTTP request",
    )
    return parser.parse_args()


def main() -> None:
    if sys.platform == "win32":
        os.system("color")

    args = parse_args()

    if args.count <= 0:
        err("--count phải là số nguyên dương lớn hơn 0.")
        sys.exit(1)

    if args.interval < 0:
        err("--interval phải là số không âm.")
        sys.exit(1)

    if args.timeout <= 0:
        err("--timeout phải lớn hơn 0.")
        sys.exit(1)

    session_name = args.session.strip() if args.session else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Đảm bảo tên session an toàn cho thư mục
    safe_session = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in session_name)
    save_dir = OUTPUT_BASE_DIR / safe_session

    print(f"\n{'='*64}")
    print(f"  {BOLD}ESP32-CAM Dataset Capture{RESET}")
    print(f"  Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*64}")
    info(f"Target ESP32-CAM IP : {args.ip}")
    info(f"Target URL          : http://{args.ip.strip().rstrip('/')}/capture")
    info(f"Session             : {safe_session}")
    info(f"Thư mục lưu ảnh     : {save_dir}")
    info(f"Kế hoạch chụp       : {args.count} ảnh | chu kỳ: {args.interval}s | timeout: {args.timeout}s")
    print(f"{'-'*64}\n")

    saved_count = 0
    failed_count = 0
    start_time = time.time()

    for idx in range(1, args.count + 1):
        step_start = time.time()
        print(f"[{idx:03d}/{args.count:03d}] Đang yêu cầu /capture... ", end="", flush=True)

        data, fetch_err = fetch_capture(args.ip, args.timeout)

        if data is None:
            failed_count += 1
            print(f"{RED}[THẤT BẠI]{RESET} {fetch_err}")
        else:
            is_valid, val_msg = is_valid_jpeg(data)
            if not is_valid:
                failed_count += 1
                print(f"{RED}[JPEG KHÔNG HỢP LỆ]{RESET} {val_msg} (nhận {len(data)} bytes)")
            else:
                save_dir.mkdir(parents=True, exist_ok=True)
                out_path = get_unique_filepath(save_dir, prefix=safe_session)
                # Kiểm tra chắc chắn không ghi đè
                if out_path.exists():
                    failed_count += 1
                    print(f"{YELLOW}[BỎ QUA - ĐÃ TỒN TẠI]{RESET} {out_path.name}")
                else:
                    try:
                        with open(out_path, "wb") as f:
                            f.write(data)
                        saved_count += 1
                        size_kb = len(data) / 1024.0
                        print(f"{GREEN}[LƯU THÀNH CÔNG]{RESET} {out_path.name} ({size_kb:.1f} KB)")
                    except Exception as write_err:
                        failed_count += 1
                        print(f"{RED}[LỖI GHI FILE]{RESET} {write_err}")

        # Giãn cách theo interval nếu chưa phải ảnh cuối
        if idx < args.count and args.interval > 0:
            elapsed = time.time() - step_start
            sleep_time = max(0.0, args.interval - elapsed)
            time.sleep(sleep_time)

    total_time = time.time() - start_time

    # ── Thống kê tổng kết ──
    print(f"\n{'='*64}")
    print(f"  {BOLD}KẾT QUẢ THỐNG KÊ THU THẬP{RESET}")
    print(f"{'='*64}")
    info(f"Tổng số ảnh yêu cầu  : {args.count}")
    ok(f"Số ảnh lưu thành công : {saved_count}")
    if failed_count > 0:
        err(f"Số lần gặp sự cố/lỗi  : {failed_count}")
    else:
        ok("Không có lỗi xảy ra.")
    info(f"Tổng thời gian chạy   : {total_time:.2f} giây")
    info(f"Đường dẫn chứa ảnh    : {save_dir}")
    print(f"{'='*64}\n")

    if saved_count == 0:
        err("Không thu thập được ảnh nào. Vui lòng kiểm tra lại địa chỉ IP, kết nối mạng và endpoint /capture của ESP32-CAM.")
        sys.exit(1)


if __name__ == "__main__":
    main()
