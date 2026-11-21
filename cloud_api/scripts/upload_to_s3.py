# -*- coding: utf-8 -*-
"""
cloud_api/scripts/upload_to_s3.py — Script hỗ trợ tải model best.pt lên AWS S3 Private.
=====================================================================================
Upload model với mã hóa server-side AES256 (SSE-S3) và Bucket Private mặc định.

Sử dụng:
    python cloud_api/scripts/upload_to_s3.py --bucket my-mask-detection-bucket --key weights/best.pt
"""

import argparse
import io
from pathlib import Path
import subprocess
import sys

# Đảm bảo hiển thị UTF-8 trên Windows console
if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "runs" / "detect" / "results" / "train" / "antigravity_gpu_v1" / "weights" / "best.pt"


def upload_via_aws_cli(model_path: Path, bucket: str, s3_key: str):
    print(f"[*] Đang tải {model_path} lên s3://{bucket}/{s3_key} ...")
    cmd = [
        "aws", "s3", "cp",
        str(model_path),
        f"s3://{bucket}/{s3_key}",
        "--sse", "AES256"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(res.stdout)
        print(f"[*] Upload thành công: s3://{bucket}/{s3_key}")
        return True
    except FileNotFoundError:
        print("[!] Không tìm thấy công cụ 'aws' CLI trong PATH.")
        print("    Bạn có thể dùng AWS Console Web hoặc cài đặt AWS CLI.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[!] Upload thất bại: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload best.pt lên AWS S3 Private")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Đường dẫn file best.pt")
    parser.add_argument("--bucket", required=True, help="Tên S3 Bucket của bạn")
    parser.add_argument("--key", default="weights/best.pt", help="S3 Object Key")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[!] File weights không tồn tại: {model_path}")
        sys.exit(1)

    ok = upload_via_aws_cli(model_path, args.bucket, args.key)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
