# -*- coding: utf-8 -*-
"""
cloud_api/scripts/download_model_s3.py — Tải model weights từ AWS S3 xuống EC2.
=============================================================================
Sử dụng EC2 IAM Role (Instance Profile) để xác thực an toàn không lưu credentials.

Sử dụng trên EC2:
    python cloud_api/scripts/download_model_s3.py --bucket my-mask-detection-bucket --key weights/best.pt --dest /opt/mask_detection/weights/best.pt
"""

import argparse
from pathlib import Path
import subprocess
import sys


def download_model(bucket: str, s3_key: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Tải model từ s3://{bucket}/{s3_key} về {dest_path} ...")

    # 1. Thử dùng AWS CLI
    cmd = ["aws", "s3", "cp", f"s3://{bucket}/{s3_key}", str(dest_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(res.stdout)
        print(f"[*] Tải model thành công: {dest_path} ({dest_path.stat().st_size} bytes)")
        return True
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError as e:
        print(f"[!] Lỗi AWS CLI: {e.stderr}")
        return False

    # 2. Thử dùng boto3 nếu có
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.download_file(bucket, s3_key, str(dest_path))
        print(f"[*] Tải model qua boto3 thành công: {dest_path}")
        return True
    except ImportError:
        print("[!] Không tìm thấy cả 'aws' CLI lẫn thư viện 'boto3'. Hãy cài đặt AWS CLI hoặc boto3.")
        return False
    except Exception as e:
        print(f"[!] Lỗi tải từ S3 qua boto3: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Tải model weights từ S3 xuống EC2")
    parser.add_argument("--bucket", required=True, help="Tên S3 Bucket")
    parser.add_argument("--key", default="weights/best.pt", help="S3 Object Key")
    parser.add_argument("--dest", default="/opt/mask_detection/weights/best.pt", help="Đường dẫn đích lưu model")
    args = parser.parse_args()

    dest_p = Path(args.dest)
    success = download_model(args.bucket, args.key, dest_p)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
