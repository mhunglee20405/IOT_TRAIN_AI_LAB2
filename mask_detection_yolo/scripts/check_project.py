# -*- coding: utf-8 -*-
import sys
import io
# Buộc stdout dùng UTF-8 trên Windows (tránh lỗi charmap CP1252)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
scripts/check_project.py

Script kiểm tra cấu trúc project YOLO Mask Detection.
Chạy từ thư mục gốc mask_detection_yolo/:
    python scripts/check_project.py
"""

import os
import sys
import yaml

# ─────────────────────────────────────────────
# Màu sắc console (ANSI) — hoạt động trên Windows 10+
# ─────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):
    print(f"  {GREEN}[OK]{RESET}  {msg}")

def fail(msg):
    print(f"  {RED}[FAIL]{RESET} {msg}")

def header(msg):
    print(f"\n{YELLOW}{'-' * 55}{RESET}")
    print(f"{YELLOW}  {msg}{RESET}")
    print(f"{YELLOW}{'-' * 55}{RESET}")


# ─────────────────────────────────────────────
# Xác định thư mục gốc project
# Script nằm tại: <project_root>/scripts/check_project.py
# ─────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def check_directories() -> list[str]:
    """Kiểm tra các thư mục bắt buộc tồn tại."""
    required_dirs = [
        "dataset",
        "dataset/images",
        "dataset/images/train",
        "dataset/images/val",
        "dataset/images/test",
        "dataset/labels",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/test",
        "dataset_raw",
        "dataset_raw/public_sources",
        "models",
        "results",
        "scripts",
        "src",
    ]

    errors = []
    for d in required_dirs:
        full_path = os.path.join(PROJECT_ROOT, d)
        if os.path.isdir(full_path):
            ok(f"Thư mục tồn tại: {d}/")
        else:
            fail(f"Thư mục THIẾU: {d}/")
            errors.append(d)
    return errors


def check_files() -> list[str]:
    """Kiểm tra các file bắt buộc tồn tại."""
    required_files = [
        "dataset.yaml",
        "requirements.txt",
        ".gitignore",
        "README.md",
        "scripts/check_project.py",
        "scripts/import_roboflow_dataset.py",
        "scripts/check_dataset.py",
        "scripts/visualize_labels.py",
        "scripts/extract_frames.py",
        "dataset/SOURCES.md",
    ]

    errors = []
    for f in required_files:
        full_path = os.path.join(PROJECT_ROOT, f)
        if os.path.isfile(full_path):
            ok(f"File tồn tại: {f}")
        else:
            fail(f"File THIẾU: {f}")
            errors.append(f)
    return errors


def check_dataset_yaml() -> list[str]:
    """
    Kiem tra dataset.yaml:
    - Phai co dung 2 training class: mask, no_mask
    - KHONG duoc co class unknown (unknown la runtime fallback)
    """
    yaml_path = os.path.join(PROJECT_ROOT, "dataset.yaml")
    errors = []

    if not os.path.isfile(yaml_path):
        fail("dataset.yaml khong ton tai, bo qua kiem tra noi dung.")
        return ["dataset.yaml missing"]

    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            fail(f"dataset.yaml khong parse duoc YAML: {e}")
            return ["dataset.yaml parse error"]

    # Kiem tra keys
    for key in ("path", "train", "val", "test", "names", "nc"):
        if key in cfg:
            ok(f"dataset.yaml co key '{key}'")
        else:
            fail(f"dataset.yaml THIEU key '{key}'")
            errors.append(f"missing key: {key}")

    names = cfg.get("names", {})
    nc    = cfg.get("nc", None)

    # Kiem tra nc = 2
    if nc == 2:
        ok(f"nc = 2 (2 training classes)")
    else:
        fail(f"nc = {nc}, phai la 2.")
        errors.append("nc != 2")

    # Kiem tra KHONG co unknown
    if isinstance(names, dict) and 2 in names:
        fail("[ERROR] Training dataset must contain exactly 2 classes: mask, no_mask.")
        fail(f"  Tim thay class 2='{names[2]}' — unknown chi la runtime fallback, khong phai training class!")
        errors.append("class 2 (unknown) found in training config")

    # Kiem tra 2 class chinh
    expected = {0: "mask", 1: "no_mask"}
    for cid, cname in expected.items():
        actual = names.get(cid) if isinstance(names, dict) else None
        if actual == cname:
            ok(f"Class ID {cid}: '{cname}'  OK")
        else:
            fail(f"Class ID {cid} sai! Expected='{cname}', Got='{actual}'")
            errors.append(f"class_id {cid} wrong")

    return errors


def check_no_forbidden_items() -> list[str]:
    """Kiểm tra không tồn tại các thứ không được phép ở Bước 1."""
    forbidden_dirs = ["runs"]  # YOLO tự tạo khi train
    warnings = []

    for d in forbidden_dirs:
        full_path = os.path.join(PROJECT_ROOT, d)
        if os.path.isdir(full_path):
            print(f"  {YELLOW}[WARN]{RESET} Thư mục '{d}/' tồn tại — có thể đã chạy training?")
            warnings.append(d)

    return warnings


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # Bật màu trên Windows
    if sys.platform == "win32":
        os.system("color")

    print(f"\n{'=' * 55}")
    print(f"  YOLO Mask Detection - Project Structure Check")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"{'=' * 55}")

    all_errors = []

    header("1. Kiểm tra thư mục")
    all_errors += check_directories()

    header("2. Kiểm tra file")
    all_errors += check_files()

    header("3. Kiểm tra nội dung dataset.yaml")
    all_errors += check_dataset_yaml()

    header("4. Kiểm tra items không được phép")
    check_no_forbidden_items()  # chỉ warning, không tính lỗi

    # ─── Kết quả tổng hợp ───
    print(f"\n{'=' * 55}")
    if not all_errors:
        print(f"  {GREEN}[OK] Project structure is valid.{RESET}")
        print(f"  San sang cho BUOC 2: Thu thap & chuan hoa dataset.")
    else:
        print(f"  {RED}[FAIL] Co {len(all_errors)} loi can sua:{RESET}")
        for e in all_errors:
            print(f"    - {e}")
        print(f"\n  Vui long sua cac loi tren truoc khi tiep tuc.")
    print(f"{'=' * 55}\n")

    sys.exit(0 if not all_errors else 1)


if __name__ == "__main__":
    main()
