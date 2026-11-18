# -*- coding: utf-8 -*-
import sys
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
scripts/check_dataset.py

Kiem tra tinh hop le cua dataset YOLO Mask Detection.
Chay tu thu muc goc mask_detection_yolo/:
    python scripts/check_dataset.py

Kiem tra:
  - Anh co label tuong ung
  - Label co anh tuong ung
  - Class ID chi duoc 0, 1, 2
  - Moi dong label co dung 5 gia tri
  - Toa do nam trong range hop le
  - Width/height > 0
  - Thong ke so luong instance theo split
  - Canh bao class imbalance
"""

import os
import glob


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR  = os.path.join(PROJECT_ROOT, "dataset")

VALID_CLASS_IDS = {0, 1}
CLASS_NAMES     = {0: "mask", 1: "no_mask"}
IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS          = ["train", "val", "test"]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def ok(msg):   print(f"  {GREEN}[OK]{RESET}   {msg}")
def err(msg):  print(f"  {RED}[ERR]{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}[WARN]{RESET} {msg}")
def info(msg): print(f"  {CYAN}[INFO]{RESET} {msg}")

def header(title):
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def get_images(split: str) -> dict:
    """Lay tat ca anh trong split, tra ve dict {stem: path}."""
    img_dir = os.path.join(DATASET_DIR, "images", split)
    if not os.path.isdir(img_dir):
        return {}
    stems = {}
    for ext in IMAGE_EXTS:
        for p in glob.glob(os.path.join(img_dir, f"*{ext}")):
            stem = os.path.splitext(os.path.basename(p))[0]
            stems[stem] = p
        for p in glob.glob(os.path.join(img_dir, f"*{ext.upper()}")):
            stem = os.path.splitext(os.path.basename(p))[0]
            stems[stem] = p
    return stems


def get_labels(split: str) -> dict:
    """Lay tat ca label .txt trong split, tra ve dict {stem: path}."""
    lbl_dir = os.path.join(DATASET_DIR, "labels", split)
    if not os.path.isdir(lbl_dir):
        return {}
    return {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(lbl_dir, "*.txt"))
    }


def check_label_file(label_path: str) -> tuple[list[str], dict, bool]:
    """
    Kiem tra noi dung mot file label YOLO.
    Tra ve (danh sach loi, dict so luong instance theo class, is_empty).
    Empty label = negative image = hop le.
    """
    errors = []
    counts = {0: 0, 1: 0}

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # File label trong = negative image -> hop le
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return errors, counts, True  # empty = valid negative

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue

        parts = line.split()

        # Phai co dung 5 gia tri
        if len(parts) != 5:
            errors.append(
                f"  Dong {line_no}: can 5 gia tri, got {len(parts)} | {label_path}"
            )
            continue

        # Parse gia tri
        try:
            class_id = int(parts[0])
            x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            errors.append(f"  Dong {line_no}: gia tri khong parse duoc | {label_path}")
            continue

        # Kiem tra class ID (chi 0 va 1)
        if class_id not in VALID_CLASS_IDS:
            errors.append(
                f"  Dong {line_no}: class_id={class_id} KHONG HOP LE "
                f"(chi duoc 0=mask, 1=no_mask) | {label_path}"
            )
        else:
            counts[class_id] += 1

        # Kiem tra range toa do
        for name, val in [("x_center", x_c), ("y_center", y_c), ("width", w), ("height", h)]:
            if not (0.0 <= val <= 1.0):
                errors.append(
                    f"  Dong {line_no}: {name}={val:.6f} ngoai range [0,1] | {label_path}"
                )

        # Width va height phai > 0
        if w <= 0:
            errors.append(f"  Dong {line_no}: width={w} phai > 0 | {label_path}")
        if h <= 0:
            errors.append(f"  Dong {line_no}: height={h} phai > 0 | {label_path}")

    return errors, counts, False


# ──────────────────────────────────────────────
# Per-split check
# ──────────────────────────────────────────────

def check_split(split: str) -> tuple[int, dict, list[str]]:
    """
    Kiem tra mot split (train/val/test).
    Tra ve (so anh, tong instance theo class, danh sach loi).
    """
    header(f"Split: {split.upper()}")

    img_dict = get_images(split)
    lbl_dict = get_labels(split)
    img_stems = set(img_dict.keys())
    lbl_stems = set(lbl_dict.keys())
    all_errors = []

    img_count = len(img_stems)
    lbl_count = len(lbl_stems)

    info(f"Images : {img_count}")
    info(f"Labels : {lbl_count}")

    # Anh khong co label
    no_label = img_stems - lbl_stems
    if no_label:
        warn(f"{len(no_label)} anh KHONG co label tuong ung:")
        for s in sorted(no_label)[:5]:
            print(f"    - {s}")
        if len(no_label) > 5:
            print(f"    ... va {len(no_label) - 5} anh khac")

    # Label khong co anh
    no_image = lbl_stems - img_stems
    if no_image:
        warn(f"{len(no_image)} label KHONG co anh tuong ung:")
        for s in sorted(no_image)[:5]:
            print(f"    - {s}")
        if len(no_image) > 5:
            print(f"    ... va {len(no_image) - 5} label khac")

    # Kiem tra duplicate filename (chi can kiem tra trong images)
    all_basenames = [os.path.basename(p) for p in img_dict.values()]
    seen = set()
    duplicates = []
    for bn in all_basenames:
        if bn in seen:
            duplicates.append(bn)
        seen.add(bn)
    if duplicates:
        warn(f"{len(duplicates)} duplicate filename:")
        for d in duplicates[:5]:
            print(f"    - {d}")

    # Kiem tra noi dung tung file label
    lbl_dir = os.path.join(DATASET_DIR, "labels", split)
    total_counts  = {0: 0, 1: 0}
    format_errors = []
    empty_count   = 0

    if os.path.isdir(lbl_dir):
        for lbl_file in glob.glob(os.path.join(lbl_dir, "*.txt")):
            file_errors, counts, is_empty = check_label_file(lbl_file)
            format_errors.extend(file_errors)
            if is_empty:
                empty_count += 1
            else:
                for k in total_counts:
                    if k in counts:
                        total_counts[k] += counts[k]

    all_errors.extend(format_errors)

    total_instances = sum(total_counts.values())
    print()
    info(f"Total instances : {total_instances}")
    info(f"Empty labels (negative images): {empty_count}")
    for class_id, name in CLASS_NAMES.items():
        pct = (total_counts[class_id] / total_instances * 100) if total_instances > 0 else 0
        bar = "#" * int(pct / 5)
        print(f"    {name:12s} (id={class_id}): {total_counts[class_id]:5d}  [{bar:<20}] {pct:.1f}%")

    if format_errors:
        err(f"{len(format_errors)} loi format label:")
        for e in format_errors[:10]:
            print(e)
        if len(format_errors) > 10:
            print(f"    ... va {len(format_errors) - 10} loi khac")
    else:
        if img_count > 0:
            ok("Tat ca label hop le.")

    return img_count, total_counts, all_errors


# ──────────────────────────────────────────────
# Class imbalance warning
# ──────────────────────────────────────────────

def warn_imbalance(total_counts: dict):
    """Canh bao neu co class imbalance qua lon (chi 2 class)."""
    vals = [v for v in total_counts.values() if v > 0]
    if len(vals) < 2:
        return
    ratio = max(vals) / min(vals)
    if ratio > 5:
        dominant = max(total_counts, key=total_counts.get)
        minority = min((k for k in total_counts if total_counts[k] > 0), key=lambda k: total_counts[k])
        warn(
            f"Class imbalance cao: "
            f"{CLASS_NAMES[dominant]}({total_counts[dominant]}) / "
            f"{CLASS_NAMES[minority]}({total_counts[minority]}) = {ratio:.1f}x"
        )
        warn("Nen them anh de can bang truoc khi train.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if sys.platform == "win32":
        os.system("color")

    print("\n" + "=" * 60)
    print("  YOLO Mask Detection - Dataset Validator")
    print(f"  Dataset dir: {DATASET_DIR}")
    print("=" * 60)

    all_errors_total = []
    grand_total_counts = {0: 0, 1: 0}
    grand_total_images = 0

    for split in SPLITS:
        img_count, counts, errs = check_split(split)
        all_errors_total.extend(errs)
        grand_total_images += img_count
        for k in grand_total_counts:
            grand_total_counts[k] += counts[k]

    # Tong ket
    header("TONG KET")
    info(f"Tong anh     : {grand_total_images}")
    total_inst = sum(grand_total_counts.values())
    info(f"Tong instance: {total_inst}")
    for class_id, name in CLASS_NAMES.items():
        print(f"    {name:12s}: {grand_total_counts[class_id]}")

    warn_imbalance(grand_total_counts)

    print("\n" + "=" * 60)
    if not all_errors_total:
        print(f"  {GREEN}[OK] Dataset validation passed.{RESET}")
        if grand_total_images == 0:
            print(f"  {YELLOW}[NOTE] Dataset dang trong - hay them anh va annotation.{RESET}")
        else:
            print(f"  San sang cho buoc annotation / training config.")
    else:
        print(f"  {RED}[FAIL] Co {len(all_errors_total)} loi can sua.{RESET}")
        print(f"  Sua cac loi tren truoc khi train.")
    print("=" * 60 + "\n")

    sys.exit(0 if not all_errors_total else 1)


if __name__ == "__main__":
    main()
