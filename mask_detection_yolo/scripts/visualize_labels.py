# -*- coding: utf-8 -*-
import sys
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
scripts/visualize_labels.py

Ve bounding box YOLO len anh de kiem tra annotation.
Luu preview vao results/dataset_preview/.
Khong thay doi anh goc.

Vi du su dung:
    # Xem ngau nhien 10 anh tu train split
    python scripts/visualize_labels.py --split train --n 10

    # Xem tat ca anh tu val split, luu preview
    python scripts/visualize_labels.py --split val --save

    # Xem mot anh cu the
    python scripts/visualize_labels.py --image dataset/images/train/abc.jpg
"""

import argparse
import os
import random
import glob
import cv2
import numpy as np


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR  = os.path.join(PROJECT_ROOT, "dataset")
PREVIEW_DIR  = os.path.join(PROJECT_ROOT, "results", "dataset_preview")

CLASS_NAMES = {0: "mask", 1: "no_mask"}

# Mau sac cho tung class (BGR)
CLASS_COLORS = {
    0: (50,  200,  50),   # mask    -> xanh la
    1: (50,   50, 220),   # no_mask -> do
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO annotations tren anh."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--split", "-s",
        choices=["train", "val", "test"],
        default="train",
        help="Split de visualize (train/val/test). Default: train."
    )
    group.add_argument(
        "--image", "-i",
        help="Duong dan truc tiep den mot anh cu the."
    )
    parser.add_argument(
        "--n", "--count",
        dest="n",
        type=int,
        default=10,
        help="So luong anh ngau nhien khi dung --split. Default=10. 0=tat ca."
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Luu preview ra results/dataset_preview/ thay vi hien thi."
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Hien thi anh qua cua so (can GUI). Mac dinh chi luu neu --save."
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Do day duong vien bounding box. Default=2."
    )
    return parser.parse_args()


def find_label_path(image_path: str) -> str | None:
    """Tim file label .txt tuong ung voi anh."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    # Thu nhieu vi tri label co the
    candidates = [
        image_path.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\"),
    ]
    for c in candidates:
        label_path = os.path.splitext(c)[0] + ".txt"
        if os.path.isfile(label_path):
            return label_path
    return None


def draw_boxes(image_path: str, label_path: str | None, thickness: int) -> np.ndarray:
    """
    Doc anh va ve bounding box YOLO.
    Tra ve anh da ve box (numpy array BGR).
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [ERR] Khong doc duoc anh: {image_path}")
        return None

    h, w = img.shape[:2]

    if label_path is None or not os.path.isfile(label_path):
        # Ve watermark "NO LABEL"
        cv2.putText(img, "NO LABEL", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return img

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            x_c = float(parts[1]) * w
            y_c = float(parts[2]) * h
            bw  = float(parts[3]) * w
            bh  = float(parts[4]) * h
        except ValueError:
            continue

        # Tinh toa do goc
        x1 = int(x_c - bw / 2)
        y1 = int(y_c - bh / 2)
        x2 = int(x_c + bw / 2)
        y2 = int(y_c + bh / 2)

        # Clip trong anh
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color = CLASS_COLORS.get(class_id, (200, 200, 200))
        label_text = CLASS_NAMES.get(class_id, f"id={class_id}")

        # Ve bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Ve label background
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(img,
                      (x1, label_y - th - 4),
                      (x1 + tw + 6, label_y + 2),
                      color, -1)

        # Ve text
        cv2.putText(img, label_text,
                    (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Ve ten file goc o goc trai tren
    cv2.putText(img, os.path.basename(image_path),
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)

    return img


def get_images_in_split(split: str) -> list[str]:
    """Lay danh sach duong dan anh trong split."""
    img_dir = os.path.join(DATASET_DIR, "images", split)
    if not os.path.isdir(img_dir):
        return []
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext.upper()}")))
    return sorted(paths)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    os.makedirs(PREVIEW_DIR, exist_ok=True)

    # Xac dinh danh sach anh can visualize
    if args.image:
        image_paths = [args.image]
        split_label = "custom"
    else:
        split_label = args.split
        image_paths = get_images_in_split(args.split)

        if not image_paths:
            print(f"[WARN] Khong tim thay anh nao trong split '{args.split}'.")
            print(f"  Hay them anh vao: dataset/images/{args.split}/")
            sys.exit(0)

        # Lay ngau nhien n anh
        if args.n > 0 and args.n < len(image_paths):
            image_paths = random.sample(image_paths, args.n)

    print(f"\n{'=' * 55}")
    print(f"  visualize_labels.py  |  split={split_label}  |  n={len(image_paths)}")
    print(f"{'=' * 55}\n")

    saved_count = 0
    for img_path in image_paths:
        lbl_path = find_label_path(img_path)
        vis = draw_boxes(img_path, lbl_path, args.thickness)

        if vis is None:
            continue

        # Luu preview
        if args.save or not args.show:
            out_name = f"{split_label}_{os.path.basename(img_path)}"
            out_path = os.path.join(PREVIEW_DIR, out_name)
            cv2.imwrite(out_path, vis)
            saved_count += 1
            print(f"  [Saved] {out_path}")

        # Hien thi qua cua so neu yeu cau
        if args.show:
            cv2.imshow("Annotation Preview (q to quit)", vis)
            key = cv2.waitKey(0)
            if key == ord("q") or key == 27:
                break

    if args.show:
        cv2.destroyAllWindows()

    print(f"\n[OK] Da luu {saved_count} preview anh vao: {PREVIEW_DIR}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
