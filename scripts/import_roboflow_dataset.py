# -*- coding: utf-8 -*-
import sys
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
scripts/import_roboflow_dataset.py

Import dataset Roboflow (raw) sang canonical dataset/.
Khong sua du lieu goc.

Vi du su dung:
    # Kiem tra truoc (dry-run)
    python scripts/import_roboflow_dataset.py --dry-run

    # Import that
    python scripts/import_roboflow_dataset.py

    # Chi import mot split
    python scripts/import_roboflow_dataset.py --split train
"""

import argparse
import hashlib
import os
import shutil
import sys
import glob
import yaml


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_SOURCE = os.path.join(
    PROJECT_ROOT, "dataset_raw", "public_sources", "roboflow_mask_wearing"
)
DEFAULT_DEST = os.path.join(PROJECT_ROOT, "dataset")

# Roboflow split name -> canonical split name
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test":  "test",
}

PREFIX      = "rfmw_"
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Class mapping Roboflow goc phai khop
EXPECTED_NC    = 2
EXPECTED_NAMES = {0: "mask", 1: "no-mask"}

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def ok(msg):    print(f"  {GREEN}[OK]{RESET}    {msg}")
def err(msg):   print(f"  {RED}[ERR]{RESET}   {msg}")
def warn(msg):  print(f"  {YELLOW}[WARN]{RESET}  {msg}")
def info(msg):  print(f"  {CYAN}[INFO]{RESET}  {msg}")
def skip(msg):  print(f"  [SKIP]  {msg}")

def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Import Roboflow dataset vao canonical dataset/."
    )
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help=f"Thu muc source Roboflow. Default: {DEFAULT_SOURCE}"
    )
    parser.add_argument(
        "--dest", default=DEFAULT_DEST,
        help=f"Thu muc canonical dataset. Default: {DEFAULT_DEST}"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chi kiem tra, khong copy file that."
    )
    parser.add_argument(
        "--split", choices=["train", "valid", "val", "test"],
        default=None,
        help="Chi import mot split cu the. Default: tat ca."
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

def validate_source(source_dir: str) -> bool:
    """Kiem tra source directory va data.yaml."""
    if not os.path.isdir(source_dir):
        err(f"Source khong ton tai: {source_dir}")
        return False

    yaml_path = os.path.join(source_dir, "data.yaml")
    if not os.path.isfile(yaml_path):
        err(f"Khong tim thay data.yaml: {yaml_path}")
        return False

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    nc = cfg.get("nc", None)
    raw_names = cfg.get("names", [])

    # Chap nhan ca list va dict
    if isinstance(raw_names, list):
        names_dict = {i: v for i, v in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names_dict = {int(k): v for k, v in raw_names.items()}
    else:
        err("data.yaml: 'names' khong hop le.")
        return False

    # Kiem tra nc
    if nc != EXPECTED_NC:
        err(f"data.yaml nc={nc}, expected {EXPECTED_NC}.")
        return False

    # Kiem tra class names
    for cid, cname in EXPECTED_NAMES.items():
        actual = names_dict.get(cid, "")
        if actual != cname:
            err(
                f"Class mapping sai: id={cid} expected='{cname}' got='{actual}'\n"
                f"  Expected:\n"
                f"    0 = mask\n"
                f"    1 = no-mask"
            )
            return False

    ok(f"data.yaml hop le: nc={nc}, names={list(names_dict.values())}")
    return True


def validate_split_dirs(source_dir: str, splits: list[str]) -> bool:
    """Kiem tra thu muc images/ va labels/ cho tung split."""
    all_ok = True
    for split in splits:
        for sub in ("images", "labels"):
            d = os.path.join(source_dir, split, sub)
            if os.path.isdir(d):
                ok(f"Tim thay: {split}/{sub}/")
            else:
                err(f"Khong tim thay: {split}/{sub}/")
                all_ok = False
    return all_ok


# ──────────────────────────────────────────────
# Import logic
# ──────────────────────────────────────────────

def get_image_files(directory: str) -> list[str]:
    """Lay danh sach file anh trong thu muc."""
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        files.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    return sorted(files)


def import_split(
    source_dir: str,
    dest_dir: str,
    src_split: str,
    dst_split: str,
    dry_run: bool,
) -> dict:
    """
    Import mot split tu source sang dest.
    Tra ve stats: {copied, skipped, errors, collisions}
    """
    stats = {"copied": 0, "skipped": 0, "errors": 0, "collisions": 0}

    src_img_dir = os.path.join(source_dir, src_split, "images")
    src_lbl_dir = os.path.join(source_dir, src_split, "labels")
    dst_img_dir = os.path.join(dest_dir, "images", dst_split)
    dst_lbl_dir = os.path.join(dest_dir, "labels", dst_split)

    if not dry_run:
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

    image_files = get_image_files(src_img_dir)
    info(f"Tim thay {len(image_files)} anh trong {src_split}/images/")

    for img_path in image_files:
        base_name = os.path.basename(img_path)
        base_stem = os.path.splitext(base_name)[0]
        base_ext  = os.path.splitext(base_name)[1].lower()

        dst_img_name = PREFIX + base_stem + base_ext
        dst_lbl_name = PREFIX + base_stem + ".txt"

        dst_img_path = os.path.join(dst_img_dir, dst_img_name)
        dst_lbl_path = os.path.join(dst_lbl_dir, dst_lbl_name)
        src_lbl_path = os.path.join(src_lbl_dir, base_stem + ".txt")

        # ── Xu ly anh ──
        if os.path.exists(dst_img_path):
            if file_md5(img_path) == file_md5(dst_img_path):
                skip(f"{dst_img_name} da import.")
                stats["skipped"] += 1
            else:
                err(f"COLLISION noi dung khac: {dst_img_name}")
                stats["collisions"] += 1
        else:
            if not dry_run:
                shutil.copy2(img_path, dst_img_path)
            stats["copied"] += 1

        # ── Xu ly label ──
        if os.path.exists(src_lbl_path):
            if os.path.exists(dst_lbl_path):
                pass  # da xu ly cung voi anh
            else:
                if not dry_run:
                    shutil.copy2(src_lbl_path, dst_lbl_path)
        else:
            # Khong co label -> tao file label rong (negative image)
            if not dry_run and not os.path.exists(dst_lbl_path):
                open(dst_lbl_path, "w").close()
            warn(f"Khong co label cho: {base_name} -> tao empty label")

    return stats


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if sys.platform == "win32":
        os.system("color")

    args = parse_args()

    mode = "[DRY-RUN]" if args.dry_run else "[IMPORT]"
    print(f"\n{'=' * 60}")
    print(f"  import_roboflow_dataset.py  {mode}")
    print(f"  Source : {args.source}")
    print(f"  Dest   : {args.dest}")
    print(f"{'=' * 60}\n")

    # Xac dinh splits can import
    if args.split:
        # Normalize: val -> valid (Roboflow dung 'valid')
        s = args.split
        if s == "val":
            s = "valid"
        splits_to_import = {s: SPLIT_MAP.get(s, s)}
    else:
        splits_to_import = SPLIT_MAP  # {"train":"train", "valid":"val", "test":"test"}

    # Validation source
    print(f"-- Kiem tra source --")
    if not validate_source(args.source):
        sys.exit(1)

    if not validate_split_dirs(args.source, list(splits_to_import.keys())):
        sys.exit(1)

    print(f"\n-- Class mapping --")
    info("0 -> mask")
    info("1 -> no-mask (canonical: no_mask)")
    print()

    # Import tung split
    grand_stats = {"copied": 0, "skipped": 0, "errors": 0, "collisions": 0}

    for src_split, dst_split in splits_to_import.items():
        print(f"\n-- [{src_split.upper()} -> {dst_split.upper()}] --")
        stats = import_split(
            source_dir=args.source,
            dest_dir=args.dest,
            src_split=src_split,
            dst_split=dst_split,
            dry_run=args.dry_run,
        )
        info(f"Copied : {stats['copied']}")
        info(f"Skipped: {stats['skipped']}")
        if stats["collisions"]:
            err(f"Collisions: {stats['collisions']}")
        for k in grand_stats:
            grand_stats[k] += stats[k]

    # Tong ket
    print(f"\n{'=' * 60}")
    if grand_stats["collisions"] > 0:
        err(f"Co {grand_stats['collisions']} collision! Kiem tra lai.")
        sys.exit(1)
    elif args.dry_run:
        ok(f"Dry-run OK. Se copy {grand_stats['copied']} file.")
        print(f"  Chay lai khong co --dry-run de import that.")
    else:
        ok(f"Roboflow dataset imported successfully.")
        info(f"Copied : {grand_stats['copied']}")
        info(f"Skipped: {grand_stats['skipped']}")
        print(f"\n  Tiep theo: python scripts/check_dataset.py")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
