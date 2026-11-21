# -*- coding: utf-8 -*-
"""
scripts/extract_frames.py

Trich frame tu video de chuan bi dataset YOLO.
Default: 1 frame moi giay (interval=1.0).

Vi du su dung:
    python scripts/extract_frames.py \
        --video dataset_raw/own_videos/group01.mp4 \
        --output dataset_raw/extracted_frames \
        --interval 1.0
"""

import argparse
import os
import sys
import cv2


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Trich frame tu video theo khoang thoi gian co dinh."
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Duong dan den file video (mp4, avi, mov, ...)."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Thu muc luu frame anh output."
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        help="Khoang cach giua 2 frame (giay). Default=1.0."
    )
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=95,
        help="Chat luong JPEG (1-100). Default=95."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Tien to ten file. Default: ten video (khong co extension)."
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    """Loai bo ky tu dac biet khoi ten file."""
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    return "".join(c if c in keep else "_" for c in name)


def extract_frames(video_path: str, output_dir: str, interval: float,
                   quality: int, prefix: str) -> int:
    """
    Mo video va trich frame theo interval (giay).
    Tra ve so frame da luu.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Khong mo duoc video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"  Video : {video_path}")
    print(f"  FPS   : {fps:.2f}")
    print(f"  Frames: {total_frames}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Interval: {interval}s -> lay ~{int(duration / interval)} frames")
    print(f"  Output: {output_dir}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # So frame phai skip giua 2 lan lay
    frame_step = max(1, int(round(fps * interval)))

    saved_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chi lay frame theo buoc
        if frame_idx % frame_step == 0:
            out_name = f"{prefix}_frame_{saved_count + 1:06d}.jpg"
            out_path = os.path.join(output_dir, out_name)

            # Khong overwrite neu file da ton tai
            if os.path.exists(out_path):
                print(f"  [SKIP] Da ton tai: {out_name}")
            else:
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                saved_count += 1

                # In tien trinh moi 10 frame
                if saved_count % 10 == 0:
                    print(f"  Saved {saved_count} frames...", end="\r")

        frame_idx += 1

    cap.release()
    return saved_count


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # Kiem tra video ton tai
    if not os.path.isfile(args.video):
        print(f"[ERROR] File video khong ton tai: {args.video}")
        sys.exit(1)

    # Xac dinh prefix
    if args.prefix:
        prefix = sanitize_name(args.prefix)
    else:
        base = os.path.splitext(os.path.basename(args.video))[0]
        prefix = sanitize_name(base)

    print("=" * 55)
    print("  extract_frames.py - Video Frame Extractor")
    print("=" * 55)

    saved = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        interval=args.interval,
        quality=args.quality,
        prefix=prefix,
    )

    print(f"\n[OK] Hoan thanh! Da luu {saved} frames vao: {args.output}")
    print("=" * 55)


if __name__ == "__main__":
    main()
