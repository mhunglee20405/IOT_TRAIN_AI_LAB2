# -*- coding: utf-8 -*-
"""
scripts/infer_yolo.py  —  YOLO11n Mask Detection Inference
===========================================================
Usage:
    # Single image
    python scripts/infer_yolo.py --source path/to/image.jpg

    # Directory of images
    python scripts/infer_yolo.py --source path/to/folder/

    # Webcam (index 0)
    python scripts/infer_yolo.py --source 0 --show

    # Custom model & threshold
    python scripts/infer_yolo.py --model models/run_20241014_210000/weights/best.pt \
                                  --source images/ --conf 0.35

Confidence-threshold handling:
  - Detections with conf >= --conf are kept and labelled mask / no_mask.
  - Detections with conf < --conf are shown as "unknown" (low-confidence).
  - Zero detections = no bounding box drawn.  This is NOT the same as "unknown".
"""

import sys, io, argparse, os
from datetime import datetime
from pathlib import Path

if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results" / "inference"
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES  = {0: "mask", 1: "no_mask"}
# Colours per class: (B, G, R) — mask=green, no_mask=red, unknown=orange
COLOURS      = {0: (0, 200, 80), 1: (0, 60, 220), "unknown": (0, 140, 255)}

# Nạp ngưỡng calibrated từ configs/thresholds.json
CONFIG_PATH = PROJECT_ROOT / "configs" / "thresholds.json"
CONF_DEFAULT = 0.75  # Calibrated acceptance threshold
CONF_FLOOR = 0.20    # Candidate detection floor
if CONFIG_PATH.exists():
    try:
        import json
        with open(CONFIG_PATH, "r", encoding="utf-8") as _f_cfg:
            _th = json.load(_f_cfg)
            CONF_DEFAULT = float(_th.get("accept_thresh", CONF_DEFAULT))
            CONF_FLOOR = float(_th.get("conf_floor", CONF_FLOOR))
    except Exception:
        pass


G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"; B = "\033[1m"; X = "\033[0m"
def ok(m):   print(f"  {G}[OK]{X}    {m}")
def err(m):  print(f"  {R}[ERR]{X}   {m}")
def warn(m): print(f"  {Y}[WARN]{X}  {m}")
def info(m): print(f"  {C}[INFO]{X}  {m}")


# ─── Auto-find latest best.pt ────────────────────────────────────────────────

def find_latest_model():
    run_dirs = sorted(MODELS_DIR.glob("run_*/weights/best.pt"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if run_dirs:
        return run_dirs[0]
    err("No trained model found in models/. Run train_yolo.py first.")
    sys.exit(1)


# ─── Draw annotated frame ─────────────────────────────────────────────────────

def draw_boxes(img, result, conf_thresh):
    """
    Draw bounding boxes on img (in-place).
    Returns (annotated_img, stats_dict).
    """
    import cv2
    stats = {"mask": 0, "no_mask": 0, "unknown": 0, "no_detection": 0}
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        stats["no_detection"] = 1
        h, w = img.shape[:2]
        cv2.putText(img, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)
        return img, stats

    for box in boxes:
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        if conf >= conf_thresh:
            label    = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
            color    = COLOURS.get(cls_id, (200, 200, 200))
            stats[label] = stats.get(label, 0) + 1
        else:
            label = "unknown"
            color = COLOURS["unknown"]
            stats["unknown"] += 1

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label background
        text      = f"{label} {conf:.2f}"
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - baseline - 2),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img, stats


# ─── Process image list ──────────────────────────────────────────────────────

def process_images(model, image_paths, out_dir, conf_thresh, show):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = {"mask": 0, "no_mask": 0, "unknown": 0, "no_detection": 0}

    for img_path in image_paths:
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            warn(f"Could not read: {img_path}"); continue

        # Run inference with candidate floor so we capture candidates, filter by conf_thresh for known/unknown
        results = model(img, conf=CONF_FLOOR, verbose=False)
        annotated, stats = draw_boxes(img.copy(), results[0], conf_thresh)

        out_path = out_dir / f"pred_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)
        info(f"Saved: {out_path} | mask={stats['mask']} no_mask={stats['no_mask']} "
             f"unknown={stats['unknown']} no_det={stats['no_detection']}")

        for k in totals: totals[k] += stats.get(k, 0)

        if show:
            cv2.imshow("Mask Detection", annotated)
            key = cv2.waitKey(0)
            if key == ord("q"): break

    if show:
        cv2.destroyAllWindows()

    return totals


# ─── Webcam / video stream ────────────────────────────────────────────────────

def process_stream(model, source, out_dir, conf_thresh, show):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        err(f"Cannot open source: {source}"); sys.exit(1)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, conf=0.01, verbose=False)
        annotated, stats = draw_boxes(frame.copy(), results[0], conf_thresh)

        if show:
            cv2.imshow("Mask Detection — press Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        else:
            out_path = out_dir / f"frame_{frame_id:05d}.jpg"
            cv2.imwrite(str(out_path), annotated)
        frame_id += 1

    cap.release()
    if show: cv2.destroyAllWindows()
    info(f"Processed {frame_id} frames.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO11n Mask Detection Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source",  type=str, default=None,
                   help="Image path, directory, or webcam index (0)")
    p.add_argument("--model",   type=str, default=None,
                   help="Path to best.pt (auto-finds latest if omitted)")
    p.add_argument("--conf",    type=float, default=CONF_DEFAULT,
                   help="Confidence threshold; detections below shown as 'unknown'")
    p.add_argument("--out-dir", type=str, default=str(RESULTS_DIR),
                   help="Output directory for annotated images")
    p.add_argument("--show",    action="store_true",
                   help="Display annotated image(s) in window (cv2.imshow)")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--device",  type=str, default="auto")
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32": os.system("color")
    args = parse_args()

    print(f"\n{'='*60}\n  {B}YOLO11n Mask Detection — Inference{X}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}")

    # ── Model ────────────────────────────────────────────────────
    from ultralytics import YOLO
    model_path = Path(args.model) if args.model else find_latest_model()
    if not model_path.exists():
        err(f"Model not found: {model_path}"); sys.exit(1)
    ok(f"Model: {model_path}")

    # Device
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    info(f"Device: {device}")

    model = YOLO(str(model_path))

    # ── Source ───────────────────────────────────────────────────
    source = args.source
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")

    if source is None:
        warn("No --source given. Using dataset/images/test/ by default.")
        source = str(SCRIPT_DIR.parent / "dataset" / "images" / "test")

    src_path = Path(source)
    info(f"Source : {source}")
    info(f"Conf   : {args.conf}")
    info(f"Out    : {out_dir}")
    print()

    # ── Dispatch ─────────────────────────────────────────────────
    if str(source).isdigit():
        # Webcam / video index
        process_stream(model, source, out_dir, args.conf, args.show)

    elif src_path.is_dir():
        imgs = sorted(set(
            p for p in src_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ))
        info(f"Found {len(imgs)} images in {src_path}")
        totals = process_images(model, imgs, out_dir, args.conf, args.show)
        print(f"\n  {B}Summary:{X}")
        for k, v in totals.items():
            print(f"    {k:<14}: {v}")

    elif src_path.is_file():
        totals = process_images(model, [src_path], out_dir, args.conf, args.show)

    else:
        err(f"Source not found: {source}"); sys.exit(1)

    print(f"\n{'='*60}\n  {G}{B}[DONE]{X}  Results saved to: {out_dir}\n{'='*60}\n")


if __name__ == "__main__":
    main()
