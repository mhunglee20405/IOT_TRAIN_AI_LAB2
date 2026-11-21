# -*- coding: utf-8 -*-
"""
scripts/train_yolo.py  —  YOLO11n Mask Detection Training Pipeline
===================================================================
Usage (from project root, with venv active):
    python scripts/train_yolo.py
    python scripts/train_yolo.py --epochs 100 --batch 8
    python scripts/train_yolo.py --device cpu

Steps:
  1. Validate dataset (abort on errors)
  2. Data leakage check across splits
  3. Auto-detect device (CPU/GPU), cap batch on CPU
  4. Pre-flight report (counts, config)
  5. Fine-tune yolo11n.pt — raw data never touched
  6. Save to models/run_YYYYMMDD_HHMMSS/  (best.pt, last.pt, results.csv, plots)
  7. Evaluate on test split, per-class P/R/mAP
  8. Write markdown training_report.md to run dir
"""

import sys, io, argparse, os
from datetime import datetime
from pathlib import Path

if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR  = PROJECT_ROOT / "dataset"
MODELS_DIR   = PROJECT_ROOT / "models"
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"

VALID_CLASS_IDS = {0, 1}
CLASS_NAMES     = {0: "mask", 1: "no_mask"}
IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"; B = "\033[1m"; X = "\033[0m"
def ok(m):   print(f"  {G}[OK]{X}    {m}")
def err(m):  print(f"  {R}[ERR]{X}   {m}")
def warn(m): print(f"  {Y}[WARN]{X}  {m}")
def info(m): print(f"  {C}[INFO]{X}  {m}")
def hdr(t):  print(f"\n{'─'*62}\n  {B}{t}{X}\n{'─'*62}")


# ─── Dataset helpers ────────────────────────────────────────────────────────

def count_images(split):
    d = DATASET_DIR / "images" / split
    if not d.is_dir(): return 0
    # Use case-insensitive suffix check to avoid double-counting on Windows
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def count_labels(split):
    counts = {0: 0, 1: 0, "empty": 0}
    d = DATASET_DIR / "labels" / split
    if not d.is_dir(): return counts
    for lp in d.glob("*.txt"):
        lines = [ln.strip() for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            counts["empty"] += 1
            continue
        for ln in lines:
            parts = ln.split()
            if len(parts) == 5:
                try:
                    cid = int(parts[0])
                    if cid in VALID_CLASS_IDS: counts[cid] += 1
                except ValueError:
                    pass
    return counts


def validate_dataset():
    errors = []
    for split in ["train", "val", "test"]:
        lbl_dir = DATASET_DIR / "labels" / split
        if not lbl_dir.is_dir():
            errors.append(f"Missing labels dir: {lbl_dir}"); continue
        split_err = 0
        for lp in lbl_dir.glob("*.txt"):
            lines = lp.read_text(encoding="utf-8").splitlines()
            non_empty = [ln for ln in lines if ln.strip()]
            if not non_empty: continue
            for i, raw in enumerate(lines, 1):
                ln = raw.strip()
                if not ln: continue
                parts = ln.split()
                if len(parts) != 5:
                    errors.append(f"[{split}] {lp.name}:{i} need 5 vals, got {len(parts)}")
                    split_err += 1; continue
                try:
                    cid = int(parts[0])
                    x,y,w,h = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
                except ValueError:
                    errors.append(f"[{split}] {lp.name}:{i} parse error")
                    split_err += 1; continue
                if cid not in VALID_CLASS_IDS:
                    errors.append(f"[{split}] {lp.name}:{i} invalid class={cid}"); split_err += 1
                for nm, v in [("x",x),("y",y),("w",w),("h",h)]:
                    if not (0.0 <= v <= 1.0):
                        errors.append(f"[{split}] {lp.name}:{i} {nm}={v:.4f} out of range"); split_err += 1
                if w <= 0 or h <= 0:
                    errors.append(f"[{split}] {lp.name}:{i} w={w} h={h} <=0"); split_err += 1
                if split_err >= 20:
                    errors.append(f"[{split}] stopped after 20 errors"); break
            if split_err >= 20: break
    return errors


def check_leakage():
    hdr("Data Leakage Check")
    sets = {}
    for split in ["train", "val", "test"]:
        d = DATASET_DIR / "images" / split
        if not d.is_dir(): continue
        stems = set()
        for e in IMAGE_EXTS:
            for p in d.glob(f"*{e}"): stems.add(p.stem)
            for p in d.glob(f"*{e.upper()}"): stems.add(p.stem)
        sets[split] = stems
    sl = list(sets.keys()); found = False
    for i in range(len(sl)):
        for j in range(i+1, len(sl)):
            common = sets[sl[i]] & sets[sl[j]]
            if common:
                found = True
                warn(f"{len(common)} shared stems between {sl[i]} and {sl[j]}")
                for s in list(common)[:3]: warn(f"  overlap: {s}")
    if not found: ok("No data leakage detected between splits.")


# ─── Device & batch ─────────────────────────────────────────────────────────

def detect_device(requested):
    if requested and requested != "auto": return requested
    try:
        import torch
        if torch.cuda.is_available():
            info(f"GPU: {torch.cuda.get_device_name(0)}"); return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info("Apple MPS detected."); return "mps"
    except ImportError: pass
    info("No GPU — using CPU."); return "cpu"


def cap_batch(device, batch):
    if device == "cpu" and batch > 8:
        warn(f"CPU: capping batch {batch} -> 8"); return 8
    return batch


# ─── Pre-flight ──────────────────────────────────────────────────────────────

def print_preflight(args, device, batch, run_dir):
    hdr("PRE-FLIGHT REPORT")
    print(f"\n  {'Split':<8} {'Images':>7} {'mask':>8} {'no_mask':>10} {'negative':>10}")
    print(f"  {'─'*48}")
    for sp in ["train","val","test"]:
        ni = count_images(sp); c = count_labels(sp)
        print(f"  {sp:<8} {ni:>7} {c[0]:>8} {c[1]:>10} {c['empty']:>10}")
    hdr("Training Configuration")
    cfg = [
        ("Model","yolo11n.pt (YOLO11 nano)"), ("Dataset YAML", str(DATASET_YAML)),
        ("Output dir", str(run_dir)), ("Device", device),
        ("Epochs", args.epochs), ("Patience", args.patience),
        ("Batch", batch), ("Image size", args.imgsz),
        ("Workers", args.workers), ("Optimizer", args.optimizer),
        ("LR0 / LRF", f"{args.lr0} / {args.lrf}"),
        ("Mosaic", args.mosaic), ("FlipUD / FlipLR", f"{args.flipud} / {args.fliplr}"),
    ]
    for k,v in cfg: print(f"  {k:<18}: {v}")
    print()


# ─── Training ────────────────────────────────────────────────────────────────

def run_training(args, device, batch, run_dir):
    from ultralytics import YOLO
    hdr("Starting Training")
    info("Loading yolo11n.pt (downloads if needed)...")
    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(DATASET_YAML), epochs=args.epochs, imgsz=args.imgsz,
        batch=batch, device=device, patience=args.patience, workers=args.workers,
        optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf,
        mosaic=args.mosaic, flipud=args.flipud, fliplr=args.fliplr,
        seed=42, project=str(MODELS_DIR), name=run_dir.name,
        exist_ok=True, save=True, save_period=-1, plots=True,
        verbose=True, amp=(device != "cpu"),
    )
    return model, results


# ─── Evaluation ──────────────────────────────────────────────────────────────

def run_evaluation(run_dir, device):
    from ultralytics import YOLO
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        err(f"best.pt not found: {best_pt}"); return None
    hdr("Test-Set Evaluation (best.pt)")
    model = YOLO(str(best_pt))
    metrics = model.val(
        data=str(DATASET_YAML), split="test", device=device,
        imgsz=640, batch=16, plots=True, save_json=False,
        project=str(run_dir), name="eval_test", exist_ok=True, verbose=True,
    )
    print(); hdr("Evaluation Results — Test Set")
    mp50   = getattr(metrics.box,"map50",float("nan"))
    mp5095 = getattr(metrics.box,"map",  float("nan"))
    mp     = getattr(metrics.box,"mp",   float("nan"))
    mr     = getattr(metrics.box,"mr",   float("nan"))
    print(f"  {'Metric':<22} {'Value':>10}\n  {'─'*34}")
    for lbl, val in [("Precision (mean)",mp),("Recall (mean)",mr),("mAP50",mp50),("mAP50-95",mp5095)]:
        print(f"  {lbl:<22} {val:>10.4f}")
    print(f"\n  Per-class:\n  {'Class':<14} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print(f"  {'─'*50}")
    try:
        names = model.names
        for i, cid in enumerate(getattr(metrics.box,"ap_class_index",[])):
            cname = names.get(int(cid), f"cls{cid}")
            _p = (getattr(metrics.box,"p",[]+[0]*99))[i]
            _r = (getattr(metrics.box,"r",[]+[0]*99))[i]
            _a50 = (getattr(metrics.box,"ap50",[]+[0]*99))[i]
            _a   = (getattr(metrics.box,"ap"  ,[]+[0]*99))[i]
            print(f"  {cname:<14} {_p:>8.4f} {_r:>8.4f} {_a50:>8.4f} {_a:>10.4f}")
    except Exception as ex:
        warn(f"Per-class parse error: {ex}")
    return metrics


# ─── Markdown report ─────────────────────────────────────────────────────────

def write_report(run_dir, args, device, batch, eval_metrics=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# YOLO11n Mask Detection — Training Report\n\n",
        f"**Date:** {ts}  \n**Run dir:** `{run_dir}`  \n\n",
        "## Dataset\n\n| Split | Images | mask | no_mask | negative |\n|---|---:|---:|---:|---:|\n",
    ]
    for sp in ["train","val","test"]:
        n = count_images(sp); c = count_labels(sp)
        lines.append(f"| {sp} | {n} | {c[0]} | {c[1]} | {c['empty']} |\n")
    lines += [
        "\n## Config\n\n| Param | Value |\n|---|---|\n",
        f"| Model | yolo11n.pt |\n| Device | {device} |\n| Epochs | {args.epochs} |\n",
        f"| Patience | {args.patience} |\n| Batch | {batch} |\n| Imgsz | {args.imgsz} |\n",
        f"| Optimizer | {args.optimizer} |\n| LR0/LRF | {args.lr0}/{args.lrf} |\n",
        "\n## Test-Set Metrics\n\n",
    ]
    if eval_metrics is not None:
        try:
            lines += [
                "| Metric | Value |\n|---|---|\n",
                f"| Precision | {eval_metrics.box.mp:.4f} |\n",
                f"| Recall | {eval_metrics.box.mr:.4f} |\n",
                f"| mAP50 | {eval_metrics.box.map50:.4f} |\n",
                f"| mAP50-95 | {eval_metrics.box.map:.4f} |\n",
            ]
        except Exception: lines.append("_Metrics unavailable._\n")
    else:
        lines.append("_Evaluation skipped._\n")
    lines += [
        "\n## Output Files\n\n",
        "- `weights/best.pt` — Best checkpoint\n",
        "- `weights/last.pt` — Last checkpoint\n",
        "- `results.csv` — Training curves\n",
        "- `confusion_matrix.png` — Val confusion matrix\n",
        "- `eval_test/confusion_matrix.png` — Test confusion matrix\n",
        "\n> Raw Roboflow dataset NOT modified. Local Antigravity/Windows run.\n",
    ]
    rp = run_dir / "training_report.md"
    rp.write_text("".join(lines), encoding="utf-8")
    info(f"Report -> {rp}")
    return rp


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO11n mask detection",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--imgsz",     type=int,   default=640)
    p.add_argument("--batch",     type=int,   default=16, help="Auto-capped to 8 on CPU")
    p.add_argument("--patience",  type=int,   default=15)
    p.add_argument("--device",    type=str,   default="auto", help="auto|cpu|cuda:0|mps")
    p.add_argument("--workers",   type=int,   default=4,  help="Use 0 on Windows if hangs")
    p.add_argument("--optimizer", type=str,   default="AdamW", help="SGD|Adam|AdamW")
    p.add_argument("--lr0",       type=float, default=1e-3)
    p.add_argument("--lrf",       type=float, default=1e-2)
    p.add_argument("--mosaic",    type=float, default=1.0)
    p.add_argument("--flipud",    type=float, default=0.0)
    p.add_argument("--fliplr",    type=float, default=0.5)
    p.add_argument("--skip-eval", action="store_true", help="Skip test evaluation")
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32": os.system("color")
    args = parse_args()
    print(f"\n{'='*62}\n  {B}YOLO11n Mask Detection — Training Pipeline{X}")
    print(f"  Antigravity/Windows — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*62}")

    hdr("Dataset Validation")
    val_errors = validate_dataset()
    if val_errors:
        err(f"{len(val_errors)} errors — fix before training:")
        for e in val_errors[:15]: print(f"    {e}")
        sys.exit(1)
    ok("Dataset validation passed.")

    check_leakage()

    device  = detect_device(args.device)
    batch   = cap_batch(device, args.batch)
    run_dir = MODELS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print_preflight(args, device, batch, run_dir)
    run_training(args, device, batch, run_dir)

    eval_metrics = None
    if not args.skip_eval:
        eval_metrics = run_evaluation(run_dir, device)

    report_path = write_report(run_dir, args, device, batch, eval_metrics)

    best_pt = run_dir / "weights" / "best.pt"
    print(f"\n{'='*62}\n  {G}{B}[DONE] Training complete!{X}")
    print(f"  Best model : {best_pt}")
    print(f"  Run dir    : {run_dir}")
    print(f"  Report     : {report_path}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
