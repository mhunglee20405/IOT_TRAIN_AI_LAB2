"""
label_assistant.py - Công cụ hỗ trợ gán nhãn (Annotation Assistant) và kiểm tra
nhãn thủ công cho các session ảnh chụp từ ESP32-CAM theo chuẩn YOLO format.

Đặc tính:
1. Hoạt động trên các session độc lập trong dataset_raw/own_photos/<session>/
   (ví dụ: session01_mask, session01_nomask) để tránh leakage dữ liệu.
2. Hỗ trợ tiền gán nhãn tự động bằng YOLO11n (semi-automatic pre-annotation)
   với sự kiểm tra ràng buộc class theo kịch bản session:
   - session*_mask: Ưu tiên gán nhãn class 0 (mask).
   - session*_nomask: Ưu tiên gán nhãn class 1 (no_mask).
3. Chế độ kiểm tra (verify): Thống kê số lượng bounding box, phân bố class, kiểm tra
   tọa độ chuẩn hóa hợp lệ [0, 1].
4. Chế độ trực quan hóa (visualize): Xuất ảnh có vẽ bounding box GT để người dùng
   kiểm tra bằng mắt trước khi đưa vào đánh giá độc lập.
5. Tuyệt đối KHÔNG can thiệp vào tập test (dataset/images/test) và KHÔNG tự ý merge vào train.
"""

import argparse
import io
from pathlib import Path
import sys
import cv2
import numpy as np

# Đảm bảo hiển thị UTF-8 trên Windows console
if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Thêm thư mục gốc vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = {0: "mask", 1: "no_mask"}
COLORS = {0: (0, 200, 0), 1: (0, 0, 240)}


def auto_annotate_session(
    session_dir: Path,
    model_path: Path,
    default_class: int,
    conf_floor: float = 0.25,
    device: str = "0",
    overwrite: bool = False
):
    """
    Tự động đề xuất bounding box cho các ảnh trong session_dir và lưu nhãn dạng YOLO txt.
    """
    from ultralytics import YOLO

    labels_dir = session_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = (".jpg", ".jpeg", ".png")
    image_paths = sorted([p for p in session_dir.iterdir() if p.suffix.lower() in image_extensions])

    if not image_paths:
        print(f"[!] Không tìm thấy ảnh nào trong: {session_dir}")
        return

    print(f"[*] Nạp model YOLO11n từ: {model_path}")
    model = YOLO(str(model_path))

    target_class_name = CLASS_NAMES.get(default_class, f"class_{default_class}")
    print(f"[*] Bắt đầu gán nhãn cho {len(image_paths)} ảnh trong {session_dir.name} "
          f"(Target Class: {default_class} - {target_class_name})...")

    annotated_count = 0
    total_boxes = 0

    for img_path in image_paths:
        label_file = labels_dir / f"{img_path.stem}.txt"
        if label_file.exists() and not overwrite:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[!] Không thể đọc ảnh: {img_path.name}")
            continue

        h_img, w_img = img.shape[:2]

        # Dự đoán với conf floor thấp để bắt trọn khuôn mặt
        results = model.predict(source=img, imgsz=640, conf=conf_floor, device=device, verbose=False)[0]

        lines = []
        for box in results.boxes:
            conf = float(box.conf[0])
            pred_cls = int(box.cls[0])

            # Lấy tọa độ chuẩn hóa xywh
            xywh = box.xywhn[0].tolist()
            xc, yc, w, h = xywh

            # Ràng buộc tọa độ trong [0, 1]
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            # Trong session có kiểm soát (session_mask hoặc session_nomask),
            # gán nhãn theo kịch bản session nếu là đối tượng chính
            assigned_cls = default_class

            lines.append(f"{assigned_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            total_boxes += 1

        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        annotated_count += 1

    print(f"[+] Hoàn tất: {annotated_count} ảnh đã gán nhãn, tổng cộng {total_boxes} bounding boxes.")
    print(f"[+] Thư mục nhãn: {labels_dir}")


def verify_session_labels(session_dir: Path):
    """
    Kiểm tra tính hợp lệ của các file nhãn trong session.
    """
    labels_dir = session_dir / "labels"
    image_extensions = (".jpg", ".jpeg", ".png")
    image_paths = sorted([p for p in session_dir.iterdir() if p.suffix.lower() in image_extensions])

    print(f"\n{'='*60}")
    print(f"BÁO CÁO KIỂM TRA NHÃN: {session_dir.name}")
    print(f"{'='*60}")
    print(f"Tổng số ảnh: {len(image_paths)}")

    if not labels_dir.exists():
        print(f"[!] Thư mục labels chưa tồn tại: {labels_dir}")
        return

    label_files = list(labels_dir.glob("*.txt"))
    print(f"Tổng số file label: {len(label_files)}")

    class_counts = {0: 0, 1: 0}
    empty_labels = 0
    invalid_coords = 0
    total_boxes = 0

    for img_path in image_paths:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"[!] Thiếu nhãn cho ảnh: {img_path.name}")
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            empty_labels += 1
            continue

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                print(f"[!] Dòng không hợp lệ trong {lbl_path.name}: '{line}'")
                continue
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:5]))

            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            total_boxes += 1

            for c in coords:
                if c < 0.0 or c > 1.0:
                    invalid_coords += 1

    print(f"Tổng số bounding boxes: {total_boxes}")
    print(f"- Class 0 (mask):    {class_counts.get(0, 0)}")
    print(f"- Class 1 (no_mask): {class_counts.get(1, 0)}")
    print(f"- Số ảnh không có box: {empty_labels}")
    print(f"- Tọa độ ngoài phạm vi [0, 1]: {invalid_coords}")
    print(f"{'='*60}\n")


def visualize_session_labels(session_dir: Path, output_dir: Path, limit: int = 10):
    """
    Vẽ bounding box từ nhãn lên ảnh và lưu vào output_dir để kiểm tra trực quan.
    """
    labels_dir = session_dir / "labels"
    if not labels_dir.exists():
        print(f"[!] Không tìm thấy thư mục nhãn: {labels_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = (".jpg", ".jpeg", ".png")
    image_paths = sorted([p for p in session_dir.iterdir() if p.suffix.lower() in image_extensions])[:limit]

    print(f"[*] Đang xuất {len(image_paths)} ảnh mẫu trực quan vào: {output_dir}...")

    for img_path in image_paths:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])

                x1 = int((xc - w / 2.0) * w_img)
                y1 = int((yc - h / 2.0) * h_img)
                x2 = int((xc + w / 2.0) * w_img)
                y2 = int((yc + h / 2.0) * h_img)

                color = COLORS.get(cls_id, (255, 255, 255))
                label_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img, f"GT: {label_name}", (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        out_file = output_dir / f"gt_{img_path.name}"
        cv2.imwrite(str(out_file), img)

    print(f"[+] Đã lưu ảnh trực quan tại: {output_dir}")


def review_session_labels(session_dir: Path, reviewer: str = "", confirm_all: bool = False):
    """
    Quy trình kiểm tra & xác nhận Ground Truth thủ công từng ảnh cho session.
    Nếu confirm_all=False và không có reviewer, đánh dấu trạng thái là PROVISIONAL.
    """
    labels_dir = session_dir / "labels"
    image_extensions = (".jpg", ".jpeg", ".png")
    image_paths = sorted([p for p in session_dir.iterdir() if p.suffix.lower() in image_extensions])
    status_file = session_dir / "verification_status.json"

    print(f"\n{'='*65}")
    print(f"QUY TRÌNH REVIEW XÁC MINH GROUND TRUTH: {session_dir.name}")
    print(f"{'='*65}")
    print(f"Tổng số ảnh: {len(image_paths)}")

    if not labels_dir.exists():
        print(f"[!] Thư mục labels chưa tồn tại: {labels_dir}")
        return

    import json
    from datetime import datetime

    if confirm_all and reviewer:
        status_data = {
            "session": session_dir.name,
            "verified": True,
            "status": "VERIFIED_MANUAL",
            "reviewed_by": reviewer,
            "reviewed_at": datetime.now().isoformat(),
            "total_images": len(image_paths),
            "notes": "Đã kiểm tra và xác nhận thủ công vị trí bounding box và nhãn class."
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status_data, f, indent=2, ensure_ascii=False)
        print(f"[+] ĐÃ XÁC MINH THÀNH CÔNG: Trạng thái session được cập nhật thành VERIFIED_MANUAL bởi {reviewer}.")
    else:
        # Kiểm tra xem file status hiện có chưa
        is_verified = False
        if status_file.exists():
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    curr_status = json.load(f)
                    is_verified = curr_status.get("verified", False)
            except Exception:
                pass

        if not is_verified:
            status_data = {
                "session": session_dir.name,
                "verified": False,
                "status": "PROVISIONAL",
                "reviewed_by": "unverified",
                "updated_at": datetime.now().isoformat(),
                "total_images": len(image_paths),
                "notes": "Nhãn được tạo tự động bởi YOLO proposal. Cần kiểm tra thủ công từng ảnh để nâng cấp lên VERIFIED."
            }
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(status_data, f, indent=2, ensure_ascii=False)
            print(f"[!] CẢNH BÁO: Session chưa có người xác nhận thủ công từng ảnh!")
            print(f"[!] Trạng thái dữ liệu: PROVISIONAL (Tạm thời - chờ review).")
            print("[*] Để xác nhận nhãn thủ công sau khi kiểm tra ảnh, chạy lệnh:")
            print(f"    .\\venv\\Scripts\\python.exe scripts/label_assistant.py review --session-dir {session_dir} --reviewer 'NguoiReview' --confirm-all")

    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Công cụ hỗ trợ gán nhãn và kiểm tra nhãn YOLO cho ảnh ESP32-CAM."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Lệnh auto
    auto_parser = subparsers.add_parser("auto", help="Tự động đề xuất nhãn cho session ảnh.")
    auto_parser.add_argument("--session-dir", type=str, required=True,
                             help="Đường dẫn tới thư mục session (ví dụ: dataset_raw/own_photos/session01_mask).")
    auto_parser.add_argument("--model", type=str,
                             default="runs/detect/results/train/antigravity_gpu_v1/weights/best.pt",
                             help="Đường dẫn tới trọng số YOLO11n best.pt.")
    auto_parser.add_argument("--default-class", type=int, required=True, choices=[0, 1],
                             help="Class mặc định cho session (0: mask, 1: no_mask).")
    auto_parser.add_argument("--conf-floor", type=float, default=0.20,
                             help="Ngưỡng confidence sàn để bắt candidate faces.")
    auto_parser.add_argument("--device", type=str, default="0",
                             help="Thiết bị chạy model (0 hoặc cpu).")
    auto_parser.add_argument("--overwrite", action="store_true",
                             help="Ghi đè nhãn nếu file đã tồn tại.")

    # Lệnh verify
    verify_parser = subparsers.add_parser("verify", help="Kiểm tra thống kê và tính hợp lệ của file nhãn.")
    verify_parser.add_argument("--session-dir", type=str, required=True,
                               help="Đường dẫn tới thư mục session.")

    # Lệnh review
    review_parser = subparsers.add_parser("review", help="Quy trình xác minh thủ công Ground Truth (PROVISIONAL vs VERIFIED).")
    review_parser.add_argument("--session-dir", type=str, required=True,
                               help="Đường dẫn tới thư mục session.")
    review_parser.add_argument("--reviewer", type=str, default="",
                               help="Tên người thực hiện xác minh nhãn thủ công.")
    review_parser.add_argument("--confirm-all", action="store_true",
                               help="Xác nhận toàn bộ nhãn trong session đã kiểm tra đúng.")

    # Lệnh visualize
    vis_parser = subparsers.add_parser("visualize", help="Vẽ bounding box GT lên ảnh mẫu.")
    vis_parser.add_argument("--session-dir", type=str, required=True,
                            help="Đường dẫn tới thư mục session.")
    vis_parser.add_argument("--output-dir", type=str, default="results/calibration/esp32_annotated_samples",
                            help="Thư mục xuất ảnh kiểm tra.")
    vis_parser.add_argument("--limit", type=int, default=10,
                            help="Số lượng ảnh mẫu tối đa cần xuất.")

    args = parser.parse_args()

    session_path = Path(args.session_dir)
    if not session_path.is_absolute():
        session_path = PROJECT_ROOT / session_path

    if args.command == "auto":
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        auto_annotate_session(
            session_dir=session_path,
            model_path=model_path,
            default_class=args.default_class,
            conf_floor=args.conf_floor,
            device=args.device,
            overwrite=args.overwrite
        )
    elif args.command == "verify":
        verify_session_labels(session_dir=session_path)
    elif args.command == "review":
        review_session_labels(session_dir=session_path, reviewer=args.reviewer, confirm_all=args.confirm_all)
    elif args.command == "visualize":
        out_path = Path(args.output_dir)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        visualize_session_labels(session_dir=session_path, output_dir=out_path, limit=args.limit)


if __name__ == "__main__":
    main()
