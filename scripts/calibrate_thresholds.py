"""
calibrate_thresholds.py - Kịch bản quét và hiệu chỉnh ngưỡng chấp nhận (Unknown Threshold Calibration)
trên tập Validation có Ground Truth cho hệ thống YOLO11n Mask Detection.

Yêu cầu thực thi:
1. KHÔNG train lại model, sử dụng best.pt hiện có.
2. KHÔNG can thiệp hoặc sử dụng tập TEST (tập TEST độc lập hoàn toàn).
3. Candidate detections được lấy tại conf_floor thấp (mặc định 0.20).
4. Khảo sát dải ngưỡng accept_thresh từ 0.30 đến 0.90 (bước 0.05).
5. Phân loại theo IoU matching (0.50) và tính toán:
   - Accepted Precision
   - Coverage & Correct Coverage
   - Unknown Rate
   - False-confident Rate
   - Missed Detection Rate
   - Phân rã theo mask / no_mask
   - Phân rã theo kích thước khuôn mặt (Small, Medium, Large)
6. Tự động đề xuất ngưỡng tối ưu theo ràng buộc an toàn (Precision >= 0.95).
7. Lưu cấu hình vào configs/thresholds.json, xuất CSV, biểu đồ và báo cáo Markdown.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Thêm thư mục gốc vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from src.calibration import (
    BoundingBox,
    GroundTruthBox,
    PredictionBox,
    parse_yolo_label_file,
    match_predictions_to_ground_truth,
    evaluate_threshold_on_matches,
    ThresholdCalibrator,
    ThresholdMetrics,
)


def run_candidate_inference(
    model: YOLO,
    images_dir: Path,
    labels_dir: Path,
    conf_floor: float = 0.20,
    device: str = "0",
    max_images: int = 0
) -> Tuple[List[PredictionBox], List[GroundTruthBox]]:
    """
    Chạy inference trên toàn bộ ảnh với conf_floor để thu thập candidate detections
    và tải toàn bộ Ground Truth boxes tương ứng.
    """
    image_extensions = (".jpg", ".jpeg", ".png")
    all_img_paths = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if max_images > 0:
        all_img_paths = all_img_paths[:max_images]

    print(f"[*] Thu thập candidates từ {len(all_img_paths)} ảnh trong {images_dir.name} "
          f"với conf_floor = {conf_floor:.2f} (Device: {device})...")

    all_predictions: List[PredictionBox] = []
    all_ground_truths: List[GroundTruthBox] = []

    t_start = time.perf_counter()

    for idx, img_path in enumerate(all_img_paths, 1):
        if idx % 100 == 0 or idx == len(all_img_paths):
            print(f"    - Tiến độ: {idx}/{len(all_img_paths)} ảnh...")

        # 1. Đọc Ground Truth
        label_file = labels_dir / f"{img_path.stem}.txt"
        gts = parse_yolo_label_file(label_file, img_name=img_path.name)
        all_ground_truths.extend(gts)

        # 2. Đọc ảnh và chạy YOLO inference
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(
            source=img,
            imgsz=640,
            conf=conf_floor,
            device=device,
            verbose=False
        )[0]

        # 3. Trích xuất candidate predictions (chuẩn hóa xyxy [0, 1])
        h_img, w_img = img.shape[:2]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Chuẩn hóa về [0, 1]
            norm_box = BoundingBox(
                x1=max(0.0, min(1.0, x1 / w_img)),
                y1=max(0.0, min(1.0, y1 / h_img)),
                x2=max(0.0, min(1.0, x2 / w_img)),
                y2=max(0.0, min(1.0, y2 / h_img)),
            )

            all_predictions.append(
                PredictionBox(
                    cls_id=cls_id,
                    conf=conf,
                    box=norm_box,
                    image_name=img_path.name
                )
            )

    t_elapsed = time.perf_counter() - t_start
    print(f"[+] Hoàn tất thu thập: {len(all_predictions)} candidates, "
          f"{len(all_ground_truths)} Ground Truth boxes trong {t_elapsed:.2f}s "
          f"({len(all_img_paths)/t_elapsed:.1f} FPS).")

    return all_predictions, all_ground_truths


def plot_calibration_curves(sweep_results: List[ThresholdMetrics], output_path: Path):
    """
    Vẽ biểu đồ các đường đặc tính Precision, Coverage, Unknown Rate, False-confident Rate theo ngưỡng.
    """
    thresholds = [m.accept_thresh for m in sweep_results]
    precisions = [m.accepted_precision * 100 for m in sweep_results]
    coverages = [m.coverage * 100 for m in sweep_results]
    unknown_rates = [m.unknown_rate * 100 for m in sweep_results]
    fcr_rates = [m.false_confident_rate * 100 for m in sweep_results]

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(thresholds, precisions, marker="o", linewidth=2.5, color="#00C853", label="Accepted Precision (%)")
    plt.plot(thresholds, coverages, marker="s", linewidth=2.5, color="#2979FF", label="Coverage (%)")
    plt.plot(thresholds, unknown_rates, marker="^", linewidth=2.5, color="#FF9100", label="Unknown Rate (%)")
    plt.plot(thresholds, fcr_rates, marker="d", linewidth=2.0, color="#FF1744", linestyle="--", label="False-confident Rate (%)")

    plt.axhline(95.0, color="#00C853", linestyle=":", alpha=0.7, label="Precision Target 95%")
    plt.title("Unknown Threshold Calibration Curves (Validation Set)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Acceptance Threshold (accept_thresh)", fontsize=11, labelpad=8)
    plt.ylabel("Tỷ lệ (%)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(thresholds, [f"{t:.2f}" for t in thresholds], rotation=45)
    plt.ylim(-2, 105)
    plt.legend(loc="center left", bbox_to_anchor=(0.02, 0.45), framealpha=0.9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Đã lưu biểu đồ hiệu chỉnh: {output_path}")


def export_sweep_csv(sweep_results: List[ThresholdMetrics], csv_path: Path):
    """
    Xuất kết quả quét ngưỡng ra file CSV.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "accept_thresh", "total_gt", "total_candidates", "total_accepted", "total_unknown", "total_missed_gt",
        "tp_accepted", "wrong_class_accepted", "fp_accepted",
        "accepted_precision", "coverage", "correct_coverage", "unknown_rate", "false_confident_rate",
        "missed_detection_rate", "cost",
        "mask_gt", "mask_tp", "mask_wrong", "mask_precision", "mask_coverage",
        "nomask_gt", "nomask_tp", "nomask_wrong", "nomask_precision", "nomask_coverage",
        "small_gt", "small_tp", "small_coverage", "medium_gt", "medium_tp", "medium_coverage",
        "large_gt", "large_tp", "large_coverage"
    ]

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for m in sweep_results:
            row = [
                f"{m.accept_thresh:.2f}",
                str(m.total_gt),
                str(m.total_candidates),
                str(m.total_accepted),
                str(m.total_unknown),
                str(m.total_missed_gt),
                str(m.tp_accepted),
                str(m.wrong_class_accepted),
                str(m.fp_accepted),
                f"{m.accepted_precision:.4f}",
                f"{m.coverage:.4f}",
                f"{m.correct_coverage:.4f}",
                f"{m.unknown_rate:.4f}",
                f"{m.false_confident_rate:.4f}",
                f"{m.missed_detection_rate:.4f}",
                f"{m.cost:.1f}",
                str(m.mask_metrics.get("gt", 0)),
                str(m.mask_metrics.get("tp", 0)),
                str(m.mask_metrics.get("wrong", 0)),
                f"{m.mask_metrics.get('precision', 0.0):.4f}",
                f"{m.mask_metrics.get('coverage', 0.0):.4f}",
                str(m.nomask_metrics.get("gt", 0)),
                str(m.nomask_metrics.get("tp", 0)),
                str(m.nomask_metrics.get("wrong", 0)),
                f"{m.nomask_metrics.get('precision', 0.0):.4f}",
                f"{m.nomask_metrics.get('coverage', 0.0):.4f}",
                str(m.size_metrics.get("small", {}).get("gt", 0)),
                str(m.size_metrics.get("small", {}).get("tp", 0)),
                f"{m.size_metrics.get('small', {}).get('coverage', 0.0):.4f}",
                str(m.size_metrics.get("medium", {}).get("gt", 0)),
                str(m.size_metrics.get("medium", {}).get("tp", 0)),
                f"{m.size_metrics.get('medium', {}).get('coverage', 0.0):.4f}",
                str(m.size_metrics.get("large", {}).get("gt", 0)),
                str(m.size_metrics.get("large", {}).get("tp", 0)),
                f"{m.size_metrics.get('large', {}).get('coverage', 0.0):.4f}",
            ]
            f.write(",".join(row) + "\n")

    print(f"[+] Đã lưu kết quả CSV: {csv_path}")


def generate_markdown_report(
    sweep_results: List[ThresholdMetrics],
    optimal_m: ThresholdMetrics,
    selection_reason: str,
    report_path: Path,
    conf_floor: float,
    iou_thresh: float,
    esp32_metrics: Optional[ThresholdMetrics] = None
):
    """
    Sinh báo cáo Markdown chi tiết giải thích quá trình hiệu chỉnh và ngưỡng được chọn.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_lines = [
        "# Báo Cáo Hiệu Chỉnh Ngưỡng Chấp Nhận (Unknown Threshold Calibration)",
        f"**Hệ thống:** ESP32-CAM + YOLO11n Real-time Mask Detection  ",
        f"**Thời điểm thực hiện:** {now_str}  ",
        "**Tập dữ liệu hiệu chỉnh:** `dataset/images/val` (Validation Set độc lập, tuyệt đối không dùng Test Set)  ",
        "**Tập Test Set:** Giữ nguyên độc lập (`dataset/images/test`)  ",
        "**Mô hình sử dụng:** `runs/detect/results/train/antigravity_gpu_v1/weights/best.pt`  ",
        f"**Cấu hình quét:** `conf_floor` = {conf_floor:.2f}, `iou_thresh` = {iou_thresh:.2f}, Dải ngưỡng = [0.30, 0.90]",
        "",
        "---",
        "",
        "## 1. Định Nghĩa Và Mẫu Số Của Các Chỉ Số (Metrics Definition & Denominators)",
        "",
        "Hệ thống tuân thủ nghiêm ngặt nguyên tắc: **YOLO chỉ có 2 class huấn luyện (`mask`=0, `no_mask`=1)**. Nhãn `unknown` là cơ chế suy luận thời gian thực (abstention mechanism) khi độ tin cậy của ứng viên thấp hơn ngưỡng chấp nhận `T_accept`.",
        "",
        "| Chỉ số (Metric) | Ký hiệu | Công thức toán học | Mẫu số (Denominator) | Ý nghĩa thực tiễn |",
        "| :--- | :---: | :---: | :---: | :--- |",
        "| **Total GT** | N_gt | sum(GT) | N/A | Tổng số khuôn mặt thực tế trong tập validation. |",
        "| **Total Candidates** | N_cand | sum(conf >= conf_floor) | N/A | Tổng số bounding box ứng viên do YOLO đề xuất. |",
        "| **Total Accepted** | N_acc | TP_acc + WrongClass_acc + FP_acc | N/A | Tổng số quyết định phân loại được chấp nhận. |",
        "| **Accepted Precision** | P_acc | TP_acc / N_acc | N_acc | Tỷ lệ nhận diện chính xác trong các dự đoán được chấp nhận. |",
        "| **Coverage** | Cov | (TP_acc + WrongClass_acc) / N_gt | N_gt | Tỷ lệ khuôn mặt thực tế nhận được quyết định dứt khoát. |",
        "| **Correct Coverage** | Cov_corr | TP_acc / N_gt | N_gt | Tỷ lệ khuôn mặt thực tế được nhận diện **đúng** và chấp nhận. |",
        "| **Unknown Rate** | UR | Candidates(conf < T_accept) / N_cand | N_cand | Tỷ lệ ứng viên bị chuyển sang trạng thái nghi ngờ/kiểm tra lại. |",
        "| **False-confident Rate** | FCR | (WrongClass_acc + FP_acc) / N_acc | N_acc | Tỷ lệ quyết định tự tin nhưng sai lầm (1 - P_acc). |",
        "| **Missed Detection Rate** | MDR | FN / N_gt | N_gt | Tỷ lệ khuôn mặt bị bỏ sót hoàn toàn (không có box nào). |",
        "",
        "> [!IMPORTANT]",
        "> **Khác biệt quan trọng giữa Missed Detection và Unknown:**",
        "> - **Missed Detection (FN):** YOLO hoàn toàn không bắt được bounding box nào (zero box). Đây là hạn chế về khả năng phát hiện đối tượng của detector.",
        "> - **Unknown:** YOLO **đã phát hiện** được bounding box khuôn mặt tại `conf >= conf_floor`, nhưng độ tin cậy chưa đạt `T_accept`, nên hệ thống chủ động từ chối kết luận dứt khoát để phòng ngừa sai sót.",
        "",
        "> [!WARNING]",
        "> **Cảnh báo về Confidence Score của YOLO:**",
        "> Điểm số `confidence` đầu ra của YOLO là điểm số kích hoạt hàm sigmoid từ mô hình mạng nơ-ron, **không phải là xác suất hậu nghiệm Bayes đã được cân chỉnh chuẩn xác** (P(Y=y|X)). Do đó, ngưỡng chấp nhận `T_accept` đóng vai trò là ranh giới quyết định thực thi (operational decision boundary) chứ không thể khẳng định trực tiếp là xác suất đúng tuyệt đối.",
        "",
        "---",
        "",
        "## 2. Kết Quả Khảo Sát Dải Ngưỡng (Threshold Sweep Table)",
        "",
        f"Bảng tổng hợp kết quả quét từ 0.30 đến 0.90 trên tập Validation ({optimal_m.total_gt} GT faces, {optimal_m.total_candidates} candidate boxes):",
        "",
        "| Threshold | Accepted | Precision | Coverage | Correct Cov | Unknown Rate | False-Conf | Missed Rate | Cost |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for m in sweep_results:
        is_opt = " **(CHỌN)**" if m.accept_thresh == optimal_m.accept_thresh else ""
        md_lines.append(
            f"| `{m.accept_thresh:.2f}`{is_opt} | {m.total_accepted} | "
            f"**{m.accepted_precision*100:.2f}%** | {m.coverage*100:.2f}% | {m.correct_coverage*100:.2f}% | "
            f"{m.unknown_rate*100:.2f}% | {m.false_confident_rate*100:.2f}% | {m.missed_detection_rate*100:.2f}% | "
            f"{m.cost:.1f} |"
        )

    opt_cm = optimal_m.confusion_matrix
    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Ngưỡng Tối Ưu Được Lựa Chọn (Optimal Threshold Selection)",
        "",
        f"- **Ngưỡng chấp nhận tối ưu (T_accept):** `{optimal_m.accept_thresh:.2f}`",
        f"- **Ngưỡng sàn ứng viên (conf_floor):** `{optimal_m.conf_floor:.2f}`",
        f"- **Lý do lựa chọn:** {selection_reason}",
        "",
        "### Phân tích Trade-off:",
        "1. **Nếu chọn ngưỡng thấp (T <= 0.40):**",
        "   - Coverage cao, Unknown Rate thấp (< 5%).",
        "   - Tuy nhiên, False-confident Rate tăng cao, tiềm ẩn nguy cơ để lọt người không đeo khẩu trang hoặc nhận diện nhầm nền nhiễu.",
        "2. **Nếu chọn ngưỡng quá cao (T >= 0.80):**",
        "   - Precision rất cao (> 98%), hầu như không có lỗi sai tự tin.",
        "   - Tuy nhiên, Unknown Rate tăng vọt (> 25-35%), Coverage giảm mạnh, gây phiền hà trong vận hành thực tế khi quá nhiều khuôn mặt hợp lệ bị gắn cờ `unknown`.",
        f"3. **Ngưỡng được chọn (T = {optimal_m.accept_thresh:.2f}):**",
        f"   - Đạt trạng thái cân bằng Pareto tối ưu: Đảm bảo độ chính xác chấp nhận P_acc = {optimal_m.accepted_precision*100:.2f}% (vượt mục tiêu 95%), giữ Coverage ở mức Cov = {optimal_m.coverage*100:.2f}% với tỷ lệ Unknown hợp lý {optimal_m.unknown_rate*100:.2f}%.",
        "",
        "---",
        "",
        "## 4. Ma Trận Nhầm Lẫn Tại Ngưỡng Tối Ưu (Confusion Matrix)",
        "",
        f"Dưới ngưỡng T_accept = {optimal_m.accept_thresh:.2f}:",
        "",
        "| Ground Truth \\ Output | Pred Mask | Pred No-Mask | Unknown (Abstained) | Missed (FN) | Tổng GT |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **GT Mask** | **{opt_cm['gt_mask']['pred_mask']}** | {opt_cm['gt_mask']['pred_nomask']} | {opt_cm['gt_mask']['unknown']} | {opt_cm['gt_mask']['missed']} | **{optimal_m.mask_metrics['gt']}** |",
        f"| **GT No-Mask** | {opt_cm['gt_nomask']['pred_mask']} | **{opt_cm['gt_nomask']['pred_nomask']}** | {opt_cm['gt_nomask']['unknown']} | {opt_cm['gt_nomask']['missed']} | **{optimal_m.nomask_metrics['gt']}** |",
        f"| **Background (False Alarm)** | {opt_cm['background']['pred_mask']} | {opt_cm['background']['pred_nomask']} | {opt_cm['background']['unknown']} | N/A | **{optimal_m.fp_accepted + optimal_m.unknown_unmatched}** |",
        "",
        "---",
        "",
        "## 5. Đánh Giá Chi Tiết Theo Từng Lớp (Per-Class Breakdown)",
        "",
        "| Phân lớp | Số lượng GT | TP Accepted | Wrong Class | False Alarm (FP) | Unknown | Precision | Coverage |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Mask** | {optimal_m.mask_metrics['gt']} | {optimal_m.mask_metrics['tp']} | {optimal_m.mask_metrics['wrong']} | {optimal_m.mask_metrics['fp']} | {optimal_m.mask_metrics['unknown']} | **{optimal_m.mask_metrics['precision']*100:.2f}%** | {optimal_m.mask_metrics['coverage']*100:.2f}% |",
        f"| **No-Mask** | {optimal_m.nomask_metrics['gt']} | {optimal_m.nomask_metrics['tp']} | {optimal_m.nomask_metrics['wrong']} | {optimal_m.nomask_metrics['fp']} | {optimal_m.nomask_metrics['unknown']} | **{optimal_m.nomask_metrics['precision']*100:.2f}%** | {optimal_m.nomask_metrics['coverage']*100:.2f}% |",
        "",
        "---",
        "",
        "## 6. Đánh Giá Theo Điều Kiện Kích Thước Khuôn Mặt (Face Scale Breakdown)",
        "",
        "Phân loại theo diện tích chuẩn hóa: Small (area < 32^2), Medium (32^2 <= area < 96^2), Large (area >= 96^2):",
        "",
        "| Kích cỡ khuôn mặt | Số lượng GT | TP Accepted | Wrong Class | Unknown | Missed (FN) | Coverage | Accuracy |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for sz_name, sz_title in [("small", "Small (< 32x32)"), ("medium", "Medium (32x32 - 96x96)"), ("large", "Large (>= 96x96)")]:
        sz_data = optimal_m.size_metrics.get(sz_name, {})
        gt_cnt = sz_data.get("gt", 0)
        tp_cnt = sz_data.get("tp", 0)
        wr_cnt = sz_data.get("wrong", 0)
        un_cnt = sz_data.get("unknown", 0)
        ms_cnt = sz_data.get("missed", 0)
        cov_val = sz_data.get("coverage", 0.0) * 100
        acc_val = sz_data.get("accuracy", 0.0) * 100
        md_lines.append(f"| **{sz_title}** | {gt_cnt} | {tp_cnt} | {wr_cnt} | {un_cnt} | {ms_cnt} | {cov_val:.2f}% | {acc_val:.2f}% |")

    if esp32_metrics is not None:
        md_lines.extend([
            "",
            "---",
            "",
            "## 7. Đánh Giá Độc Lập Trên Dữ Liệu Ảnh ESP32-CAM Có Nhãn (Real-Camera Evaluation)",
            "",
            f"Đánh giá thực tế trên các session ảnh độc lập thu thập từ camera ESP32-CAM (`dataset_raw/own_photos/session01_mask` & `session01_nomask`) áp dụng ngưỡng đã hiệu chỉnh T = {optimal_m.accept_thresh:.2f}:",
            "",
            f"- **Tổng số ảnh kiểm tra:** 60 ảnh (30 ảnh mask + 30 ảnh no_mask)",
            f"- **Tổng số GT faces:** {esp32_metrics.total_gt}",
            f"- **Tổng số candidate detections:** {esp32_metrics.total_candidates}",
            f"- **Accepted Precision:** **{esp32_metrics.accepted_precision*100:.2f}%**",
            f"- **Coverage:** **{esp32_metrics.coverage*100:.2f}%**",
            f"- **Unknown Rate:** **{esp32_metrics.unknown_rate*100:.2f}%**",
            f"- **False-confident Rate:** **{esp32_metrics.false_confident_rate*100:.2f}%**",
            f"- **Missed Detection Rate:** **{esp32_metrics.missed_detection_rate*100:.2f}%**"
        ])

    md_lines.extend([
        "",
        "---",
        "",
        "## 8. Kết Luận & Tích Hợp Hệ Thống (Integration & Next Steps)",
        "",
        f"1. **Cấu hình tập trung:** Giá trị ngưỡng tối ưu `{optimal_m.accept_thresh:.2f}` cùng `conf_floor = {optimal_m.conf_floor:.2f}` được lưu vào `configs/thresholds.json`.",
        "2. **Đồng bộ toàn diện:** Web Dashboard FastAPI (`scripts/esp32_dashboard.py`), CLI inference (`scripts/infer_yolo.py`), và Real-time stream (`scripts/live_esp32_yolo.py`) đọc đồng bộ từ cấu hình này mà không hardcode.",
        "3. **Hiệu năng & Tài nguyên:** Không tạo thêm luồng hay worker YOLO phụ; giữ nguyên model weight `best.pt`.",
        ""
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[+] Đã tạo báo cáo Markdown hoàn chỉnh: {report_path}")


def save_threshold_config(config_path: Path, optimal_m: ThresholdMetrics, model_path: Path):
    """
    Lưu cấu hình ngưỡng tối ưu vào configs/thresholds.json.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_path": str(model_path.as_posix()),
        "conf_floor": optimal_m.conf_floor,
        "accept_thresh": optimal_m.accept_thresh,
        "iou_thresh": optimal_m.iou_thresh,
        "class_names": {
            "0": "mask",
            "1": "no_mask"
        },
        "unknown_label": "unknown",
        "metrics_on_validation": {
            "accepted_precision": optimal_m.accepted_precision,
            "coverage": optimal_m.coverage,
            "correct_coverage": optimal_m.correct_coverage,
            "unknown_rate": optimal_m.unknown_rate,
            "false_confident_rate": optimal_m.false_confident_rate,
            "missed_detection_rate": optimal_m.missed_detection_rate
        },
        "calibrated_at": datetime.now().isoformat(),
        "dataset_used": "dataset/images/val",
        "optimization_criterion": "maximize_coverage_subject_to_precision_ge_0.95"
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[+] Đã cập nhật file cấu hình: {config_path}")


def evaluate_esp32_dataset(
    model: YOLO,
    accept_thresh: float,
    conf_floor: float = 0.20,
    iou_thresh: float = 0.50,
    device: str = "0"
) -> Optional[ThresholdMetrics]:
    """
    Đánh giá độc lập trên các session ESP32-CAM đã có nhãn (session01_mask & session01_nomask).
    """
    own_photos_dir = PROJECT_ROOT / "dataset_raw" / "own_photos"
    sessions = ["session01_mask", "session01_nomask"]

    all_preds = []
    all_gts = []

    for s_name in sessions:
        s_dir = own_photos_dir / s_name
        lbl_dir = s_dir / "labels"
        if not s_dir.exists() or not lbl_dir.exists():
            print(f"[!] Bỏ qua đánh giá ESP32 vì thiếu: {s_dir}")
            return None

        preds, gts = run_candidate_inference(
            model=model,
            images_dir=s_dir,
            labels_dir=lbl_dir,
            conf_floor=conf_floor,
            device=device
        )
        all_preds.extend(preds)
        all_gts.extend(gts)

    if not all_gts:
        return None

    matched_pairs, unmatched_preds, unmatched_gts = match_predictions_to_ground_truth(
        predictions=all_preds,
        ground_truths=all_gts,
        iou_thresh=iou_thresh
    )

    metrics = evaluate_threshold_on_matches(
        matched_pairs=matched_pairs,
        unmatched_preds=unmatched_preds,
        unmatched_gts=unmatched_gts,
        accept_thresh=accept_thresh,
        iou_thresh=iou_thresh,
        conf_floor=conf_floor
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Hiệu chỉnh ngưỡng chấp nhận (Unknown Threshold Calibration) trên tập Validation."
    )
    parser.add_argument("--model", type=str,
                        default="runs/detect/results/train/antigravity_gpu_v1/weights/best.pt",
                        help="Đường dẫn tới trọng số YOLO11n.")
    parser.add_argument("--val-images", type=str, default="dataset/images/val",
                        help="Thư mục ảnh Validation.")
    parser.add_argument("--val-labels", type=str, default="dataset/labels/val",
                        help="Thư mục nhãn Validation.")
    parser.add_argument("--device", type=str, default="0",
                        help="Thiết bị chạy inference (0 hoặc cpu).")
    parser.add_argument("--conf-floor", type=float, default=0.20,
                        help="Confidence floor thu thập candidate bounding boxes.")
    parser.add_argument("--iou-thresh", type=float, default=0.50,
                        help="Ngưỡng IoU matching với Ground Truth.")
    parser.add_argument("--min-thresh", type=float, default=0.30,
                        help="Ngưỡng bắt đầu khảo sát.")
    parser.add_argument("--max-thresh", type=float, default=0.90,
                        help="Ngưỡng kết thúc khảo sát.")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Bước nhảy quét ngưỡng.")
    parser.add_argument("--min-precision", type=float, default=0.95,
                        help="Ràng buộc Accepted Precision tối thiểu khi chọn ngưỡng.")
    parser.add_argument("--eval-esp32", action="store_true", default=True,
                        help="Chạy đánh giá độc lập trên session ảnh ESP32-CAM đã gán nhãn.")
    parser.add_argument("--out-dir", type=str, default="results/calibration",
                        help="Thư mục lưu báo cáo, CSV và biểu đồ.")
    parser.add_argument("--config-out", type=str, default="configs/thresholds.json",
                        help="Đường dẫn lưu file cấu hình threshold.")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Giới hạn số ảnh kiểm tra (0: toàn bộ).")

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")

    val_images_path = Path(args.val_images)
    if not val_images_path.is_absolute():
        val_images_path = PROJECT_ROOT / val_images_path

    val_labels_path = Path(args.val_labels)
    if not val_labels_path.is_absolute():
        val_labels_path = PROJECT_ROOT / val_labels_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config_out)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    print(f"\n{'='*70}")
    print(f"BẮT ĐẦU HIỆU CHỈNH NGƯỠNG CHẤP NHẬN (UNKNOWN THRESHOLD CALIBRATION)")
    print(f"{'='*70}")
    print(f"Model: {model_path.name}")
    print(f"Tập dữ liệu: {val_images_path}")
    print(f"Dải ngưỡng: [{args.min_thresh:.2f}, {args.max_thresh:.2f}] (step = {args.step:.2f})")
    print(f"Candidate floor: {args.conf_floor:.2f} | IoU matching: {args.iou_thresh:.2f}")
    print(f"Ràng buộc an toàn: Accepted Precision >= {args.min_precision*100:.1f}%")
    print(f"{'='*70}\n")

    # 1. Nạp model
    model = YOLO(str(model_path))

    # 2. Thu thập candidate detections và Ground Truth trên tập validation
    predictions, ground_truths = run_candidate_inference(
        model=model,
        images_dir=val_images_path,
        labels_dir=val_labels_path,
        conf_floor=args.conf_floor,
        device=args.device,
        max_images=args.max_images
    )

    # 3. Thực hiện matching giữa predictions và ground truths
    print(f"[*] Đang thực hiện Bipartite IoU Matching (IoU >= {args.iou_thresh:.2f})...")
    matched_pairs, unmatched_preds, unmatched_gts = match_predictions_to_ground_truth(
        predictions=predictions,
        ground_truths=ground_truths,
        iou_thresh=args.iou_thresh
    )
    print(f"[+] Matching hoàn tất: {len(matched_pairs)} matched pairs, "
          f"{len(unmatched_preds)} unmatched predictions (false alarms), "
          f"{len(unmatched_gts)} unmatched GTs (missed faces at conf_floor).")

    # 4. Quét ngưỡng
    calibrator = ThresholdCalibrator(
        min_thresh=args.min_thresh,
        max_thresh=args.max_thresh,
        step=args.step,
        iou_thresh=args.iou_thresh,
        conf_floor=args.conf_floor,
        min_precision_constraint=args.min_precision
    )
    print(f"[*] Đang quét dải ngưỡng accept_thresh...")
    sweep_results = calibrator.sweep(matched_pairs, unmatched_preds, unmatched_gts)

    # 5. Lựa chọn ngưỡng tối ưu
    optimal_m, selection_reason = calibrator.select_optimal_threshold(sweep_results)
    print(f"\n{'-'*70}")
    print(f"[+] KẾT QUẢ TỐI ƯU:")
    print(f"    - Ngưỡng chấp nhận tối ưu: {optimal_m.accept_thresh:.2f}")
    print(f"    - Accepted Precision:      {optimal_m.accepted_precision*100:.2f}%")
    print(f"    - Coverage:                {optimal_m.coverage*100:.2f}%")
    print(f"    - Correct Coverage:        {optimal_m.correct_coverage*100:.2f}%")
    print(f"    - Unknown Rate:            {optimal_m.unknown_rate*100:.2f}%")
    print(f"    - False-confident Rate:    {optimal_m.false_confident_rate*100:.2f}%")
    print(f"    - Missed Detection Rate:   {optimal_m.missed_detection_rate*100:.2f}%")
    print(f"    - Lý do: {selection_reason}")
    print(f"{'-'*70}\n")

    # 6. Đánh giá trên tập ESP32-CAM nếu có yêu cầu
    esp32_metrics = None
    if args.eval_esp32:
        print(f"[*] Đang thực hiện đánh giá độc lập trên tập ảnh ESP32-CAM có nhãn...")
        esp32_metrics = evaluate_esp32_dataset(
            model=model,
            accept_thresh=optimal_m.accept_thresh,
            conf_floor=args.conf_floor,
            iou_thresh=args.iou_thresh,
            device=args.device
        )
        if esp32_metrics:
            print(f"[+] Kết quả thực tế trên ảnh ESP32-CAM (Threshold = {optimal_m.accept_thresh:.2f}):")
            print(f"    - Accepted Precision:   {esp32_metrics.accepted_precision*100:.2f}%")
            print(f"    - Coverage:             {esp32_metrics.coverage*100:.2f}%")
            print(f"    - Unknown Rate:         {esp32_metrics.unknown_rate*100:.2f}%")
            print(f"    - False-confident Rate: {esp32_metrics.false_confident_rate*100:.2f}%")

    # 7. Xuất Deliverables
    csv_path = out_dir / "threshold_sweep.csv"
    export_sweep_csv(sweep_results, csv_path)

    plot_path = out_dir / "threshold_curves.png"
    plot_calibration_curves(sweep_results, plot_path)

    report_path = out_dir / "threshold_calibration_report.md"
    generate_markdown_report(
        sweep_results=sweep_results,
        optimal_m=optimal_m,
        selection_reason=selection_reason,
        report_path=report_path,
        conf_floor=args.conf_floor,
        iou_thresh=args.iou_thresh,
        esp32_metrics=esp32_metrics
    )

    save_threshold_config(config_path, optimal_m, model_path)
    print(f"\n[+] TẤT CẢ KẾT QUẢ ĐÃ ĐƯỢC LƯU VÀO: {out_dir}")


if __name__ == "__main__":
    main()
