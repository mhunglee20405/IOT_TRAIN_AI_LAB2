"""
calibration.py - Module hiệu chỉnh ngưỡng chấp nhận (Unknown Threshold Calibration)
và đánh giá độ chính xác thực tế cho hệ thống ESP32-CAM + YOLO11n.

Quy tắc phân loại:
- Model YOLO11n có 2 training classes: mask (0), no_mask (1).
- KHÔNG có class thứ 3 trong model.
- Candidate detections được thu thập ở confidence floor thấp (conf_floor >= 0.15 - 0.20).
- Nếu candidate detection có confidence < accept_thresh:
    => Gán nhãn "unknown" (khuôn mặt không đủ độ tin cậy để kết luận).
- Không phát hiện được bounding box (zero box) => Missed Detection (FN),
  tuyệt đối KHÔNG gộp chung hay coi là "unknown".
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import math
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box dạng xyxy chuẩn hóa [0, 1] hoặc tọa độ pixel."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        return (self.width / h) if h > 1e-6 else 1.0


@dataclass
class GroundTruthBox:
    """Đối tượng Ground Truth từ file nhãn YOLO txt."""
    cls_id: int          # 0: mask, 1: no_mask
    box: BoundingBox
    image_name: str = ""
    matched: bool = False

    @property
    def size_category(self) -> str:
        """
        Phân loại kích thước đối tượng theo diện tích chuẩn hóa:
        - Small:  area < 32^2 / 640^2 = 1024 / 409600 = 0.0025
        - Medium: 0.0025 <= area < 96^2 / 640^2 = 9216 / 409600 = 0.0225
        - Large:  area >= 0.0225
        """
        a = self.box.area
        if a < 0.0025:
            return "small"
        elif a < 0.0225:
            return "medium"
        else:
            return "large"


@dataclass
class PredictionBox:
    """Đối tượng Candidate Prediction từ YOLO11n."""
    cls_id: int          # 0: mask, 1: no_mask
    conf: float          # Confidence điểm số [0, 1]
    box: BoundingBox
    image_name: str = ""
    matched_gt: Optional[GroundTruthBox] = None


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """
    Tính Intersection over Union (IoU) giữa 2 BoundingBox.
    """
    x_left = max(box_a.x1, box_b.x1)
    y_top = max(box_a.y1, box_b.y1)
    x_right = min(box_a.x2, box_b.x2)
    y_bottom = min(box_a.y2, box_b.y2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    box_a_area = box_a.area
    box_b_area = box_b.area
    union_area = box_a_area + box_b_area - inter_area

    if union_area <= 1e-9:
        return 0.0

    return inter_area / union_area


def parse_yolo_label_file(label_path: Path, img_name: str = "") -> List[GroundTruthBox]:
    """
    Đọc file annotation chuẩn YOLO format:
    Mỗi dòng: <class_id> <x_center> <y_center> <width> <height>
    Tất cả tọa độ được chuẩn hóa trong khoảng [0, 1].
    """
    boxes = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            x1 = max(0.0, xc - w / 2.0)
            y1 = max(0.0, yc - h / 2.0)
            x2 = min(1.0, xc + w / 2.0)
            y2 = min(1.0, yc + h / 2.0)
            boxes.append(
                GroundTruthBox(
                    cls_id=cls_id,
                    box=BoundingBox(x1, y1, x2, y2),
                    image_name=img_name
                )
            )
    return boxes


def match_predictions_to_ground_truth(
    predictions: List[PredictionBox],
    ground_truths: List[GroundTruthBox],
    iou_thresh: float = 0.50
) -> Tuple[List[Tuple[PredictionBox, GroundTruthBox, float]], List[PredictionBox], List[GroundTruthBox]]:
    """
    Thuật toán Greedy Bipartite Matching theo IoU giảm dần.
    Mỗi prediction khớp với tối đa 1 GT có IoU >= iou_thresh.
    Mỗi GT khớp với tối đa 1 prediction có IoU cao nhất.

    Trả về:
    - matched_pairs: List[(pred, gt, iou)]
    - unmatched_preds: List[pred] (false alarm trên nền background)
    - unmatched_gts: List[gt] (chưa được phát hiện bởi candidate nào)
    """
    # Reset matched status
    for gt in ground_truths:
        gt.matched = False
    for p in predictions:
        p.matched_gt = None

    # Tạo danh sách các cặp ứng viên có IoU >= iou_thresh
    candidate_pairs = []
    for pred in predictions:
        for gt in ground_truths:
            iou = compute_iou(pred.box, gt.box)
            if iou >= iou_thresh:
                candidate_pairs.append((iou, pred, gt))

    # Sắp xếp IoU giảm dần (ưu tiên cặp có độ trùng khớp tốt nhất)
    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pairs = []
    matched_preds_set = set()
    matched_gts_set = set()

    for iou, pred, gt in candidate_pairs:
        pred_id = id(pred)
        gt_id = id(gt)
        if pred_id not in matched_preds_set and gt_id not in matched_gts_set:
            matched_preds_set.add(pred_id)
            matched_gts_set.add(gt_id)
            pred.matched_gt = gt
            gt.matched = True
            matched_pairs.append((pred, gt, iou))

    unmatched_preds = [p for p in predictions if id(p) not in matched_preds_set]
    unmatched_gts = [gt for gt in ground_truths if id(gt) not in matched_gts_set]

    return matched_pairs, unmatched_preds, unmatched_gts


@dataclass
class ThresholdMetrics:
    """
    Tập số liệu đánh giá hoàn chỉnh cho một giá trị accept_thresh xác định.
    Ghi rõ định nghĩa toán học và mẫu số của từng metric.
    """
    accept_thresh: float
    iou_thresh: float
    conf_floor: float

    # Số lượng đếm cơ sở
    total_gt: int
    total_candidates: int
    total_accepted: int
    total_unknown: int
    total_missed_gt: int

    # Phân loại accepted detections
    tp_accepted: int             # Accepted và đúng nhãn GT
    wrong_class_accepted: int    # Accepted nhưng sai nhãn GT (mask vs no_mask)
    fp_accepted: int             # Accepted nhưng không khớp GT nào (báo giả background)

    # Phân loại candidate unknown
    unknown_matched_gt: int      # Candidate có conf < accept_thresh nhưng khớp GT
    unknown_unmatched: int       # Candidate có conf < accept_thresh và không khớp GT

    # Metrics chính (tỷ lệ)
    accepted_precision: float
    coverage: float
    correct_coverage: float
    unknown_rate: float
    false_confident_rate: float
    missed_detection_rate: float

    # Metrics riêng từng class
    mask_metrics: Dict[str, Any] = field(default_factory=dict)
    nomask_metrics: Dict[str, Any] = field(default_factory=dict)

    # Metrics theo điều kiện kích thước khuôn mặt (Small / Medium / Large)
    size_metrics: Dict[str, Any] = field(default_factory=dict)

    # Ma trận nhầm lẫn 3-class + missed
    # Dòng: Ground Truth [mask, no_mask, background]
    # Cột: Prediction/Abstention [pred_mask, pred_no_mask, unknown, missed]
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Điểm chi phí đánh đổi (Cost function)
    cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_threshold_on_matches(
    matched_pairs: List[Tuple[PredictionBox, GroundTruthBox, float]],
    unmatched_preds: List[PredictionBox],
    unmatched_gts: List[GroundTruthBox],
    accept_thresh: float,
    iou_thresh: float = 0.50,
    conf_floor: float = 0.20,
    cost_weights: Optional[Dict[str, float]] = None
) -> ThresholdMetrics:
    """
    Tính toán toàn bộ các metric đánh giá khi áp dụng một ngưỡng accept_thresh cụ thể.

    ĐỊNH NGHĨA VÀ MẪU SỐ CỦA CÁC METRIC:
    -------------------------------------------------------------------------
    1. Total GT (N_gt): Tổng số đối tượng khuôn mặt thực tế trong tập validation.
       Mẫu số của: Coverage, Correct Coverage, Missed Detection Rate.

    2. Total Candidates (N_cand): Tổng số bounding box do YOLO đề xuất tại conf >= conf_floor.
       Mẫu số của: Unknown Rate.

    3. Total Accepted (N_acc): Tổng số bounding box có conf >= accept_thresh.
       N_acc = tp_accepted + wrong_class_accepted + fp_accepted.
       Mẫu số của: Accepted Precision, False-confident Rate.

    4. Accepted Precision:
       P_acc = tp_accepted / N_acc  (nếu N_acc == 0, quy ước bằng 1.0 vì không có lỗi sai nào phát sinh)

    5. Coverage:
       Coverage = (tp_accepted + wrong_class_accepted) / N_gt
       Tỷ lệ khuôn mặt thực tế nhận được quyết định phân loại dứt khoát (không bị unknown hay missed).

    6. Correct Coverage:
       Cov_correct = tp_accepted / N_gt
       Tỷ lệ khuôn mặt thực tế được nhận diện ĐÚNG và chấp nhận tin cậy.

    7. Unknown Rate:
       Unknown Rate = (candidates có conf < accept_thresh) / N_cand
       Tỷ lệ ứng viên bị hệ thống từ chối kết luận, chuyển sang kiểm tra thủ công.

    8. False-confident Rate:
       FCR = (wrong_class_accepted + fp_accepted) / N_acc = 1.0 - Accepted Precision
       Tỷ lệ các quyết định tự tin nhưng sai lầm (rủi ro an ninh/y tế cao nhất).

    9. Missed Detection Rate:
       MDR = total_missed_gt / N_gt
       Tỷ lệ khuôn mặt thực tế hoàn toàn bị YOLO bỏ sót (không có bounding box ứng viên nào).
       LƯU Ý: Missed detection != unknown!
    -------------------------------------------------------------------------
    """
    if cost_weights is None:
        cost_weights = {
            "w_wrong": 10.0,   # Nhầm class cực kỳ nguy hiểm (no_mask tưởng mask)
            "w_fp": 5.0,       # Báo giả background
            "w_miss": 4.0,     # Bỏ sót hoàn toàn đối tượng
            "w_unknown": 1.0   # Unknown tốn công sức kiểm tra lại nhưng an toàn
        }

    total_gt = len(matched_pairs) + len(unmatched_gts)
    total_candidates = len(matched_pairs) + len(unmatched_preds)

    tp_accepted = 0
    wrong_class_accepted = 0
    fp_accepted = 0

    unknown_matched_gt = 0
    unknown_unmatched = 0

    # Phân tích theo class (0: mask, 1: no_mask)
    class_stats = {
        0: {"name": "mask", "gt": 0, "tp": 0, "wrong": 0, "fp": 0, "unknown": 0, "missed": 0},
        1: {"name": "no_mask", "gt": 0, "tp": 0, "wrong": 0, "fp": 0, "unknown": 0, "missed": 0}
    }

    # Phân tích theo kích cỡ (small, medium, large)
    size_stats = {
        "small": {"gt": 0, "tp": 0, "wrong": 0, "unknown": 0, "missed": 0},
        "medium": {"gt": 0, "tp": 0, "wrong": 0, "unknown": 0, "missed": 0},
        "large": {"gt": 0, "tp": 0, "wrong": 0, "unknown": 0, "missed": 0}
    }

    # Ma trận nhầm lẫn
    # Dòng: gt_mask, gt_nomask, bg
    # Cột: pred_mask, pred_nomask, unknown, missed
    cm = {
        "gt_mask": {"pred_mask": 0, "pred_nomask": 0, "unknown": 0, "missed": 0},
        "gt_nomask": {"pred_mask": 0, "pred_nomask": 0, "unknown": 0, "missed": 0},
        "background": {"pred_mask": 0, "pred_nomask": 0, "unknown": 0, "missed": 0}
    }

    # 1. Duyệt qua các cặp đã match
    for pred, gt, iou in matched_pairs:
        gt_cls = gt.cls_id
        pred_cls = pred.cls_id
        gt_size = gt.size_category

        class_stats[gt_cls]["gt"] += 1
        size_stats[gt_size]["gt"] += 1

        gt_row_key = "gt_mask" if gt_cls == 0 else "gt_nomask"

        if pred.conf >= accept_thresh:
            if pred_cls == gt_cls:
                tp_accepted += 1
                class_stats[gt_cls]["tp"] += 1
                size_stats[gt_size]["tp"] += 1
                pred_col_key = "pred_mask" if pred_cls == 0 else "pred_nomask"
                cm[gt_row_key][pred_col_key] += 1
            else:
                wrong_class_accepted += 1
                class_stats[gt_cls]["wrong"] += 1
                size_stats[gt_size]["wrong"] += 1
                pred_col_key = "pred_mask" if pred_cls == 0 else "pred_nomask"
                cm[gt_row_key][pred_col_key] += 1
        else:
            # Dưới ngưỡng chấp nhận => UNKNOWN
            unknown_matched_gt += 1
            class_stats[gt_cls]["unknown"] += 1
            size_stats[gt_size]["unknown"] += 1
            cm[gt_row_key]["unknown"] += 1

    # 2. Duyệt qua các prediction không match GT nào (false alarms)
    for pred in unmatched_preds:
        pred_cls = pred.cls_id
        if pred.conf >= accept_thresh:
            fp_accepted += 1
            class_stats[pred_cls]["fp"] += 1
            pred_col_key = "pred_mask" if pred_cls == 0 else "pred_nomask"
            cm["background"][pred_col_key] += 1
        else:
            unknown_unmatched += 1
            cm["background"]["unknown"] += 1

    # 3. Duyệt qua các GT bị bỏ sót hoàn toàn (không có box ứng viên nào)
    for gt in unmatched_gts:
        gt_cls = gt.cls_id
        gt_size = gt.size_category
        class_stats[gt_cls]["gt"] += 1
        class_stats[gt_cls]["missed"] += 1
        size_stats[gt_size]["gt"] += 1
        size_stats[gt_size]["missed"] += 1
        gt_row_key = "gt_mask" if gt_cls == 0 else "gt_nomask"
        cm[gt_row_key]["missed"] += 1

    total_accepted = tp_accepted + wrong_class_accepted + fp_accepted
    total_unknown = unknown_matched_gt + unknown_unmatched
    total_missed_gt = len(unmatched_gts)

    # Tính toán các tỷ lệ với mẫu số an toàn
    accepted_precision = (tp_accepted / total_accepted) if total_accepted > 0 else 1.0
    coverage = ((tp_accepted + wrong_class_accepted) / total_gt) if total_gt > 0 else 0.0
    correct_coverage = (tp_accepted / total_gt) if total_gt > 0 else 0.0
    unknown_rate = (total_unknown / total_candidates) if total_candidates > 0 else 0.0
    false_confident_rate = ((wrong_class_accepted + fp_accepted) / total_accepted) if total_accepted > 0 else 0.0
    missed_detection_rate = (total_missed_gt / total_gt) if total_gt > 0 else 0.0

    # Chi phí đánh đổi (Cost function)
    cost = (
        cost_weights["w_wrong"] * wrong_class_accepted
        + cost_weights["w_fp"] * fp_accepted
        + cost_weights["w_miss"] * total_missed_gt
        + cost_weights["w_unknown"] * total_unknown
    )

    # Hoàn thiện số liệu cho mask & no_mask
    mask_m = {
        "gt": class_stats[0]["gt"],
        "tp": class_stats[0]["tp"],
        "wrong": class_stats[0]["wrong"],
        "fp": class_stats[0]["fp"],
        "unknown": class_stats[0]["unknown"],
        "missed": class_stats[0]["missed"],
        "precision": (class_stats[0]["tp"] / (class_stats[0]["tp"] + class_stats[0]["wrong"] + class_stats[0]["fp"]))
        if (class_stats[0]["tp"] + class_stats[0]["wrong"] + class_stats[0]["fp"]) > 0 else 1.0,
        "coverage": ((class_stats[0]["tp"] + class_stats[0]["wrong"]) / class_stats[0]["gt"])
        if class_stats[0]["gt"] > 0 else 0.0,
    }
    nomask_m = {
        "gt": class_stats[1]["gt"],
        "tp": class_stats[1]["tp"],
        "wrong": class_stats[1]["wrong"],
        "fp": class_stats[1]["fp"],
        "unknown": class_stats[1]["unknown"],
        "missed": class_stats[1]["missed"],
        "precision": (class_stats[1]["tp"] / (class_stats[1]["tp"] + class_stats[1]["wrong"] + class_stats[1]["fp"]))
        if (class_stats[1]["tp"] + class_stats[1]["wrong"] + class_stats[1]["fp"]) > 0 else 1.0,
        "coverage": ((class_stats[1]["tp"] + class_stats[1]["wrong"]) / class_stats[1]["gt"])
        if class_stats[1]["gt"] > 0 else 0.0,
    }

    # Hoàn thiện số liệu cho size categories
    size_res = {}
    for sz, data in size_stats.items():
        n_gt = data["gt"]
        size_res[sz] = {
            "gt": n_gt,
            "tp": data["tp"],
            "wrong": data["wrong"],
            "unknown": data["unknown"],
            "missed": data["missed"],
            "coverage": ((data["tp"] + data["wrong"]) / n_gt) if n_gt > 0 else 0.0,
            "accuracy": (data["tp"] / (data["tp"] + data["wrong"])) if (data["tp"] + data["wrong"]) > 0 else 1.0
        }

    return ThresholdMetrics(
        accept_thresh=round(accept_thresh, 4),
        iou_thresh=round(iou_thresh, 4),
        conf_floor=round(conf_floor, 4),
        total_gt=total_gt,
        total_candidates=total_candidates,
        total_accepted=total_accepted,
        total_unknown=total_unknown,
        total_missed_gt=total_missed_gt,
        tp_accepted=tp_accepted,
        wrong_class_accepted=wrong_class_accepted,
        fp_accepted=fp_accepted,
        unknown_matched_gt=unknown_matched_gt,
        unknown_unmatched=unknown_unmatched,
        accepted_precision=round(accepted_precision, 4),
        coverage=round(coverage, 4),
        correct_coverage=round(correct_coverage, 4),
        unknown_rate=round(unknown_rate, 4),
        false_confident_rate=round(false_confident_rate, 4),
        missed_detection_rate=round(missed_detection_rate, 4),
        mask_metrics=mask_m,
        nomask_metrics=nomask_m,
        size_metrics=size_res,
        confusion_matrix=cm,
        cost=round(cost, 2)
    )


class ThresholdCalibrator:
    """
    Quản lý việc quét ngưỡng và tìm ngưỡng tối ưu trên tập Validation.
    """

    def __init__(
        self,
        min_thresh: float = 0.30,
        max_thresh: float = 0.90,
        step: float = 0.05,
        iou_thresh: float = 0.50,
        conf_floor: float = 0.20,
        min_precision_constraint: float = 0.95
    ):
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.step = step
        self.iou_thresh = iou_thresh
        self.conf_floor = conf_floor
        self.min_precision_constraint = min_precision_constraint

    def sweep(
        self,
        matched_pairs: List[Tuple[PredictionBox, GroundTruthBox, float]],
        unmatched_preds: List[PredictionBox],
        unmatched_gts: List[GroundTruthBox]
    ) -> List[ThresholdMetrics]:
        """
        Quét ngưỡng chấp nhận từ min_thresh đến max_thresh theo bước step.
        """
        results = []
        curr = self.min_thresh
        while curr <= self.max_thresh + 1e-6:
            m = evaluate_threshold_on_matches(
                matched_pairs=matched_pairs,
                unmatched_preds=unmatched_preds,
                unmatched_gts=unmatched_gts,
                accept_thresh=curr,
                iou_thresh=self.iou_thresh,
                conf_floor=self.conf_floor
            )
            results.append(m)
            curr += self.step
        return results

    def select_optimal_threshold(
        self,
        sweep_results: List[ThresholdMetrics]
    ) -> Tuple[ThresholdMetrics, str]:
        """
        Lựa chọn ngưỡng tối ưu dựa trên 2 tiêu chuẩn:
        1. Ràng buộc an toàn: Maximize Coverage sao cho Accepted Precision >= min_precision_constraint (mặc định 0.95).
        2. Nếu không có ngưỡng nào đạt ràng buộc, chọn ngưỡng có Cost Function nhỏ nhất.

        Trả về: (optimal_metric, lý do lựa chọn).
        """
        valid_candidates = [
            m for m in sweep_results
            if m.accepted_precision >= self.min_precision_constraint
        ]

        if valid_candidates:
            # Chọn ngưỡng đạt Coverage cao nhất trong các ngưỡng thỏa mãn Precision
            # Nếu coverage bằng nhau, chọn ngưỡng có False Confident Rate thấp hơn (tức thresh cao hơn / an toàn hơn)
            best = max(valid_candidates, key=lambda x: (x.coverage, -x.false_confident_rate, -x.cost))
            reason = (
                f"Đạt ràng buộc an toàn: Accepted Precision = {best.accepted_precision:.4f} "
                f"(>= {self.min_precision_constraint:.2f}), tối đa hóa Coverage = {best.coverage:.4f} "
                f"với tỷ lệ Unknown = {best.unknown_rate:.4f}."
            )
            return best, reason
        else:
            # Fallback chọn theo Cost function nhỏ nhất
            best = min(sweep_results, key=lambda x: x.cost)
            reason = (
                f"Không có ngưỡng nào đạt Precision >= {self.min_precision_constraint:.2f}. "
                f"Lựa chọn theo hàm chi phí tối thiểu (Min Cost = {best.cost:.1f}, "
                f"Precision = {best.accepted_precision:.4f}, Coverage = {best.coverage:.4f})."
            )
            return best, reason
