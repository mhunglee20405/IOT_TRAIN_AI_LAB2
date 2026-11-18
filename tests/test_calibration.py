"""
test_calibration.py - Unit tests cho Unknown Threshold Calibration,
thuật toán Greedy IoU Matching, phân loại candidate unknown,
tính toán metrics đánh giá và lựa chọn ngưỡng tối ưu.
"""

import json
from pathlib import Path
import tempfile
import unittest

from src.calibration import (
    BoundingBox,
    GroundTruthBox,
    PredictionBox,
    compute_iou,
    match_predictions_to_ground_truth,
    evaluate_threshold_on_matches,
    ThresholdCalibrator,
    ThresholdMetrics,
    parse_yolo_label_file
)


class TestIoUCalculation(unittest.TestCase):
    """Kiểm tra hàm tính Intersection over Union (IoU)."""

    def test_identical_boxes(self):
        box1 = BoundingBox(0.1, 0.1, 0.5, 0.5)
        box2 = BoundingBox(0.1, 0.1, 0.5, 0.5)
        self.assertAlmostEqual(compute_iou(box1, box2), 1.0, places=5)

    def test_disjoint_boxes(self):
        box1 = BoundingBox(0.0, 0.0, 0.2, 0.2)
        box2 = BoundingBox(0.5, 0.5, 0.8, 0.8)
        self.assertEqual(compute_iou(box1, box2), 0.0)

    def test_partial_overlap(self):
        # Box 1: [0, 0, 2, 2], Area = 4
        # Box 2: [1, 0, 3, 2], Area = 4
        # Inter: [1, 0, 2, 2], Inter = 1 * 2 = 2
        # Union: 4 + 4 - 2 = 6
        # IoU = 2 / 6 = 1/3 ~ 0.33333
        box1 = BoundingBox(0.0, 0.0, 2.0, 2.0)
        box2 = BoundingBox(1.0, 0.0, 3.0, 2.0)
        self.assertAlmostEqual(compute_iou(box1, box2), 1.0 / 3.0, places=4)

    def test_degenerate_zero_area(self):
        box1 = BoundingBox(0.1, 0.1, 0.1, 0.1)
        box2 = BoundingBox(0.1, 0.1, 0.5, 0.5)
        self.assertEqual(compute_iou(box1, box2), 0.0)


class TestGreedyMatching(unittest.TestCase):
    """Kiểm tra thuật toán ghép cặp Greedy Bipartite Matching."""

    def test_matching_prefers_higher_iou(self):
        # 1 GT tại [0.2, 0.2, 0.6, 0.6]
        gt = GroundTruthBox(cls_id=0, box=BoundingBox(0.2, 0.2, 0.6, 0.6))

        # 2 predictions: pred1 IoU thấp hơn (~0.60), pred2 IoU cao hơn (~0.90)
        pred1 = PredictionBox(cls_id=0, conf=0.80, box=BoundingBox(0.2, 0.2, 0.7, 0.6))
        pred2 = PredictionBox(cls_id=0, conf=0.75, box=BoundingBox(0.2, 0.2, 0.61, 0.61))

        matched_pairs, unmatched_preds, unmatched_gts = match_predictions_to_ground_truth(
            predictions=[pred1, pred2],
            ground_truths=[gt],
            iou_thresh=0.50
        )

        self.assertEqual(len(matched_pairs), 1)
        self.assertEqual(len(unmatched_preds), 1)
        self.assertEqual(len(unmatched_gts), 0)

        # pred2 phải là box được ghép vì IoU cao hơn
        self.assertEqual(id(matched_pairs[0][0]), id(pred2))
        self.assertEqual(id(unmatched_preds[0]), id(pred1))

    def test_below_iou_thresh_not_matched(self):
        gt = GroundTruthBox(cls_id=1, box=BoundingBox(0.0, 0.0, 0.5, 0.5))
        # Box chỉ trùng một góc nhỏ (< 0.50)
        pred = PredictionBox(cls_id=1, conf=0.90, box=BoundingBox(0.4, 0.4, 0.9, 0.9))

        matched_pairs, unmatched_preds, unmatched_gts = match_predictions_to_ground_truth(
            predictions=[pred],
            ground_truths=[gt],
            iou_thresh=0.50
        )

        self.assertEqual(len(matched_pairs), 0)
        self.assertEqual(len(unmatched_preds), 1)
        self.assertEqual(len(unmatched_gts), 1)


class TestEvaluationMetrics(unittest.TestCase):
    """Kiểm tra tính toán các metric hiệu chỉnh ngưỡng và sự khác biệt giữa Unknown và Missed."""

    def setUp(self):
        # Tạo tình huống nhân tạo kiểm thử:
        # GT 1: mask (0)
        # GT 2: mask (0)
        # GT 3: no_mask (1)
        # GT 4: no_mask (1) - BỊ BỎ SÓT HOÀN TOÀN (Missed Detection)
        self.gt1 = GroundTruthBox(cls_id=0, box=BoundingBox(0.1, 0.1, 0.3, 0.3))
        self.gt2 = GroundTruthBox(cls_id=0, box=BoundingBox(0.4, 0.4, 0.6, 0.6))
        self.gt3 = GroundTruthBox(cls_id=1, box=BoundingBox(0.7, 0.7, 0.9, 0.9))
        self.gt4 = GroundTruthBox(cls_id=1, box=BoundingBox(0.0, 0.7, 0.2, 0.9)) # Không ai match

        # Pred 1: Khớp GT 1, đúng class (0), conf = 0.85
        self.pred1 = PredictionBox(cls_id=0, conf=0.85, box=BoundingBox(0.1, 0.1, 0.3, 0.3))
        # Pred 2: Khớp GT 2, đúng class (0), conf = 0.55 (dưới ngưỡng 0.60 => UNKNOWN)
        self.pred2 = PredictionBox(cls_id=0, conf=0.55, box=BoundingBox(0.4, 0.4, 0.6, 0.6))
        # Pred 3: Khớp GT 3, SAI CLASS (dự đoán 0 thay vì 1), conf = 0.70
        self.pred3 = PredictionBox(cls_id=0, conf=0.70, box=BoundingBox(0.7, 0.7, 0.9, 0.9))
        # Pred 4: Không khớp GT nào (báo giả background), conf = 0.65
        self.pred4 = PredictionBox(cls_id=1, conf=0.65, box=BoundingBox(0.5, 0.1, 0.7, 0.3))

        self.matched_pairs = [
            (self.pred1, self.gt1, 1.0),
            (self.pred2, self.gt2, 1.0),
            (self.pred3, self.gt3, 1.0),
        ]
        self.unmatched_preds = [self.pred4]
        self.unmatched_gts = [self.gt4]

    def test_metrics_at_accept_thresh_0_60(self):
        # Với accept_thresh = 0.60:
        # - Pred 1 (conf 0.85 >= 0.60): Đúng class -> TP_accepted = 1
        # - Pred 2 (conf 0.55 < 0.60): -> UNKNOWN (matched) = 1
        # - Pred 3 (conf 0.70 >= 0.60): Sai class -> WrongClass_accepted = 1
        # - Pred 4 (conf 0.65 >= 0.60): Background -> FP_accepted = 1
        # - GT 4: Không match ai -> Missed GT (FN) = 1
        metrics = evaluate_threshold_on_matches(
            matched_pairs=self.matched_pairs,
            unmatched_preds=self.unmatched_preds,
            unmatched_gts=self.unmatched_gts,
            accept_thresh=0.60,
            iou_thresh=0.50,
            conf_floor=0.20
        )

        self.assertEqual(metrics.total_gt, 4)
        self.assertEqual(metrics.total_candidates, 4)
        self.assertEqual(metrics.tp_accepted, 1)
        self.assertEqual(metrics.wrong_class_accepted, 1)
        self.assertEqual(metrics.fp_accepted, 1)
        self.assertEqual(metrics.total_accepted, 3)
        self.assertEqual(metrics.total_unknown, 1)
        self.assertEqual(metrics.total_missed_gt, 1)

        # Accepted Precision = 1 / 3 ~ 0.3333
        self.assertAlmostEqual(metrics.accepted_precision, 1.0 / 3.0, places=4)
        # Coverage = (1 TP + 1 Wrong) / 4 GT = 2 / 4 = 0.50
        self.assertAlmostEqual(metrics.coverage, 0.50, places=4)
        # Unknown Rate = 1 / 4 candidates = 0.25
        self.assertAlmostEqual(metrics.unknown_rate, 0.25, places=4)
        # False-confident Rate = (1 Wrong + 1 FP) / 3 = 2 / 3 ~ 0.6667
        self.assertAlmostEqual(metrics.false_confident_rate, 2.0 / 3.0, places=4)
        # Missed Detection Rate = 1 / 4 GT = 0.25
        self.assertAlmostEqual(metrics.missed_detection_rate, 0.25, places=4)

    def test_missed_is_not_unknown(self):
        """Xác minh Missed Detection (zero box) không bị gộp chung với Unknown."""
        # Thêm 1 candidate unknown không khớp GT nào (background unknown)
        extra_cand = PredictionBox(cls_id=0, conf=0.30, box=BoundingBox(0.9, 0.1, 0.95, 0.2))
        unmatched_preds_copy = list(self.unmatched_preds) + [extra_cand]

        metrics = evaluate_threshold_on_matches(
            matched_pairs=self.matched_pairs,
            unmatched_preds=unmatched_preds_copy,
            unmatched_gts=self.unmatched_gts,
            accept_thresh=0.60,
            iou_thresh=0.50,
            conf_floor=0.20
        )
        # GT total = GT accepted (TP + Wrong) + GT unknown + GT missed
        gt_accounted = (metrics.tp_accepted + metrics.wrong_class_accepted +
                        metrics.unknown_matched_gt + metrics.total_missed_gt)
        self.assertEqual(gt_accounted, metrics.total_gt)

        # total_missed_gt = 1 (do gt4 không có box nào)
        self.assertEqual(metrics.total_missed_gt, 1)
        # total_unknown = 2 (pred2 matched + extra_cand unmatched)
        self.assertEqual(metrics.total_unknown, 2)
        self.assertNotEqual(metrics.total_missed_gt, metrics.total_unknown)


class TestThresholdOptimizer(unittest.TestCase):
    """Kiểm tra thuật toán tìm ngưỡng thỏa mãn ràng buộc Precision >= 0.95."""

    def test_constrained_selection(self):
        calibrator = ThresholdCalibrator(min_precision_constraint=0.90)

        # Giả lập 3 ngưỡng
        m1 = ThresholdMetrics(
            accept_thresh=0.50, iou_thresh=0.50, conf_floor=0.20,
            total_gt=100, total_candidates=100, total_accepted=90, total_unknown=10, total_missed_gt=5,
            tp_accepted=75, wrong_class_accepted=15, fp_accepted=0,
            unknown_matched_gt=10, unknown_unmatched=0,
            accepted_precision=0.8333, coverage=0.90, correct_coverage=0.75,
            unknown_rate=0.10, false_confident_rate=0.1667, missed_detection_rate=0.05
        )
        m2 = ThresholdMetrics(
            accept_thresh=0.70, iou_thresh=0.50, conf_floor=0.20,
            total_gt=100, total_candidates=100, total_accepted=70, total_unknown=30, total_missed_gt=5,
            tp_accepted=65, wrong_class_accepted=5, fp_accepted=0,
            unknown_matched_gt=30, unknown_unmatched=0,
            accepted_precision=0.9286, coverage=0.70, correct_coverage=0.65,
            unknown_rate=0.30, false_confident_rate=0.0714, missed_detection_rate=0.05
        )
        m3 = ThresholdMetrics(
            accept_thresh=0.85, iou_thresh=0.50, conf_floor=0.20,
            total_gt=100, total_candidates=100, total_accepted=50, total_unknown=50, total_missed_gt=5,
            tp_accepted=48, wrong_class_accepted=2, fp_accepted=0,
            unknown_matched_gt=50, unknown_unmatched=0,
            accepted_precision=0.9600, coverage=0.50, correct_coverage=0.48,
            unknown_rate=0.50, false_confident_rate=0.0400, missed_detection_rate=0.05
        )

        # m2 và m3 đều có Precision >= 0.90. m2 có coverage cao hơn (0.70 > 0.50) -> m2 được chọn!
        best, reason = calibrator.select_optimal_threshold([m1, m2, m3])
        self.assertEqual(best.accept_thresh, 0.70)
        self.assertIn("0.9286", reason)


class TestYoloLabelParser(unittest.TestCase):
    """Kiểm tra đọc file label txt định dạng chuẩn YOLO."""

    def test_parse_valid_label(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as f:
            f.write("0 0.5 0.5 0.2 0.4\n1 0.8 0.8 0.1 0.2\n")
            f_path = Path(f.name)

        try:
            boxes = parse_yolo_label_file(f_path)
            self.assertEqual(len(boxes), 2)
            self.assertEqual(boxes[0].cls_id, 0)
            self.assertAlmostEqual(boxes[0].box.x1, 0.4)
            self.assertAlmostEqual(boxes[0].box.x2, 0.6)
            self.assertEqual(boxes[1].cls_id, 1)
        finally:
            if f_path.exists():
                f_path.unlink()


if __name__ == "__main__":
    unittest.main()
