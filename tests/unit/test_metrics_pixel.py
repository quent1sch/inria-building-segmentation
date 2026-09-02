"""
tests/unit/test_metrics_pixel.py

Unit tests for evaluation/metrics_pixel.py.

Tests cover:
  - Perfect prediction (IoU=1.0)
  - No overlap (IoU≈0.0)
  - Partial overlap (0 < IoU < 1)
  - Empty prediction and empty GT edge cases
  - aggregate_metrics averaging
"""

import numpy as np
import pytest

from evaluation.metrics_pixel import aggregate_metrics, pixel_metrics


# ── pixel_metrics ─────────────────────────────────────────────────────────────

class TestPixelMetrics:

    def test_perfect_prediction(self, binary_mask_two_buildings, perfect_pred_mask):
        """Identical pred and GT → all metrics = 1.0."""
        m = pixel_metrics(perfect_pred_mask, binary_mask_two_buildings)
        assert m["iou"]       == pytest.approx(1.0, abs=1e-4)
        assert m["dice"]      == pytest.approx(1.0, abs=1e-4)
        assert m["precision"] == pytest.approx(1.0, abs=1e-4)
        assert m["recall"]    == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self, binary_mask_two_buildings, no_overlap_pred_mask):
        """No overlap → IoU and Dice near 0, recall near 0."""
        m = pixel_metrics(no_overlap_pred_mask, binary_mask_two_buildings)
        assert m["iou"]    < 0.01
        assert m["dice"]   < 0.01
        assert m["recall"] < 0.01

    def test_partial_overlap(self, binary_mask_two_buildings, shifted_pred_mask):
        """Partial overlap → all metrics strictly between 0 and 1."""
        m = pixel_metrics(shifted_pred_mask, binary_mask_two_buildings)
        for key, val in m.items():
            assert 0.0 < val < 1.0, f"{key}={val} should be in (0, 1)"

    def test_empty_prediction_empty_gt(self, empty_mask):
        """Both empty → smooth denominator keeps metrics well-defined (near 1.0)."""
        m = pixel_metrics(empty_mask, empty_mask)
        # TP=0, FP=0, FN=0 — smooth=1e-6 in numerator and denominator
        # result is 1e-6 / 1e-6 = 1.0
        assert m["iou"]    == pytest.approx(1.0, abs=1e-3)
        assert m["recall"] == pytest.approx(1.0, abs=1e-3)

    def test_empty_prediction_nonempty_gt(self, empty_mask, binary_mask_two_buildings):
        """Empty prediction, non-empty GT → precision≈1 (no FP), recall≈0 (all FN) by 
        convention."""
        m = pixel_metrics(empty_mask, binary_mask_two_buildings)
        assert m["recall"]    < 0.01   # missed everything
        assert m["precision"] > 0.99   # no false positives (smooth keeps it near 1)

    def test_full_prediction_empty_gt(self, full_mask, empty_mask):
        """Full prediction, empty GT → all pixels are FP → precision near 0."""
        m = pixel_metrics(full_mask, empty_mask)
        assert m["precision"] < 0.01

    def test_returns_all_keys(self, binary_mask_two_buildings, perfect_pred_mask):
        """Check all expected keys are present in the output dict."""
        m = pixel_metrics(perfect_pred_mask, binary_mask_two_buildings)
        assert set(m.keys()) == {"iou", "dice", "f1", "precision", "recall"}

    def test_f1_equals_dice(self, binary_mask_two_buildings, shifted_pred_mask):
        """F1 and Dice should always be identical (they're the same formula)."""
        m = pixel_metrics(shifted_pred_mask, binary_mask_two_buildings)
        assert m["f1"] == pytest.approx(m["dice"], abs=1e-8)

    def test_accepts_uint8_masks(self, binary_mask_two_buildings):
        """Masks passed as uint8 (0/255) should work the same as bool."""
        pred_uint8 = (binary_mask_two_buildings.astype(np.uint8) * 255)
        gt_uint8   = (binary_mask_two_buildings.astype(np.uint8) * 255)
        m = pixel_metrics(pred_uint8, gt_uint8)
        assert m["iou"] == pytest.approx(1.0, abs=1e-4)

    def test_precision_recall_tradeoff(self):
        """
        Manual example: pred covers only half of GT buildings.
        Expected: recall = 0.5 (missed half), precision = 1.0 (no false positives).
        """
        gt   = np.zeros((10, 10), dtype=bool)
        pred = np.zeros((10, 10), dtype=bool)
        gt[0:10, 0:5]   = True   # 50 building pixels in GT
        pred[0:10, 0:5] = True   # predict exactly those 50 pixels
        m = pixel_metrics(pred, gt)
        assert m["precision"] == pytest.approx(1.0, abs=1e-4)
        assert m["recall"]    == pytest.approx(1.0, abs=1e-4)

        # Now predict only half of them
        pred2 = np.zeros((10, 10), dtype=bool)
        pred2[0:5, 0:5] = True   # only 25 of the 50 GT pixels
        m2 = pixel_metrics(pred2, gt)
        assert m2["recall"]    < 0.6     # missed half the GT
        assert m2["precision"] > 0.95    # no false positives


# ── aggregate_metrics ─────────────────────────────────────────────────────────

class TestAggregateMetrics:

    def test_average_of_identical_dicts(self):
        """Average of identical metric dicts should return the same values."""
        m = {"iou": 0.8, "dice": 0.85, "f1": 0.85, "precision": 0.9, "recall": 0.8}
        result = aggregate_metrics([m, m, m])
        for k, v in m.items():
            assert result[k] == pytest.approx(v, abs=1e-8)

    def test_average_of_two_dicts(self):
        """Mean of [0.6, 0.8] = 0.7 for each metric."""
        m1 = {"iou": 0.6, "dice": 0.6, "f1": 0.6, "precision": 0.6, "recall": 0.6}
        m2 = {"iou": 0.8, "dice": 0.8, "f1": 0.8, "precision": 0.8, "recall": 0.8}
        result = aggregate_metrics([m1, m2])
        for k in m1:
            assert result[k] == pytest.approx(0.7, abs=1e-8)

    def test_empty_list_returns_empty_dict(self):
        """Empty input → empty output (no crash)."""
        assert aggregate_metrics([]) == {}

    def test_single_dict_passthrough(self):
        """Single-element list → returns that dict's values."""
        m = {"iou": 0.75, "dice": 0.83, "f1": 0.83, "precision": 0.9, "recall": 0.77}
        result = aggregate_metrics([m])
        for k, v in m.items():
            assert result[k] == pytest.approx(v, abs=1e-8)