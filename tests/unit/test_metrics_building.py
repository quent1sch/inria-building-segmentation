"""
tests/unit/test_metrics_building.py

Unit tests for evaluation/metrics_building.py.

Tests cover:
  - polygon_iou: exact overlap, no overlap, partial overlap
  - match_polygons: perfect match, no match, partial match, greedy priority
  - _size_stratified: correct bin assignment and recall computation
"""

import numpy as np
import pytest
from shapely.geometry import box

from evaluation.metrics_building import (
    _aggregate_size_strata,
    _size_stratified,
    match_polygons,
    polygon_iou,
)


# ── polygon_iou ───────────────────────────────────────────────────────────────

class TestPolygonIou:

    def test_identical_polygons(self):
        """Identical polygons → IoU = 1.0."""
        p = box(0, 0, 10, 10)
        assert polygon_iou(p, p) == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self):
        """Non-overlapping polygons → IoU = 0.0."""
        p1 = box(0, 0, 10, 10)
        p2 = box(20, 20, 30, 30)
        assert polygon_iou(p1, p2) == pytest.approx(0.0, abs=1e-6)

    def test_50_percent_overlap(self):
        """50% overlap → IoU = area_intersection / area_union = 50/150 ≈ 0.333."""
        p1 = box(0, 0, 10, 10)    # area = 100
        p2 = box(5, 0, 15, 10)    # area = 100, intersection = 50
        # union = 100 + 100 - 50 = 150
        assert polygon_iou(p1, p2) == pytest.approx(50 / 150, abs=1e-4)

    def test_full_containment(self):
        """Small polygon fully inside large → IoU = small_area / large_area."""
        large = box(0, 0, 10, 10)   # area = 100
        small = box(2, 2, 5, 5)     # area = 9
        # intersection = 9, union = 100
        assert polygon_iou(small, large) == pytest.approx(9 / 100, abs=1e-4)


# ── match_polygons ────────────────────────────────────────────────────────────

class TestMatchPolygons:

    def test_perfect_one_to_one_match(self):
        """One pred exactly overlapping one GT → 1 match, 0 unmatched."""
        pred  = [box(0, 0, 10, 10)]
        gt    = [box(0, 0, 10, 10)]
        matches, unmatched_pred, unmatched_gt = match_polygons(pred, gt, iou_threshold=0.5)
        assert len(matches)        == 1
        assert len(unmatched_pred) == 0
        assert len(unmatched_gt)   == 0
        assert matches[0][2] == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap_no_match(self):
        """No overlap → 0 matches, both pred and GT unmatched."""
        pred = [box(0, 0, 5, 5)]
        gt   = [box(50, 50, 60, 60)]
        matches, unmatched_pred, unmatched_gt = match_polygons(pred, gt, iou_threshold=0.5)
        assert len(matches)        == 0
        assert len(unmatched_pred) == 1
        assert len(unmatched_gt)   == 1

    def test_below_threshold_not_matched(self):
        """Overlap below iou_threshold → not a match."""
        pred = [box(0, 0, 10, 10)]
        gt   = [box(5, 0, 15, 10)]   # IoU ≈ 0.333
        matches, unmatched_pred, unmatched_gt = match_polygons(
            pred, gt, iou_threshold=0.5
        )
        assert len(matches) == 0

    def test_above_threshold_matched(self):
        """Overlap above threshold → matched."""
        pred = [box(0, 0, 10, 10)]
        gt   = [box(5, 0, 15, 10)]   # IoU ≈ 0.333
        matches, unmatched_pred, unmatched_gt = match_polygons(
            pred, gt, iou_threshold=0.3   # lower threshold
        )
        assert len(matches) == 1

    def test_greedy_best_match_wins(self):
        """
        Two predictions both overlapping the same GT polygon.
        The higher-IoU prediction should win.
        """
        gt    = [box(0, 0, 10, 10)]
        pred1 = box(0, 0, 10, 10)    # perfect match — IoU = 1.0
        pred2 = box(1, 1, 11, 11)    # partial match — IoU < 1.0
        pred  = [pred1, pred2]

        matches, unmatched_pred, unmatched_gt = match_polygons(pred, gt, iou_threshold=0.5)
        assert len(matches) == 1
        # The matched pred should be index 0 (perfect match)
        assert matches[0][0] == 0
        assert len(unmatched_pred) == 1  # pred2 is unmatched (FP)
        assert unmatched_pred[0]   == 1

    def test_multiple_buildings_all_matched(self):
        """
        3 non-overlapping pred/GT pairs should all match perfectly.
        """
        pred = [box(0, 0, 5, 5), box(10, 10, 15, 15), box(20, 20, 25, 25)]
        gt   = [box(0, 0, 5, 5), box(10, 10, 15, 15), box(20, 20, 25, 25)]
        matches, unmatched_pred, unmatched_gt = match_polygons(pred, gt, iou_threshold=0.5)
        assert len(matches)        == 3
        assert len(unmatched_pred) == 0
        assert len(unmatched_gt)   == 0

    def test_empty_pred(self):
        """Empty prediction → 0 matches, all GT unmatched."""
        gt = [box(0, 0, 10, 10)]
        matches, unmatched_pred, unmatched_gt = match_polygons([], gt)
        assert len(matches)        == 0
        assert len(unmatched_pred) == 0
        assert len(unmatched_gt)   == 1

    def test_empty_gt(self):
        """Empty GT → 0 matches, all pred unmatched."""
        pred = [box(0, 0, 10, 10)]
        matches, unmatched_pred, unmatched_gt = match_polygons(pred, [])
        assert len(matches)        == 0
        assert len(unmatched_pred) == 1
        assert len(unmatched_gt)   == 0

    def test_both_empty(self):
        """Both empty → 0 matches, nothing unmatched."""
        matches, unmatched_pred, unmatched_gt = match_polygons([], [])
        assert len(matches)        == 0
        assert len(unmatched_pred) == 0
        assert len(unmatched_gt)   == 0


# ── _size_stratified ──────────────────────────────────────────────────────────

class TestSizeStratified:

    def test_all_detected(self):
        """All GT buildings detected → recall = 1.0 in each stratum."""
        # 1px = 1m at resolution=1.0
        # small: area < 50m², medium: 50-500m², large: >500m²
        small_poly  = box(0, 0, 5, 5)    # area = 25 px → 25 m² at res=1.0 → small
        medium_poly = box(0, 0, 10, 10)  # area = 100 px → 100 m²        → medium
        large_poly  = box(0, 0, 30, 30)  # area = 900 px → 900 m²        → large

        gt_polys         = [small_poly, medium_poly, large_poly]
        matched_gt_idxs  = {0, 1, 2}   # all matched
        resolution       = 1.0

        result = _size_stratified(gt_polys, matched_gt_idxs, resolution)
        for label in ["small", "medium", "large"]:
            assert label in result
            assert result[label]["recall"] == pytest.approx(1.0, abs=1e-4)

    def test_none_detected(self):
        """No GT buildings detected → recall = 0.0 in all strata."""
        small_poly  = box(0, 0, 5, 5)
        medium_poly = box(0, 0, 10, 10)
        gt_polys    = [small_poly, medium_poly]
        matched_gt_idxs = set()   # nothing matched

        result = _size_stratified(gt_polys, matched_gt_idxs, resolution=1.0)
        for label, data in result.items():
            assert data["recall"] == pytest.approx(0.0, abs=1e-4)

    def test_partial_detection_per_stratum(self):
        """
        2 small buildings, 1 detected → small recall = 0.5.
        1 large building, 1 detected → large recall = 1.0.
        """
        small1 = box(0, 0, 5, 5)    # 25 m² → small
        small2 = box(10, 0, 15, 5)  # 25 m² → small
        large1 = box(0, 10, 30, 40) # 900 m² → large

        gt_polys        = [small1, small2, large1]
        matched_gt_idxs = {0, 2}   # small1 and large1 detected, small2 missed

        result = _size_stratified(gt_polys, matched_gt_idxs, resolution=1.0)
        assert result["small"]["recall"] == pytest.approx(0.5, abs=1e-4)
        assert result["large"]["recall"] == pytest.approx(1.0, abs=1e-4)

    def test_resolution_scaling(self):
        """
        At resolution=0.1m/px, a 10×10 pixel polygon = 1m² → classified as small.
        At resolution=1.0m/px, same polygon = 100m² → classified as medium.
        """
        poly = box(0, 0, 10, 10)   # 100 pixels²

        result_fine   = _size_stratified([poly], {0}, resolution=0.1)
        result_coarse = _size_stratified([poly], {0}, resolution=1.0)

        # At 0.1m/px: 100px × 0.01m²/px = 1m² → small
        assert "small" in result_fine
        # At 1.0m/px: 100px × 1.0m²/px = 100m² → medium
        assert "medium" in result_coarse


# ── _aggregate_size_strata ────────────────────────────────────────────────────

class TestAggregateSizeStrata:

    def test_aggregates_counts_correctly(self):
        """
        Two samples each with 2 GT small buildings, 1 detected.
        Aggregated: n_gt=4, n_detected=2, recall=0.5.
        """
        sample = {
            "by_size": {
                "small": {"n_gt": 2, "n_detected": 1, "recall": 0.5}
            }
        }
        result = _aggregate_size_strata([sample, sample])
        assert result["small"]["n_gt"]       == 4
        assert result["small"]["n_detected"] == 2
        assert result["small"]["recall"]     == pytest.approx(0.5, abs=1e-4)