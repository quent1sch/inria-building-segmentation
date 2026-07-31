"""
evaluation/metrics_building.py

Object-level (building-level) evaluation.

Pixel metrics miss a key user concern: whether individual buildings are
detected at all. A model that correctly segments 95% of pixels in a large
building but completely misses 10 small buildings looks good at pixel level
but fails in practice.

Approach
--------
  1. Vectorize both prediction and GT masks into polygons
  2. Match predicted polygons to GT polygons by IoU overlap (COCO-style)
  3. A predicted building is a TP if it overlaps a GT building by IoU >= iou_threshold
  4. Compute detection Precision, Recall, F1 and mean matched IoU

Matching uses a greedy algorithm (sort by IoU, match highest first) which
is standard for building detection. For large sets, a Shapely STRtree is
used for spatial indexing to avoid O(n x m) brute-force comparison.

Size-stratified metrics
------------------------
Buildings are stratified by real-world area into three bins:
  small  : < 50 m²   (sheds, garages)
  medium : 50-500 m²  (typical residential buildings)
  large  : > 500 m²   (commercial / industrial)

These require resolution to be known. If unknown, stratification is skipped.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


# ── polygon IoU ───────────────────────────────────────────────────────────────

def polygon_iou(p1, p2) -> float:
    """Compute IoU between two shapely polygons."""
    try:
        intersection = p1.intersection(p2).area
        union = p1.union(p2).area
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0


# ── greedy matching ───────────────────────────────────────────────────────────

def match_polygons(
    pred_polys: list,
    gt_polys: list,
    iou_threshold: float = 0.5,
) -> tuple[list[tuple], list[int], list[int]]:
    """
    Greedily match predicted polygons to GT polygons by IoU.

    Uses a Shapely STRtree for spatial indexing - only candidate pairs that
    actually overlap are scored, making this efficient for large scenes.

    Returns
    -------
    matches: list of (pred_idx, gt_idx, iou) for matched pairs
    unmatched_pred: indices of unmatched predicted polygons (false positives)
    unmatched_gt: indices of unmatched GT polygons (false negatives / misses)
    """
    from shapely.strtree import STRtree

    if not pred_polys or not gt_polys:
        return [], list(range(len(pred_polys))), list(range(len(gt_polys)))

    gt_tree = STRtree(gt_polys)
    matched_gt = set()
    matched_pred = set()
    candidates = [] # (iou, pred_idx, gt_idx)

    for p_idx, pred in enumerate(pred_polys):
        # Only query GT polygons whose bounding box intersects
        candidate_gt_idxs = gt_tree.query(pred)
        for g_idx in candidate_gt_idxs:
            iou = polygon_iou(pred, gt_polys[g_idx])
            if iou >= iou_threshold:
                candidates.append((iou, p_idx, g_idx))

    # Sort descending by IoU, greedily assign
    candidates.sort(key=lambda x: -x[0])
    matches = []
    for iou, p_idx, g_idx in candidates:
        if p_idx not in matched_pred and g_idx not in matched_gt:
            matches.append((p_idx, g_idx, iou))
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)

    unmatched_pred = [i for i in range(len(pred_polys)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt_polys)) if i not in matched_gt]

    return matches, unmatched_pred, unmatched_gt


# ── size stratification ───────────────────────────────────────────────────────

SIZE_BINS = {
    "small": (0,   50),
    "medium": (50,  500),
    "large": (500, float("inf")),
}


def get_size_label(area_m2: float) -> str:
    for label, (lo, hi) in SIZE_BINS.items():
        if lo <= area_m2 < hi:
            return label
    return "large"


# ── per-sample building metrics ───────────────────────────────────────────────

def building_metrics_sample(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    resolution: Optional[float],
    iou_threshold: float = 0.5,
    simplify_tolerance_m: float = 0.5,
    min_area_m2: float = 10.0,
) -> dict:
    """
    Compute building-level metrics for a single image.

    Vectorizes both masks, matches polygons, returns detection metrics.
    """
    from api.vectorize import vectorize

    pred_geojson = vectorize(pred_mask, resolution=resolution,
                             simplify_tolerance_m=simplify_tolerance_m,
                             min_area_m2=min_area_m2)
    gt_geojson   = vectorize(gt_mask, resolution=resolution,
                             simplify_tolerance_m=simplify_tolerance_m,
                             min_area_m2=min_area_m2)

    pred_polys = _geojson_to_polygons(pred_geojson)
    gt_polys   = _geojson_to_polygons(gt_geojson)

    matches, unmatched_pred, unmatched_gt = match_polygons(
        pred_polys, gt_polys, iou_threshold=iou_threshold
    )

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    smooth = 1e-6
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0
    miss_rate = fn / (tp + fn + smooth)
    false_alarm = fp / (tp + fp + smooth)

    result = {
        "n_pred": len(pred_polys),
        "n_gt": len(gt_polys),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou_matched": float(mean_iou),
        "miss_rate": float(miss_rate),
        "false_alarm_rate": float(false_alarm),
    }

    # Size-stratified metrics (requires resolution)
    if resolution is not None:
        result["by_size"] = _size_stratified(
            pred_polys, gt_polys, matches, unmatched_pred, unmatched_gt, resolution
        )

    return result


def _geojson_to_polygons(geojson: dict) -> list:
    """Extract shapely Polygon objects from a GeoJSON FeatureCollection."""
    from shapely.geometry import shape
    polys = []
    for feature in geojson.get("features", []):
        try:
            polys.append(shape(feature["geometry"]))
        except Exception:
            continue
    return polys


def _size_stratified(
    pred_polys, gt_polys, matches, unmatched_pred, unmatched_gt, resolution
) -> dict:
    """Compute detection recall per building size stratum."""
    matched_gt_idxs = {g_idx for _, g_idx, _ in matches}
    results = {}

    for label, (lo, hi) in SIZE_BINS.items():
        # GT buildings in this stratum
        gt_in_bin = [
            i for i, p in enumerate(gt_polys)
            if lo <= p.area * resolution ** 2 < hi
        ]
        if not gt_in_bin:
            continue
        detected = sum(1 for i in gt_in_bin if i in matched_gt_idxs)
        results[label] = {
            "n_gt": len(gt_in_bin),
            "n_detected": detected,
            "recall": detected / len(gt_in_bin),
        }

    return results


# ── evaluation runner ─────────────────────────────────────────────────────────

def run(
    samples,
    model,
    out_dir: Path,
    iou_threshold: float = 0.5,
    simplify_tolerance_m: float = 0.5,
    min_area_m2: float = 10.0,
) -> dict:
    """
    Run building-level evaluation over all samples.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    city_results: dict[str, list] = defaultdict(list)

    for i, sample in enumerate(samples):
        print(f"  [{i+1}] {sample.name} ({sample.city})", end=" ", flush=True)

        mask, _ = model.predict(
            sample.image,
            input_resolution=sample.resolution,
            resample=True,
        )

        m = building_metrics_sample(
            mask, sample.gt_mask,
            resolution=sample.resolution,
            iou_threshold=iou_threshold,
            simplify_tolerance_m=simplify_tolerance_m,
            min_area_m2=min_area_m2,
        )
        city_results[sample.city].append(m)
        print(f"Precision={m['precision']:.3f} Recall={m['recall']:.3f} F1={m['f1']:.3f}")

    # ── aggregate ─────────────────────────────────────────────────────────
    scalar_keys = ["precision", "recall", "f1", "mean_iou_matched",
                   "miss_rate", "false_alarm_rate"]

    results: dict = {"per_city": {}, "overall": {}, "iou_threshold": iou_threshold}
    all_samples = []

    for city, samples_list in city_results.items():
        agg = {k: float(np.mean([s[k] for s in samples_list])) for k in scalar_keys}
        agg["n_samples"] = len(samples_list)
        agg["total_pred"] = sum(s["n_pred"] for s in samples_list)
        agg["total_gt"] = sum(s["n_gt"] for s in samples_list)
        agg["total_tp"] = sum(s["tp"] for s in samples_list)
        agg["total_fp"] = sum(s["fp"] for s in samples_list)
        agg["total_fn"] = sum(s["fn"] for s in samples_list)

        # Aggregate size-stratified if available
        if any("by_size" in s for s in samples_list):
            agg["by_size"] = _aggregate_size_strata(samples_list)

        results["per_city"][city] = agg
        all_samples.extend(samples_list)

    if all_samples:
        agg_all = {k: float(np.mean([s[k] for s in all_samples])) for k in scalar_keys}
        agg_all["total_tp"] = sum(s["tp"] for s in all_samples)
        agg_all["total_fp"] = sum(s["fp"] for s in all_samples)
        agg_all["total_fn"] = sum(s["fn"] for s in all_samples)
        if any("by_size" in s for s in all_samples):
            agg_all["by_size"] = _aggregate_size_strata(all_samples)
        results["overall"] = agg_all

    # ── save and print ─────────────────────────────────────────────────────
    json_path = out_dir / "metrics_building.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_table(results)
    print(f"\n  Saved -> {json_path}")
    return results


def _aggregate_size_strata(samples_list: list) -> dict:
    strata: dict[str, dict] = defaultdict(lambda: {"n_gt": 0, "n_detected": 0})
    for s in samples_list:
        for label, data in s.get("by_size", {}).items():
            strata[label]["n_gt"] += data["n_gt"]
            strata[label]["n_detected"] += data["n_detected"]
    return {
        label: {
            **data,
            "recall": data["n_detected"] / data["n_gt"] if data["n_gt"] > 0 else 0.0
        }
        for label, data in strata.items()
    }


def _print_table(results: dict) -> None:
    sep = "-" * 80
    print(f"\n{'Building-level metrics (IoU threshold='}{results['iou_threshold']}){'':^20}")
    print(sep)
    print(f"{'City':<14} {'N':>5}  {'Prec':>7} {'Recall':>7} {'F1':>7}  "
          f"{'mIoU':>7} {'Miss%':>7} {'FAlarm%':>8}")
    print(sep)

    for city, data in results["per_city"].items():
        print(f"{city:<14} {data['n_samples']:>5}  "
              f"{data['precision']:>7.3f} {data['recall']:>7.3f} {data['f1']:>7.3f}  "
              f"{data['mean_iou_matched']:>7.3f} "
              f"{100*data['miss_rate']:>6.1f}% "
              f"{100*data['false_alarm_rate']:>7.1f}%")

    if results.get("overall"):
        ov = results["overall"]
        print(sep)
        print(f"{'OVERALL':<14} {'':>5}  "
              f"{ov['precision']:>7.3f} {ov['recall']:>7.3f} {ov['f1']:>7.3f}  "
              f"{ov['mean_iou_matched']:>7.3f} "
              f"{100*ov['miss_rate']:>6.1f}% "
              f"{100*ov['false_alarm_rate']:>7.1f}%")

        if "by_size" in ov:
            print("\n  Size-stratified recall:")
            for label, data in ov["by_size"].items():
                print(f"    {label:<8}: {data['recall']:.3f}  "
                      f"({data['n_detected']}/{data['n_gt']} buildings detected)")
    print(sep)