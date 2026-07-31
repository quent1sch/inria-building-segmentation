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
 
Both conditions are evaluated and compared:
  raw_vectorized  — polygonize raw mask, area filter only (no simplification)
                    reflects pure model detection quality
  clean  — full pipeline (simplify + area filter)
           reveals whether simplification merges/splits buildings
 
The comparison between raw_vectorized and clean is informative:
  - Detection recall/precision can genuinely differ if simplification merges
    adjacent blobs or if simplified polygons shrink below the area threshold
  - Mean matched IoU measures boundary quality improvement from simplification
 
Matching uses a greedy algorithm (sort by IoU, match highest first).
For large sets, a Shapely STRtree is used for spatial indexing.
 
Size-stratified metrics
------------------------
Buildings are stratified by real-world area into three bins:
  small  : < 50 m2 (sheds, garages)
  medium : 50-500 m2 (typical residential)
  large  : > 500 m2 (commercial / industrial)
Requires resolution to be known.
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


# ── core per-sample metrics ───────────────────────────────────────────────────
 
def _compute_detection_metrics(
    pred_polys: list,
    gt_polys: list,
    resolution: Optional[float],
    iou_threshold: float,
) -> dict:
    """Compute detection metrics given two polygon lists."""
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
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou_matched": float(mean_iou),
        "miss_rate": float(miss_rate),
        "false_alarm_rate": float(false_alarm),
    }
 
    if resolution is not None:
        matched_gt_idxs = {g_idx for _, g_idx, _ in matches}
        result["by_size"] = _size_stratified(gt_polys, matched_gt_idxs, resolution)
 
    return result

def building_metrics_sample(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    resolution: Optional[float],
    iou_threshold: float = 0.5,
    simplify_tolerance_m: float = 0.5,
    min_area_m2: float = 10.0,
) -> dict:
    """
    Compute building-level metrics for both raw_vectorized and clean conditions.
 
    raw_vectorized : polygonize + area filter only (no simplification)
    clean          : polygonize + simplify + area filter
    """
    from api.vectorize import vectorize
 
    # GT always uses full postprocessing for clean polygon boundaries
    gt_geojson = vectorize(
        gt_mask, resolution=resolution,
        simplify_tolerance_m=simplify_tolerance_m,
        min_area_m2=min_area_m2,
    )
    gt_polys = _geojson_to_polygons(gt_geojson)
 
    # ── raw vectorized: area filter only, no simplification ──────────────
    pred_raw_geojson = vectorize(
        pred_mask, resolution=resolution,
        simplify_tolerance_m=0.0,
        min_area_m2=min_area_m2,
    )
    pred_raw_polys = _geojson_to_polygons(pred_raw_geojson)
    raw_metrics    = _compute_detection_metrics(
        pred_raw_polys, gt_polys, resolution, iou_threshold
    )
 
    # ── clean: full postprocessing ────────────────────────────────────────
    pred_clean_geojson = vectorize(
        pred_mask, resolution=resolution,
        simplify_tolerance_m=simplify_tolerance_m,
        min_area_m2=min_area_m2,
    )
    pred_clean_polys = _geojson_to_polygons(pred_clean_geojson)
    clean_metrics    = _compute_detection_metrics(
        pred_clean_polys, gt_polys, resolution, iou_threshold
    )
 
    return {"raw": raw_metrics, "clean": clean_metrics}


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


def _size_stratified(gt_polys, matched_gt_idxs: set, resolution: float) -> dict:
    results = {}
    for label, (lo, hi) in SIZE_BINS.items():
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


# ── aggregation ───────────────────────────────────────────────────────────────
 
SCALAR_KEYS = [
    "precision", "recall", "f1",
    "mean_iou_matched", "miss_rate", "false_alarm_rate",
]
 
 
def _aggregate_condition(samples_list: list, condition: str) -> dict:
    """Aggregate scalar metrics for one condition (raw or clean)."""
    cond_samples = [s[condition] for s in samples_list if condition in s]
    if not cond_samples:
        return {}
    agg = {k: float(np.mean([s[k] for s in cond_samples])) for k in SCALAR_KEYS}
    agg["n_samples"] = len(cond_samples)
    agg["total_tp"]  = sum(s["tp"] for s in cond_samples)
    agg["total_fp"]  = sum(s["fp"] for s in cond_samples)
    agg["total_fn"]  = sum(s["fn"] for s in cond_samples)
    if any("by_size" in s for s in cond_samples):
        agg["by_size"] = _aggregate_size_strata(cond_samples)
    return agg
 
 
def _aggregate_size_strata(samples_list: list) -> dict:
    strata: dict = defaultdict(lambda: {"n_gt": 0, "n_detected": 0})
    for s in samples_list:
        for label, data in s.get("by_size", {}).items():
            strata[label]["n_gt"]       += data["n_gt"]
            strata[label]["n_detected"] += data["n_detected"]
    return {
        label: {
            **data,
            "recall": data["n_detected"] / data["n_gt"] if data["n_gt"] > 0 else 0.0,
        }
        for label, data in strata.items()
    }


# ── evaluation runner ─────────────────────────────────────────────────────────

def run(
    samples,
    model,
    out_dir: Path,
    iou_threshold: float = 0.5,
    simplify_tolerance_m: float = 0.5,
    min_area_m2: float = 10.0,
) -> dict:
    """Run building-level evaluation (both raw and clean) over all samples."""
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
        print(
            f"raw  P={m['raw']['precision']:.3f} R={m['raw']['recall']:.3f} "
            f"F1={m['raw']['f1']:.3f} | "
            f"clean P={m['clean']['precision']:.3f} R={m['clean']['recall']:.3f} "
            f"F1={m['clean']['f1']:.3f}"
        )
 
    # ── aggregate ─────────────────────────────────────────────────────────
    results: dict = {
        "per_city": {},
        "overall": {},
        "iou_threshold": iou_threshold,
    }
    all_samples = []
 
    for city, samples_list in sorted(city_results.items()):
        results["per_city"][city] = {
            "raw": _aggregate_condition(samples_list, "raw"),
            "clean": _aggregate_condition(samples_list, "clean"),
        }
        all_samples.extend(samples_list)
 
    if all_samples:
        results["overall"] = {
            "raw": _aggregate_condition(all_samples, "raw"),
            "clean": _aggregate_condition(all_samples, "clean"),
        }
 
    # ── save ──────────────────────────────────────────────────────────────
    json_path = out_dir / "metrics_building.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
 
    _print_table(results)
    print(f"\n  Saved -> {json_path}")
    return results


def _print_table(results: dict) -> None:
    sep = "-" * 90
    thr = results["iou_threshold"]
    print(f"\n{'Building-level metrics':^90}  (IoU threshold={thr})")
    print(sep)
    print(
        f"{'City':<14}  "
        f"{'── raw vectorized ──':^32}  "
        f"{'── clean (simplified) ──':^34}"
    )
    print(
        f"{'':14}  "
        f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'mIoU':>7}  "
        f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'mIoU':>7}  "
        f"{'ΔRecall':>8}"
    )
    print(sep)
 
    def _row(name, raw, clean):
        delta = clean["recall"] - raw["recall"]
        return (
            f"{name:<14}  "
            f"{raw['precision']:>7.3f} {raw['recall']:>7.3f} "
            f"{raw['f1']:>7.3f} {raw['mean_iou_matched']:>7.3f}  "
            f"{clean['precision']:>7.3f} {clean['recall']:>7.3f} "
            f"{clean['f1']:>7.3f} {clean['mean_iou_matched']:>7.3f}  "
            f"{delta:>+8.3f}"
        )
 
    for city, data in results["per_city"].items():
        if data["raw"] and data["clean"]:
            print(_row(city, data["raw"], data["clean"]))
 
    if results.get("overall") and results["overall"].get("raw"):
        print(sep)
        print(_row("OVERALL", results["overall"]["raw"], results["overall"]["clean"]))
 
        ov = results["overall"]
        if "by_size" in ov.get("raw", {}):
            print("\n  Size-stratified recall (raw -> clean):")
            for label in SIZE_BINS:
                r_data = ov["raw"].get("by_size", {}).get(label, {})
                c_data = ov["clean"].get("by_size", {}).get(label, {})
                if r_data:
                    print(
                        f"    {label:<8}: "
                        f"{r_data.get('recall', 0):.3f} -> {c_data.get('recall', 0):.3f}  "
                        f"({r_data.get('n_detected',0)}/{r_data.get('n_gt',0)} -> "
                        f"{c_data.get('n_detected',0)}/{c_data.get('n_gt',0)})"
                    )
    print(sep)