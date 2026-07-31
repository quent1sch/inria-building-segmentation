"""
evaluation/postproc_sensitivity.py
 
Postprocessing parameter sensitivity analysis.
 
Sweeps two postprocessing parameters independently:
 
  1. simplify_tolerance (Douglas-Peucker epsilon in metres)
     Values: 0.0 (off), 0.3, 0.5, 1.0, 2.0
 
  2. min_area_m2 (minimum building footprint)
     Values: 0 (off), 5, 10, 25, 50
 
Primary metrics: building-level (detection recall, precision, F1, mean matched IoU)
  These answer what we actually care about: does postprocessing help or hurt
  building detection? Pixel metrics are inappropriate as primary here because
  the model is trained with a pixel-level loss - postprocessing will almost
  always slightly reduce pixel IoU/Dice by construction (straight-edge fitting
  introduces small boundary distortions). That expected drop is not meaningful.
 
Secondary metric: pixel IoU (one column, sanity check only)
  A large pixel IoU drop would indicate something is wrong (e.g. simplification
  merging large building groups into single polygons). A small drop is expected
  and not a concern.
 
Outputs
-------
  postproc_sensitivity.png  — two plots (simplify sweep + min_area sweep)
                               each showing building recall/F1/mIoU + pixel IoU
  postproc_sensitivity.json  — full sweep results
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

SIMPLIFY_VALUES = [0.0, 0.3, 0.5, 1.0, 2.0]
MIN_AREA_VALUES = [0.0, 5.0, 10.0, 25.0, 50.0]
IOU_THRESHOLD   = 0.5


def run(
    samples,
    model,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
 
    # Pre-compute raw masks and GT once — inference is the expensive step
    print("  Pre-computing predictions...")
    cached = []
    for sample in samples:
        mask, _ = model.predict(
            sample.image,
            input_resolution=sample.resolution,
            resample=True,
        )
        cached.append((mask, sample.gt_mask, sample.resolution))
    print(f"  {len(cached)} samples cached.")
 
    print(f"\n  Sweeping simplify_tolerance: {SIMPLIFY_VALUES}")
    simplify_results = _sweep_simplify(cached)
 
    print(f"\n  Sweeping min_area_m2: {MIN_AREA_VALUES}")
    area_results = _sweep_min_area(cached)
 
    results = {
        "primary_metric": "building-level (recall, precision, F1, mean_iou_matched)",
        "secondary_metric": "pixel IoU (sanity check — expected small drop with postprocessing)",
        "simplify_tolerance": simplify_results,
        "min_area_m2": area_results,
    }
 
    json_path = out_dir / "postproc_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
 
    _plot(results, out_dir / "postproc_sensitivity.png")
    _print_table(results)
    print(f"\n  Saved → {out_dir}/postproc_sensitivity.{{json,png}}")
    return results


# ── sweeps ────────────────────────────────────────────────────────────────────

def _sweep_simplify(cached: list) -> list[dict]:
    from api.vectorize import vectorize, polygons_to_mask
    from evaluation.metrics_pixel import pixel_metrics, aggregate_metrics
    from evaluation.metrics_building import (
        _geojson_to_polygons, _compute_detection_metrics
    )
 
    results = []
    for tol in SIMPLIFY_VALUES:
        pixel_ms    = []
        building_ms = []
 
        for mask, gt_mask, resolution in cached:
            # Pixel metrics
            if tol == 0.0:
                clean = mask
            else:
                geojson = vectorize(mask, resolution=resolution,
                                    simplify_tolerance_m=tol, min_area_m2=0.0)
                H, W  = mask.shape
                clean = polygons_to_mask(geojson, height=H, width=W)
            pixel_ms.append(pixel_metrics(clean, gt_mask))
 
            # Building metrics — vectorize both pred and GT
            pred_geojson = vectorize(mask, resolution=resolution,
                                     simplify_tolerance_m=tol, min_area_m2=0.0)
            gt_geojson = vectorize(gt_mask,resolution=resolution,
                                     simplify_tolerance_m=tol, min_area_m2=0.0)
            pred_polys = _geojson_to_polygons(pred_geojson)
            gt_polys = _geojson_to_polygons(gt_geojson)
            bm = _compute_detection_metrics(pred_polys, gt_polys, resolution, IOU_THRESHOLD)
            building_ms.append(bm)
 
        pix_agg = aggregate_metrics(pixel_ms)
        bld_keys = ["precision", "recall", "f1", "mean_iou_matched"]
        bld_agg = {k: float(np.mean([m[k] for m in building_ms])) for k in bld_keys}
 
        row = {
            "simplify_tolerance_m": tol,
            "building_recall": bld_agg["recall"],
            "building_precision": bld_agg["precision"],
            "building_f1": bld_agg["f1"],
            "building_mean_iou": bld_agg["mean_iou_matched"],
            "pixel_iou": pix_agg["iou"], # secondary / sanity check
        }
        results.append(row)
        print(
            f"    simplify={tol:.1f}m  "
            f"bld_recall={row['building_recall']:.4f}  "
            f"bld_F1={row['building_f1']:.4f}  "
            f"pix_IoU={row['pixel_iou']:.4f}"
        )
 
    return results
 
 
def _sweep_min_area(cached: list) -> list[dict]:
    from api.vectorize import vectorize, polygons_to_mask
    from evaluation.metrics_pixel import pixel_metrics, aggregate_metrics
    from evaluation.metrics_building import (
        _geojson_to_polygons, _compute_detection_metrics
    )
 
    results = []
    for min_area in MIN_AREA_VALUES:
        pixel_ms = []
        building_ms = []
 
        for mask, gt_mask, resolution in cached:
            # Pixel metrics
            if min_area == 0.0:
                clean = mask
            else:
                geojson = vectorize(mask, resolution=resolution,
                                    simplify_tolerance_m=0.0, min_area_m2=min_area)
                H, W  = mask.shape
                clean = polygons_to_mask(geojson, height=H, width=W)
            pixel_ms.append(pixel_metrics(clean, gt_mask))
 
            # Building metrics
            pred_geojson = vectorize(mask, resolution=resolution,
                                     simplify_tolerance_m=0.0, min_area_m2=min_area)
            gt_geojson = vectorize(gt_mask, resolution=resolution,
                                     simplify_tolerance_m=0.0, min_area_m2=min_area)
            pred_polys = _geojson_to_polygons(pred_geojson)
            gt_polys = _geojson_to_polygons(gt_geojson)
            bm = _compute_detection_metrics(pred_polys, gt_polys, resolution, IOU_THRESHOLD)
            building_ms.append(bm)
 
        pix_agg  = aggregate_metrics(pixel_ms)
        bld_keys = ["precision", "recall", "f1", "mean_iou_matched"]
        bld_agg  = {k: float(np.mean([m[k] for m in building_ms])) for k in bld_keys}
 
        row = {
            "min_area_m2": min_area,
            "building_recall": bld_agg["recall"],
            "building_precision": bld_agg["precision"],
            "building_f1": bld_agg["f1"],
            "building_mean_iou": bld_agg["mean_iou_matched"],
            "pixel_iou": pix_agg["iou"],
        }
        results.append(row)
        print(
            f"    min_area={min_area:.0f}m²  "
            f"bld_recall={row['building_recall']:.4f}  "
            f"bld_F1={row['building_f1']:.4f}  "
            f"pix_IoU={row['pixel_iou']:.4f}"
        )
 
    return results
 
 
# ── display ───────────────────────────────────────────────────────────────────
 
def _print_table(results: dict) -> None:
    sep = "-" * 70
    hdr = f"  {'Param':<18} {'bld_Recall':>10} {'bld_Prec':>10} {'bld_F1':>8} {'bld_mIoU':>10} {'pix_IoU*':>9}"
 
    print(f"\n  Simplify tolerance sweep (* pixel IoU: secondary/sanity check)")
    print(sep)
    print(hdr)
    print(sep)
    for r in results["simplify_tolerance"]:
        print(
            f"  {r['simplify_tolerance_m']:.1f}m{'':<15}"
            f"{r['building_recall']:>10.4f} {r['building_precision']:>10.4f} "
            f"{r['building_f1']:>8.4f} {r['building_mean_iou']:>10.4f} "
            f"{r['pixel_iou']:>9.4f}"
        )
 
    print(f"\n  Min area sweep (* pixel IoU: secondary/sanity check)")
    print(sep)
    print(hdr)
    print(sep)
    for r in results["min_area_m2"]:
        print(
            f"  {r['min_area_m2']:.0f}m²{'':<15}"
            f"{r['building_recall']:>10.4f} {r['building_precision']:>10.4f} "
            f"{r['building_f1']:>8.4f} {r['building_mean_iou']:>10.4f} "
            f"{r['pixel_iou']:>9.4f}"
        )
    print(sep)
 
 
def _plot(results: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available - skipping plot.")
        return
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 
    def _plot_sweep(ax_main, ax_secondary, rows, x_key, x_label):
        xs = [r[x_key] for r in rows]
        recalls = [r["building_recall"] for r in rows]
        f1s = [r["building_f1"] for r in rows]
        mean_ious = [r["building_mean_iou"] for r in rows]
        pixel_ious = [r["pixel_iou"] for r in rows]
 
        # Primary: building metrics
        ax_main.plot(xs, recalls, "b-o", lw=2, label="Building Recall")
        ax_main.plot(xs, f1s, "g-o", lw=2, label="Building F1")
        ax_main.plot(xs, mean_ious, "r--o", lw=2, label="Mean matched IoU")
        ax_main.set_xlabel(x_label)
        ax_main.set_ylabel("Score")
        ax_main.set_ylim([0, 1])
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(fontsize=9)
 
        # Secondary: pixel IoU sanity check
        ax_secondary.plot(xs, pixel_ious, "k-o", lw=2, label="Pixel IoU")
        ax_secondary.set_xlabel(x_label)
        ax_secondary.set_ylabel("Pixel IoU")
        ax_secondary.set_ylim([0, 1])
        ax_secondary.grid(True, alpha=0.3)
        ax_secondary.legend(fontsize=9)
        ax_secondary.set_title("Pixel IoU (sanity check - small drop expected)", fontsize=9)
 
    _plot_sweep(
        axes[0, 0], axes[0, 1],
        results["simplify_tolerance"],
        "simplify_tolerance_m",
        "Douglas-Peucker tolerance (m)",
    )
    axes[0, 0].set_title("Simplify Tolerance - Building Metrics (primary)")
 
    _plot_sweep(
        axes[1, 0], axes[1, 1],
        results["min_area_m2"],
        "min_area_m2",
        "Minimum building area (m2)",
    )
    axes[1, 0].set_title("Min Area Filter - Building Metrics (primary)")
 
    plt.suptitle(
        "Postprocessing Parameter Sensitivity\n"
        "Primary: building-level | Secondary: pixel IoU",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()