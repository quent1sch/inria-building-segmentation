"""
evaluation/postproc_sensitivity.py

Postprocessing parameter sensitivity analysis.

Sweeps two postprocessing parameters independently and measures their
impact on pixel-level metrics (IoU, Dice) and building-level recall:

  1. simplify_tolerance (Douglas-Peucker epsilon in metres)
     Values: 0.0 (off), 0.3, 0.5, 1.0, 2.0

  2. min_area_m2 (minimum building footprint)
     Values: 0 (off), 5, 10, 25, 50

Outputs
-------
  postproc_sensitivity.png - two line charts (one per parameter)
  postproc_sensitivity.json - full sweep results

This justifies the default parameter choices in config.yaml and reveals
whether postprocessing helps or hurts on this dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SIMPLIFY_VALUES = [0.0, 0.3, 0.5, 1.0, 2.0]
MIN_AREA_VALUES = [0.0, 5.0, 10.0, 25.0, 50.0]


def run(
    samples,
    model,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute raw masks and GT masks once to avoid redundant inference
    print("  Pre-computing predictions...")
    cached = []
    for sample in samples:
        mask, _ = model.predict(
            sample.image,
            input_resolution=sample.resolution,
            resample=True,
        )
        cached.append((mask, sample.gt_mask, sample.resolution))

    print(f"  Sweeping simplify_tolerance over {SIMPLIFY_VALUES}...")
    simplify_results = _sweep_simplify(cached)

    print(f"  Sweeping min_area_m2 over {MIN_AREA_VALUES}...")
    area_results = _sweep_min_area(cached)

    results = {
        "simplify_tolerance": simplify_results,
        "min_area_m2": area_results,
    }

    json_path = out_dir / "postproc_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    _plot(results, out_dir / "postproc_sensitivity.png")
    _print_table(results)

    print(f"  Saved -> {out_dir}/postproc_sensitivity.{{json,png}}")
    return results


def _sweep_simplify(cached: list) -> list[dict]:
    from api.vectorize import vectorize, polygons_to_mask
    from evaluation.metrics_pixel import pixel_metrics, aggregate_metrics

    results = []
    for tol in SIMPLIFY_VALUES:
        sample_metrics = []
        for mask, gt_mask, resolution in cached:
            if tol == 0.0:
                # No simplification - use mask directly as clean
                clean = mask
            else:
                geojson = vectorize(mask, resolution=resolution,
                                    simplify_tolerance_m=tol, min_area_m2=0.0)
                H, W  = mask.shape
                clean = polygons_to_mask(geojson, height=H, width=W)
            sample_metrics.append(pixel_metrics(clean, gt_mask))

        agg = aggregate_metrics(sample_metrics)
        agg["simplify_tolerance_m"] = tol
        results.append(agg)
        print(f"    simplify={tol:.1f}m  IoU={agg['iou']:.4f}  Dice={agg['dice']:.4f}")

    return results


def _sweep_min_area(cached: list) -> list[dict]:
    from api.vectorize import vectorize, polygons_to_mask
    from evaluation.metrics_pixel import pixel_metrics, aggregate_metrics

    results = []
    for min_area in MIN_AREA_VALUES:
        sample_metrics = []
        for mask, gt_mask, resolution in cached:
            if min_area == 0.0:
                clean = mask
            else:
                geojson = vectorize(mask, resolution=resolution,
                                    simplify_tolerance_m=0.0, min_area_m2=min_area)
                H, W  = mask.shape
                clean = polygons_to_mask(geojson, height=H, width=W)
            sample_metrics.append(pixel_metrics(clean, gt_mask))

        agg = aggregate_metrics(sample_metrics)
        agg["min_area_m2"] = min_area
        results.append(agg)
        print(f"    min_area={min_area:.0f}m²  IoU={agg['iou']:.4f}  Dice={agg['dice']:.4f}")

    return results


def _print_table(results: dict) -> None:
    sep = "-" * 50
    print("\n  Simplify tolerance sweep:")
    print(sep)
    print(f"  {'Tolerance (m)':<15} {'IoU':>8} {'Dice':>8} {'F1':>8}")
    print(sep)
    for r in results["simplify_tolerance"]:
        print(f"  {r['simplify_tolerance_m']:<15.1f} "
              f"{r['iou']:>8.4f} {r['dice']:>8.4f} {r['f1']:>8.4f}")

    print("\n  Min area sweep:")
    print(sep)
    print(f"  {'Min area (m²)':<15} {'IoU':>8} {'Dice':>8} {'F1':>8}")
    print(sep)
    for r in results["min_area_m2"]:
        print(f"  {r['min_area_m2']:<15.0f} "
              f"{r['iou']:>8.4f} {r['dice']:>8.4f} {r['f1']:>8.4f}")


def _plot(results: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available - skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── simplify tolerance ────────────────────────────────────────────────
    sr = results["simplify_tolerance"]
    xs = [r["simplify_tolerance_m"] for r in sr]
    ax1.plot(xs, [r["iou"]  for r in sr], "b-o", label="IoU")
    ax1.plot(xs, [r["dice"] for r in sr], "r--o", label="Dice")
    ax1.set_xlabel("Douglas-Peucker tolerance (m)")
    ax1.set_ylabel("Score")
    ax1.set_title("Simplification Tolerance Sensitivity")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ── min area ─────────────────────────────────────────────────────────
    ar = results["min_area_m2"]
    xs = [r["min_area_m2"] for r in ar]
    ax2.plot(xs, [r["iou"]  for r in ar], "b-o", label="IoU")
    ax2.plot(xs, [r["dice"] for r in ar], "r--o", label="Dice")
    ax2.set_xlabel("Minimum building area (m2)")
    ax2.set_ylabel("Score")
    ax2.set_title("Min Area Filter Sensitivity")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle("Postprocessing Parameter Sensitivity", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()