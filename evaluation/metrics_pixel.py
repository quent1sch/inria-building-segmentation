"""
evaluation/metrics_pixel.py

Pixel-level segmentation metrics: IoU, Dice, Precision, Recall, F1.

Computes metrics for:
  - Raw model predictions (direct threshold on prob map)
  - Clean predictions (vectorized → rasterized)

Results are aggregated per-city and overall, and saved as JSON + CSV.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


# ── core metric computation ───────────────────────────────────────────────────

def pixel_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
) -> dict[str, float]:
    """
    Compute pixel-level metrics for a single prediction/GT pair.

    Parameters
    ----------
    pred: (H, W) bool or uint8
    gt: (H, W) bool or uint8

    Returns
    -------
    dict with keys: iou, dice, precision, recall, f1
    """
    pred = pred.astype(bool).ravel()
    gt   = gt.astype(bool).ravel()

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    # tn = (~pred & ~gt).sum()  # not needed for these metrics

    smooth = 1e-6
    iou       = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    f1        = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)  # = Dice
    dice      = f1

    return {
        "iou": float(iou),
        "dice": float(dice),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def aggregate_metrics(metrics_list: list[dict]) -> dict[str, float]:
    """Average a list of per-sample metric dicts."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


# ── evaluation runner ─────────────────────────────────────────────────────────

def run(
    samples, # iterable of EvalSample
    model, # SegmentationInference instance
    out_dir: Path,
    postprocess: bool = True,
    simplify_tolerance_m: float = 0.5,
    min_area_m2: float = 10.0,
) -> dict:
    """
    Run pixel-level evaluation over all samples.

    For each sample computes metrics for:
      - raw prediction
      - clean prediction (postprocessed), if postprocess=True

    Results are grouped by city.

    Parameters
    ----------
    samples: iterable of EvalSample (from ground_truth.py)
    model: loaded SegmentationInference
    out_dir: directory to save results
    postprocess: also compute metrics for clean (vectorized) prediction
    """
    from api.vectorize import vectorize
    from api.vectorize import polygons_to_mask

    out_dir.mkdir(parents=True, exist_ok=True)

    # city -> list of per-sample metric dicts
    city_raw: dict[str, list] = defaultdict(list)
    city_clean: dict[str, list] = defaultdict(list)

    for sample in samples:
        mask, info = model.predict(
            sample.image,
            input_resolution=sample.resolution,
            resample=True,
        )

        m_raw = pixel_metrics(mask, sample.gt_mask)
        city_raw[sample.city].append(m_raw)

        if postprocess:
            geojson = vectorize(
                mask,
                resolution=sample.resolution,
                simplify_tolerance_m=simplify_tolerance_m,
                min_area_m2=min_area_m2,
            )
            H, W = sample.image.shape[:2]
            clean_mask = polygons_to_mask(geojson, height=H, width=W)
            m_clean = pixel_metrics(clean_mask, sample.gt_mask)
            city_clean[sample.city].append(m_clean)

    # ── aggregate ─────────────────────────────────────────────────────────
    results: dict = {"per_city": {}, "overall": {}}
    all_raw, all_clean = [], []

    for city in sorted(city_raw.keys()):
        agg_raw = aggregate_metrics(city_raw[city])
        results["per_city"][city] = {"raw": agg_raw, "n_samples": len(city_raw[city])}
        all_raw.extend(city_raw[city])

        if postprocess and city in city_clean:
            agg_clean = aggregate_metrics(city_clean[city])
            results["per_city"][city]["clean"] = agg_clean
            all_clean.extend(city_clean[city])

    results["overall"]["raw"] = aggregate_metrics(all_raw)
    if postprocess and all_clean:
        results["overall"]["clean"] = aggregate_metrics(all_clean)

    # ── save ──────────────────────────────────────────────────────────────
    json_path = out_dir / "metrics_pixel.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    _save_csv(results, out_dir / "metrics_pixel.csv", postprocess)

    _print_table(results, postprocess)

    print(f"\n  Saved -> {json_path}")
    return results


def _save_csv(results: dict, path: Path, postprocess: bool) -> None:
    import csv
    fieldnames = ["city", "n_samples",
                  "iou_raw", "dice_raw", "precision_raw", "recall_raw"]
    if postprocess:
        fieldnames += ["iou_clean", "dice_clean", "precision_clean", "recall_clean",
                       "iou_delta", "dice_delta"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for city, data in results["per_city"].items():
            row: dict = {
                "city": city,
                "n_samples": data["n_samples"],
                "iou_raw": round(data["raw"]["iou"], 4),
                "dice_raw": round(data["raw"]["dice"], 4),
                "precision_raw": round(data["raw"]["precision"], 4),
                "recall_raw": round(data["raw"]["recall"], 4),
            }
            if postprocess and "clean" in data:
                row["iou_clean"] = round(data["clean"]["iou"], 4)
                row["dice_clean"] = round(data["clean"]["dice"], 4)
                row["precision_clean"] = round(data["clean"]["precision"], 4)
                row["recall_clean"] = round(data["clean"]["recall"], 4)
                row["iou_delta"] = round(data["clean"]["iou"] - data["raw"]["iou"], 4)
                row["dice_delta"] = round(data["clean"]["dice"] - data["raw"]["dice"], 4)
            writer.writerow(row)

        # Overall row
        ov = results["overall"]
        row = {
            "city": "OVERALL",
            "n_samples": "",
            "iou_raw": round(ov["raw"]["iou"], 4),
            "dice_raw": round(ov["raw"]["dice"], 4),
            "precision_raw": round(ov["raw"]["precision"], 4),
            "recall_raw": round(ov["raw"]["recall"], 4),
        }
        if postprocess and "clean" in ov:
            row["iou_clean"] = round(ov["clean"]["iou"], 4)
            row["dice_clean"] = round(ov["clean"]["dice"], 4)
            row["precision_clean"] = round(ov["clean"]["precision"], 4)
            row["recall_clean"] = round(ov["clean"]["recall"], 4)
            row["iou_delta"] = round(ov["clean"]["iou"] - ov["raw"]["iou"], 4)
            row["dice_delta"] = round(ov["clean"]["dice"] - ov["raw"]["dice"], 4)
        writer.writerow(row)


def _print_table(results: dict, postprocess: bool) -> None:
    sep = "-" * 80
    print(f"\n{'Pixel-level metrics':^80}")
    print(sep)

    if postprocess:
        print(f"{'City':<14} {'N':>6}  "
              f"{'IoU_raw':>8} {'IoU_cln':>8}  "
              f"{'Dice_raw':>9} {'Dice_cln':>9}  "
              f"{'ΔIoU':>7}")
        print(sep)
        for city, data in results["per_city"].items():
            r = data["raw"]
            c = data.get("clean", {})
            delta = c.get("iou", 0) - r["iou"] if c else 0
            print(f"{city:<14} {data['n_samples']:>6}  "
                  f"{r['iou']:>8.4f} {c.get('iou', 0):>8.4f}  "
                  f"{r['dice']:>9.4f} {c.get('dice', 0):>9.4f}  "
                  f"{delta:>+7.4f}")
    else:
        print(f"{'City':<14} {'N':>6}  "
              f"{'IoU':>8} {'Dice':>8} {'Prec':>8} {'Recall':>8}")
        print(sep)
        for city, data in results["per_city"].items():
            r = data["raw"]
            print(f"{city:<14} {data['n_samples']:>6}  "
                  f"{r['iou']:>8.4f} {r['dice']:>8.4f} "
                  f"{r['precision']:>8.4f} {r['recall']:>8.4f}")

    ov = results["overall"]
    print(sep)
    r = ov["raw"]
    if postprocess and "clean" in ov:
        c = ov["clean"]
        print(f"{'OVERALL':<14} {'':>6}  "
              f"{r['iou']:>8.4f} {c['iou']:>8.4f}  "
              f"{r['dice']:>9.4f} {c['dice']:>9.4f}  "
              f"{c['iou']-r['iou']:>+7.4f}")
    else:
        print(f"{'OVERALL':<14} {'':>6}  "
              f"{r['iou']:>8.4f} {r['dice']:>8.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f}")
    print(sep)