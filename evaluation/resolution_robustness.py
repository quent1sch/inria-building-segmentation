"""
evaluation/resolution_robustness.py

Resolution robustness evaluation.

Given images at a known resolution (e.g. SWISSIMAGE at 0.1m/px), evaluates
model performance under different resolution conditions:

  native       Inference at native resolution (no resampling)
  resampled    Inference after resampling to training resolution (0.3m/px)
  coarse_0.6   Inference after downsampling to 0.6m/px (simulated coarser)
  coarse_1.0   Inference after downsampling to 1.0m/px (simulated coarser)

For each condition, the output mask is always upsampled back to the original
pixel grid before computing metrics against the GT mask. This ensures a fair
comparison — we're always measuring quality at the same resolution.

This answers two questions:
  1. How much does resolution mismatch hurt? (native vs resampled)
  2. Does our resampling pipeline fix it? (native vs resampled vs coarse)

Only available for custom mode where input resolution is known.

Outputs
-------
  resolution_robustness.png - bar chart of IoU/Dice/F1 per condition
  resolution_robustness.json - full results
  resolution_robustness.csv - summary table
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# Resolution conditions to evaluate
# (label, target_resolution_m_per_px, resample_flag)
# resample_flag=True -> use inference.py resampling pipeline
# resample_flag=False -> inference at specified resolution, no resampling
CONDITIONS = [
    ("native",      None,  False),   # inference at whatever resolution the image is
    ("resampled",   None,  True),    # resample to 0.3m/px (standard pipeline)
    ("coarse_0.6",  0.6,   False),   # simulate 0.6m/px input
    ("coarse_1.0",  1.0,   False),   # simulate 1.0m/px input
]


def run(
    samples,
    model,
    out_dir: Path,
) -> dict:
    """
    Run resolution robustness evaluation.

    Samples must have a known resolution (resolution != None).
    Samples without resolution are skipped with a warning.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    from api.inference import resample_image, upsample_prob, TRAINING_RESOLUTION
    from evaluation.metrics_pixel import pixel_metrics, aggregate_metrics

    # Filter to samples with known resolution
    valid_samples = []
    for s in samples:
        if s.resolution is None:
            warnings.warn(f"Sample {s.name} has no resolution - skipping.")
        else:
            valid_samples.append(s)

    if not valid_samples:
        raise ValueError(
            "No samples with known resolution found. "
            "Resolution robustness requires images with known m/px resolution. "
            "Use SWISSIMAGE GeoTIFF tiles or supply --resolution."
        )

    print(f"  {len(valid_samples)} samples with known resolution.")

    condition_results: dict[str, list] = {label: [] for label, _, _ in CONDITIONS}

    for i, sample in enumerate(valid_samples):
        print(f"  [{i+1}/{len(valid_samples)}] {sample.name} "
              f"(native res={sample.resolution:.3f}m/px)")

        original_h, original_w = sample.image.shape[:2]

        for label, target_res, resample in CONDITIONS:
            # ── prepare image for this condition ──────────────────────────
            if target_res is not None:
                # Simulate a coarser input by downsampling
                if target_res <= sample.resolution:
                    # Can't simulate a finer image from a coarser one
                    warnings.warn(
                        f"  Skipping condition '{label}': target {target_res}m/px "
                        f"is finer than native {sample.resolution}m/px."
                    )
                    condition_results[label].append(None)
                    continue
                print("resample image")#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                sim_image = resample_image(sample.image, sample.resolution, target_res)
                sim_resolution = target_res
            else:
                sim_image = sample.image
                sim_resolution = sample.resolution

            # ── run inference for this condition ──────────────────────────
            print("preprocess")#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            result = model._preprocess(sim_image, sim_resolution, resample=resample)
            if len(result) == 4:
                proc_image, info, oh, ow = result
            else:
                proc_image, info = result
                oh, ow = sim_image.shape[:2]

            print("prob = model._predict_array()")#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            prob = model._predict_array(proc_image)

            if info.resampled:
                prob = upsample_prob(prob, oh, ow)

            # If we simulated a coarser image, upsample back to original dims
            if target_res is not None:
                from api.inference import upsample_prob as up
                prob = up(prob, original_h, original_w)

            mask = prob > model.threshold
            print("pixel_metrics()")#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            m = pixel_metrics(mask, sample.gt_mask)
            m["condition"] = label
            condition_results[label].append(m)

    # ── aggregate ─────────────────────────────────────────────────────────
    results: dict = {"per_condition": {}}
    for label, _, _ in CONDITIONS:
        valid = [m for m in condition_results[label] if m is not None]
        if not valid:
            continue
        agg = aggregate_metrics(valid)
        agg["n_samples"] = len(valid)
        results["per_condition"][label] = agg

    # ── save ──────────────────────────────────────────────────────────────
    json_path = out_dir / "resolution_robustness.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    _save_csv(results, out_dir / "resolution_robustness.csv")
    _plot(results, out_dir / "resolution_robustness.png")
    _print_table(results)

    print(f"  Saved -> {out_dir}/resolution_robustness.{{json,png,csv}}")
    return results


def _save_csv(results: dict, path: Path) -> None:
    import csv
    fields = ["condition", "n_samples", "iou", "dice", "f1", "precision", "recall"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, data in results["per_condition"].items():
            writer.writerow({
                "condition": label,
                "n_samples": data["n_samples"],
                "iou": round(data["iou"], 4),
                "dice": round(data["dice"], 4),
                "f1": round(data["f1"], 4),
                "precision": round(data["precision"], 4),
                "recall": round(data["recall"], 4),
            })


def _print_table(results: dict) -> None:
    sep = "-" * 65
    print(f"\n{'Resolution robustness':^65}")
    print(sep)
    print(f"{'Condition':<14} {'N':>5}  "
          f"{'IoU':>8} {'Dice':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print(sep)
    for label, data in results["per_condition"].items():
        print(f"{label:<14} {data['n_samples']:>5}  "
              f"{data['iou']:>8.4f} {data['dice']:>8.4f} {data['f1']:>8.4f} "
              f"{data['precision']:>8.4f} {data['recall']:>8.4f}")
    print(sep)


def _plot(results: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available - skipping plot.")
        return

    labels = list(results["per_condition"].keys())
    ious = [results["per_condition"][lab]["iou"]  for lab in labels]
    dices = [results["per_condition"][lab]["dice"] for lab in labels]
    f1s = [results["per_condition"][lab]["f1"]   for lab in labels]

    x      = np.arange(len(labels))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, ious, width, label="IoU", color="steelblue")
    ax.bar(x, dices, width, label="Dice", color="tomato")
    ax.bar(x + width, f1s, width, label="F1", color="seagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1])
    ax.set_title("Resolution Robustness - Model Performance by Input Condition")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate with IoU values
    for i, iou in enumerate(ious):
        ax.text(i - width, iou + 0.01, f"{iou:.3f}", ha="center", va="bottom",
                fontsize=8, color="steelblue")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()