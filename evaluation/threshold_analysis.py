"""
evaluation/threshold_analysis.py

Threshold sensitivity analysis.

Sweeps the classification threshold from 0.05 to 0.95 and computes
pixel-level Precision, Recall, F1, and IoU at each step.

Outputs
-------
  threshold_analysis.png - two subplots: PR curve + F1/IoU vs threshold
  threshold_table.csv - metrics at key thresholds
  threshold_analysis.json - full sweep results + optimal threshold

Why this matters
----------------
The default threshold of 0.5 is arbitrary. For building segmentation:
  - Lower threshold -> higher recall (fewer missed buildings) but more FP
  - Higher threshold -> higher precision (fewer false alarms) but more misses
  The optimal F1 threshold may differ significantly from 0.5.
  This analysis finds the empirically best threshold for this dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


KEY_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
SWEEP_THRESHOLDS = np.linspace(0.05, 0.95, 37).tolist()


def run(
    samples,
    model,
    out_dir: Path,
) -> dict:
    """
    Run threshold analysis across all samples.

    Accumulates raw pixel counts (TP, FP, FN) across all samples at each
    threshold, then computes metrics. This is more statistically robust
    than averaging per-sample metrics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect probability maps and GT masks
    print("  Collecting probability maps...")
    prob_maps = []
    gt_masks  = []

    for sample in samples:
        # Get raw probability map (bypass threshold in model)
        prob, _ = _get_prob_map(model, sample)
        prob_maps.append(prob)
        gt_masks.append(sample.gt_mask.astype(bool))

    print(f"  Sweeping {len(SWEEP_THRESHOLDS)} thresholds over "
          f"{len(prob_maps)} samples...")

    sweep_results = []

    for thresh in SWEEP_THRESHOLDS:
        tp_total = fp_total = fn_total = 0

        for prob, gt in zip(prob_maps, gt_masks):
            pred = prob > thresh
            tp_total += int((pred &  gt).sum())
            fp_total += int((pred & ~gt).sum())
            fn_total += int((~pred & gt).sum())

        smooth = 1e-6
        precision = (tp_total + smooth) / (tp_total + fp_total + smooth)
        recall = (tp_total + smooth) / (tp_total + fn_total + smooth)
        f1 = 2 * precision * recall / (precision + recall + smooth)
        iou = (tp_total + smooth) / (tp_total + fp_total + fn_total + smooth)
        fpr = fp_total / (fp_total + (len(prob_maps[0].ravel()) * len(prob_maps)
                                        - tp_total - fp_total - fn_total) + smooth)

        sweep_results.append({
            "threshold": round(float(thresh), 3),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "fpr": float(fpr),
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
        })

    # ── find optimal threshold ─────────────────────────────────────────────
    best = max(sweep_results, key=lambda x: x["f1"])
    print(f"  Optimal threshold: {best['threshold']:.2f}  "
          f"(F1={best['f1']:.4f}  IoU={best['iou']:.4f}  "
          f"P={best['precision']:.4f}  R={best['recall']:.4f})")

    results = {
        "sweep": sweep_results,
        "optimal_threshold": best["threshold"],
        "optimal_f1": best["f1"],
        "optimal_iou": best["iou"],
    }

    # ── save ──────────────────────────────────────────────────────────────
    json_path = out_dir / "threshold_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    _save_csv(sweep_results, out_dir / "threshold_table.csv")
    _plot(sweep_results, best, out_dir / "threshold_analysis.png")

    print(f"  Saved -> {out_dir}/threshold_analysis.{{json,png,csv}}")
    return results


def _get_prob_map(model, sample) -> tuple[np.ndarray, object]:
    """
    Extract raw float probability map from the model, bypassing threshold.
    Temporarily overrides model.threshold to 0 so predict() returns all-True,
    then manually applies softmax via the internal _predict_array method.
    """
    # Access internal probability map directly
    original_h, original_w = sample.image.shape[:2]

    from api.inference import resample_image, upsample_prob

    result = model._preprocess(sample.image, sample.resolution, resample=True)
    if len(result) == 4:
        proc_image, info, orig_h, orig_w = result
    else:
        proc_image, info = result
        orig_h, orig_w = original_h, original_w

    prob = model._predict_array(proc_image)

    if info.resampled:
        prob = upsample_prob(prob, orig_h, orig_w)

    return prob, info


def _save_csv(sweep: list, path: Path) -> None:
    import csv
    fields = ["threshold", "precision", "recall", "f1", "iou"]
    # Save key thresholds + optimal
    key_rows = [r for r in sweep if round(r["threshold"], 2) in KEY_THRESHOLDS]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in key_rows:
            writer.writerow({k: round(row[k], 4) for k in fields})


def _plot(sweep: list, best: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot.")
        return

    thresholds = [r["threshold"] for r in sweep]
    precisions = [r["precision"] for r in sweep]
    recalls = [r["recall"] for r in sweep]
    f1s = [r["f1"] for r in sweep]
    ious = [r["iou"] for r in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Precision-Recall curve ────────────────────────────────────────────
    ax1.plot(recalls, precisions, "b-", lw=2, label="PR curve")
    # Mark key thresholds
    for r in sweep:
        if round(r["threshold"], 2) in KEY_THRESHOLDS:
            ax1.plot(r["recall"], r["precision"], "ko", ms=5)
            ax1.annotate(f"t={r['threshold']:.1f}",
                         (r["recall"], r["precision"]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curve")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ── F1 and IoU vs threshold ───────────────────────────────────────────
    ax2.plot(thresholds, f1s, "b-", lw=2, label="F1")
    ax2.plot(thresholds, ious, "r--", lw=2, label="IoU")
    ax2.axvline(best["threshold"], color="g", linestyle=":", lw=1.5,
                label=f"Optimal t={best['threshold']:.2f}")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("F1 and IoU vs Threshold")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle("Threshold Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()