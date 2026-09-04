"""
evaluation/visualisation.py

Qualitative visualisation: prediction grid images and summary report.

Generates:
  qualitative/predictions_grid_{city}.png - image | GT | raw pred | clean pred
  report.md - auto-generated markdown summary
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def save_prediction_grid(
    samples_with_preds: list[dict],
    out_dir: Path,
    city: str,
    n_cols: int = 4,
) -> None:
    """
    Save a qualitative grid for one city.

    Each column shows one sample:
      row 0: original image
      row 1: ground truth mask
      row 2: raw prediction overlay
      row 3: clean prediction overlay (if available)

    Parameters
    ----------
    samples_with_preds: list of dicts with keys:
        image, gt_mask, raw_mask, clean_mask (optional), name
    out_dir: output directory
    city: city label for filename
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available - skipping qualitative grid.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    has_clean = any("clean_mask" in s for s in samples_with_preds)
    n_rows_per_sample = 4 if has_clean else 3
    n   = min(len(samples_with_preds), n_cols * 3)
    cols = min(n, n_cols)
    rows = ((n + cols - 1) // cols) * n_rows_per_sample

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for i, sample in enumerate(samples_with_preds[:n]):
        col      = i % cols
        row_base = (i // cols) * n_rows_per_sample

        img      = sample["image"]
        gt       = sample["gt_mask"].astype(np.uint8) * 255
        raw_ov   = _make_overlay(img, sample["raw_mask"])

        axes[row_base, col].imshow(img)
        axes[row_base, col].set_title(sample.get("name", "")[:20], fontsize=7)

        axes[row_base + 1, col].imshow(gt, cmap="gray")
        axes[row_base + 1, col].set_title("Ground Truth", fontsize=7)

        axes[row_base + 2, col].imshow(raw_ov)
        axes[row_base + 2, col].set_title("Raw Prediction", fontsize=7)

        if has_clean and "clean_mask" in sample:
            clean_ov = _make_overlay(img, sample["clean_mask"])
            axes[row_base + 3, col].imshow(clean_ov)
            axes[row_base + 3, col].set_title("Clean Prediction", fontsize=7)

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle(f"Qualitative Results - {city}", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out_path = out_dir / f"predictions_grid_{city}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def _make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Semi-transparent red overlay for building predictions."""
    overlay = image.copy()
    overlay[mask > 0] = np.clip(
        overlay[mask > 0].astype(int) * 0.5 + np.array([255, 50, 50]) * 0.5,
        0, 255,
    ).astype(np.uint8)
    return overlay


def generate_report(
    out_dir: Path,
    mode: str,
    checkpoint_path: str,
) -> None:
    """
    Auto-generate a markdown report from ALL evaluation JSON files present
    in out_dir — regardless of when they were produced or which session
    ran them.

    This is the correct design: the report is a view over the output
    directory, not a side effect of a specific run. Use --report-only in
    evaluate.py to generate without running inference.

    Checkpoint consistency check: reads .meta_*.json sidecar files written
    by evaluate.py alongside each module output. If results from different
    checkpoints are detected, a warning is included in the report header.
    """
    import glob

    # ── checkpoint consistency check ──────────────────────────────────────
    meta_files  = list(out_dir.glob(".meta_*.json"))
    checkpoints = set()
    modules_found = []
    for mf in meta_files:
        try:
            meta = json.loads(mf.read_text())
            checkpoints.add(meta.get("checkpoint", "unknown"))
            modules_found.append(meta.get("module", mf.stem))
        except Exception:
            pass

    consistency_warning = ""
    if len(checkpoints) > 1:
        consistency_warning = (
            "\\n> ⚠️ **Warning:** results in this report were produced by "
            f"**different checkpoints**: {', '.join(f'`{c}`' for c in checkpoints)}. "
            "Results may not be comparable.\\n"
        )

    lines = [
        "# Evaluation Report\\n",
        f"**Checkpoint:** `{checkpoint_path}`  \\n",
        f"**Mode:** {mode}  \\n",
    ]
    if modules_found:
        lines.append(f"**Modules available:** {', '.join(sorted(set(modules_found)))}  \\n")
    if consistency_warning:
        lines.append(consistency_warning)
    lines.append("\\n---\\n")

    # ── pixel metrics ─────────────────────────────────────────────────────
    pixel_json = out_dir / "metrics_pixel.json"
    if pixel_json.exists():
        lines.append("## Pixel-level Metrics\n\n")
        data = json.loads(pixel_json.read_text())
        lines.append("| City | IoU (raw) | IoU (clean) | Dice (raw) | Dice (clean) |\n")
        lines.append("|---|---|---|---|---|\n")
        for city, m in data["per_city"].items():
            r = m["raw"]
            c = m.get("clean", {})
            lines.append(
                f"| {city} | {r['iou']:.4f} | {c.get('iou', '—'):.4f} "
                f"| {r['dice']:.4f} | {c.get('dice', '—'):.4f} |\n"
                if c else
                f"| {city} | {r['iou']:.4f} | — | {r['dice']:.4f} | — |\n"
            )
        ov = data["overall"]["raw"]
        lines.append(f"| **Overall** | **{ov['iou']:.4f}** | | **{ov['dice']:.4f}** | |\n\n")

    # ── building metrics ──────────────────────────────────────────────────
    # Updated to match the current metrics_building.py output structure:
    # per_city[city] = {"raw": {...}, "clean": {...}}
    # (old format was flat: per_city[city] = {"precision": ..., ...})
    building_json = out_dir / "metrics_building.json"
    if building_json.exists():
        lines.append("## Building-level Metrics\n\n")
        data = json.loads(building_json.read_text())
        lines.append(f"*IoU matching threshold: {data.get('iou_threshold', 0.5)}*\n\n")
        lines.append(
            "| City | P (raw) | R (raw) | F1 (raw) | mIoU (raw) "
            "| P (clean) | R (clean) | F1 (clean) | mIoU (clean) |\n"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|\n")
        for city, m in data["per_city"].items():
            r = m.get("raw", {})
            c = m.get("clean", {})
            lines.append(
                f"| {city} "
                f"| {r.get('precision', 0):.3f} | {r.get('recall', 0):.3f} "
                f"| {r.get('f1', 0):.3f} | {r.get('mean_iou_matched', 0):.3f} "
                f"| {c.get('precision', 0):.3f} | {c.get('recall', 0):.3f} "
                f"| {c.get('f1', 0):.3f} | {c.get('mean_iou_matched', 0):.3f} |\n"
            )
        if data.get("overall"):
            r = data["overall"].get("raw", {})
            c = data["overall"].get("clean", {})
            lines.append(
                f"| **Overall** "
                f"| **{r.get('precision', 0):.3f}** | **{r.get('recall', 0):.3f}** "
                f"| **{r.get('f1', 0):.3f}** | **{r.get('mean_iou_matched', 0):.3f}** "
                f"| **{c.get('precision', 0):.3f}** | **{c.get('recall', 0):.3f}** "
                f"| **{c.get('f1', 0):.3f}** | **{c.get('mean_iou_matched', 0):.3f}** |\n\n"
            )
        # Size-stratified recall if available
        ov_raw = data.get("overall", {}).get("raw", {})
        if "by_size" in ov_raw:
            lines.append("**Size-stratified recall (raw → clean):**\n\n")
            lines.append("| Size | GT buildings | Detected (raw) | Recall (raw) | Detected (clean) | Recall (clean) |\n")
            lines.append("|---|---|---|---|---|---|\n")
            ov_clean = data.get("overall", {}).get("clean", {})
            for label, rd in ov_raw["by_size"].items():
                cd = ov_clean.get("by_size", {}).get(label, {})
                lines.append(
                    f"| {label} | {rd.get('n_gt', 0)} "
                    f"| {rd.get('n_detected', 0)} | {rd.get('recall', 0):.3f} "
                    f"| {cd.get('n_detected', 0)} | {cd.get('recall', 0):.3f} |\n"
                )
            lines.append("\n")

    # ── threshold analysis ────────────────────────────────────────────────
    thresh_json = out_dir / "threshold_analysis.json"
    if thresh_json.exists():
        lines.append("## Threshold Analysis\n\n")
        data = json.loads(thresh_json.read_text())
        lines.append(
            f"Optimal threshold: **{data['optimal_threshold']}**  \n"
            f"F1 at optimal: **{data['optimal_f1']:.4f}**  \n"
            f"IoU at optimal: **{data['optimal_iou']:.4f}**  \n\n"
        )
        lines.append("![Threshold Analysis](threshold_analysis.png)\n\n")

    # ── postprocessing sensitivity ────────────────────────────────────────
    postproc_json = out_dir / "postproc_sensitivity.json"
    if postproc_json.exists():
        lines.append("## Postprocessing Sensitivity\n\n")
        lines.append("![Postprocessing Sensitivity](postproc_sensitivity.png)\n\n")

    # ── resolution robustness ─────────────────────────────────────────────
    res_json = out_dir / "resolution_robustness.json"
    if res_json.exists():
        lines.append("## Resolution Robustness\n\n")
        data = json.loads(res_json.read_text())
        lines.append("| Condition | IoU | Dice | F1 |\n")
        lines.append("|---|---|---|---|\n")
        for label, m in data["per_condition"].items():
            lines.append(f"| {label} | {m['iou']:.4f} | {m['dice']:.4f} | {m['f1']:.4f} |\n")
        lines.append("\n![Resolution Robustness](resolution_robustness.png)\n\n")

    report_path = out_dir / "report.md"
    report_path.write_text("".join(lines))
    print(f"  Report → {report_path}")
