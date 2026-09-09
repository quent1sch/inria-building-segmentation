"""
evaluation/visualisation.py
 
Qualitative visualisation and report generation.
 
Functions
---------
  save_prediction_grid(samples_with_preds, out_dir, city)
      Saves a grid PNG: image | GT mask | raw overlay | clean overlay.
      Written to ckpt_dir/qualitative/predictions_grid_{city}.png.
 
  generate_report(ckpt_dir, mode, checkpoint_path, selection)
      Generates report.md from all completed module runs under ckpt_dir.
 
      Expected directory structure (see evaluate.py for full details):
        ckpt_dir/
          pixel_{eval_params_hash}/
            metrics_pixel.json
            .meta_pixel.json          ← provenance: cities, n_samples, params, timestamp
          building_{hash}/
            metrics_building.json
            .meta_building.json
          threshold_{hash}/
            threshold_analysis.json
            threshold_analysis.png
            .meta_threshold.json
          qualitative/
            predictions_grid_{city}.png
          report_selection.yaml       ← controls which run per module is shown
          report.md                   ← output
 
      selection dict ({module: eval_params_hash}) is read from
      report_selection.yaml by evaluate.py and passed here.
      None or missing module → most recent run used (by generated_at).
 
      Each report section includes:
        - Plain-English description of what was measured and how to interpret it
        - Italicised context line from .meta_*.json (cities, n_samples, params, timestamp)
        - Results table
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
    samples_with_preds : list of dicts with keys:
        image, gt_mask, raw_mask, clean_mask (optional), name
    out_dir : output directory
    city    : city label for filename
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping qualitative grid.")
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

    plt.suptitle(f"Qualitative Results — {city}", fontsize=11, fontweight="bold")
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
    ckpt_dir: Path,
    mode: str,
    checkpoint_path: str,
    selection: dict = None,
) -> None:
    """
    Generate report.md aggregating results from all available module runs.

    Directory structure expected:
      ckpt_dir/
        pixel_{hash}/
          metrics_pixel.json
          .meta_pixel.json
        building_{hash}/
          metrics_building.json
          .meta_building.json
        threshold_{hash}/
          threshold_analysis.json
          threshold_analysis.png
          .meta_threshold.json
        qualitative/
          predictions_grid_{city}.png
        report_selection.yaml   ← controls which run per module is shown

    Parameters
    ----------
    ckpt_dir         : outputs/evaluation/{checkpoint_hash}/
    mode             : inria | custom
    checkpoint_path  : path to checkpoint file (for header)
    selection        : {module_name: eval_params_hash} from report_selection.yaml.
                       None or missing module → use most recent run for that module.
    """
    from datetime import datetime, timezone

    selection = selection or {}

    # ── resolve module dirs ───────────────────────────────────────────────
    # For each module, find the selected or most recent eval run subdir.
    def _resolve_module_dir(module_name: str) -> tuple[Path | None, dict]:
        """Return (module_dir, meta) for the selected or most recent run."""
        pattern = f"{module_name}_*"
        candidates = sorted(ckpt_dir.glob(pattern))
        if not candidates:
            return None, {}

        # If a specific run is selected in report_selection.yaml, use it
        selected_hash = selection.get(module_name)
        if selected_hash:
            for d in candidates:
                if d.name == f"{module_name}_{selected_hash}":
                    meta_path = d / f".meta_{module_name}.json"
                    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                    return d, meta

        # Default: most recent by generated_at in .meta_*.json
        best_dir  = None
        best_ts   = ""
        best_meta = {}
        for d in candidates:
            meta_path = d / f".meta_{module_name}.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            ts   = meta.get("generated_at", "")
            if ts > best_ts:
                best_ts, best_dir, best_meta = ts, d, meta
        return best_dir, best_meta

    # ── report header ─────────────────────────────────────────────────────
    ckpt_hash = ckpt_dir.name
    lines = [
        "# Evaluation Report\n\n",
        f"| | |\n|---|---|\n",
        f"| **Checkpoint** | `{checkpoint_path}` |\n",
        f"| **Checkpoint hash** | `{ckpt_hash}` |\n",
        f"| **Mode** | {mode} |\n",
        f"| **Generated** | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |\n\n",
        "---\n\n",
    ]

    # ── pixel metrics ─────────────────────────────────────────────────────
    pixel_dir, pixel_meta = _resolve_module_dir("pixel")
    pixel_json = pixel_dir / "metrics_pixel.json" if pixel_dir else None
    if pixel_json and pixel_json.exists():
        lines.append("## Pixel-level Metrics\n\n")
        lines.append(
            "Measures segmentation quality at the pixel level. "
            "IoU (Intersection over Union) is the primary metric — higher is better, "
            "max 1.0. **Raw**: direct model output. **Clean**: vectorized "
            "(polygonized → Douglas-Peucker simplified → area filtered) then "
            "rasterized back to pixels. A positive ΔIoU means postprocessing "
            "improves pixel accuracy; a small negative is expected and acceptable "
            "(straight-edge fitting moves a few boundary pixels).\n\n"
        )
        lines.append(_eval_context_line(pixel_meta))
        data = json.loads(pixel_json.read_text())
        lines.append("| City | IoU (raw) | IoU (clean) | ΔIoU | Dice (raw) | Dice (clean) |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for city, m in data["per_city"].items():
            r     = m.get("raw",   {})
            c     = m.get("clean", {})
            delta = c.get("iou", 0) - r.get("iou", 0) if c else 0
            sign  = "+" if delta >= 0 else ""
            lines.append(
                f"| {city} | {r.get('iou',0):.4f} | {c.get('iou',0):.4f} "
                f"| {sign}{delta:.4f} "
                f"| {r.get('dice',0):.4f} | {c.get('dice',0):.4f} |\n"
                if c else
                f"| {city} | {r.get('iou',0):.4f} | — | — "
                f"| {r.get('dice',0):.4f} | — |\n"
            )
        if "overall" in data:
            r  = data["overall"].get("raw",   {})
            c  = data["overall"].get("clean", {})
            delta = c.get("iou", 0) - r.get("iou", 0) if c else 0
            sign  = "+" if delta >= 0 else ""
            lines.append(
                f"| **Overall** | **{r.get('iou',0):.4f}** | **{c.get('iou',0):.4f}** "
                f"| **{sign}{delta:.4f}** "
                f"| **{r.get('dice',0):.4f}** | **{c.get('dice',0):.4f}** |\n\n"
                if c else
                f"| **Overall** | **{r.get('iou',0):.4f}** | — | — "
                f"| **{r.get('dice',0):.4f}** | — |\n\n"
            )

    # ── building metrics ──────────────────────────────────────────────────
    bld_dir, bld_meta = _resolve_module_dir("building")
    bld_json = bld_dir / "metrics_building.json" if bld_dir else None
    if bld_json and bld_json.exists():
        lines.append("## Building-level Metrics\n\n")
        lines.append(
            "Evaluates whether individual buildings are detected, not just pixels. "
            "A predicted polygon is a **true positive** if it overlaps a GT building "
            "by IoU ≥ 0.5 (COCO-style greedy matching). "
            "**Precision**: of predicted buildings, fraction that matched a GT building. "
            "**Recall**: of GT buildings, fraction that were detected. "
            "**Mean matched IoU**: average overlap quality for matched pairs — "
            "measures boundary quality beyond binary detection. "
            "**Miss rate**: fraction of GT buildings not detected. "
            "Size strata: small <50 m², medium 50–500 m², large >500 m².\n\n"
        )
        lines.append(_eval_context_line(bld_meta, extra_params=True))
        data = json.loads(bld_json.read_text())
        lines.append(
            "| City | P↑ (raw) | R↑ (raw) | F1↑ (raw) | mIoU↑ (raw) "
            "| P↑ (clean) | R↑ (clean) | F1↑ (clean) | mIoU↑ (clean) |\n"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|\n")
        for city, m in data["per_city"].items():
            r = m.get("raw",   {})
            c = m.get("clean", {})
            lines.append(
                f"| {city} "
                f"| {r.get('precision',0):.3f} | {r.get('recall',0):.3f} "
                f"| {r.get('f1',0):.3f} | {r.get('mean_iou_matched',0):.3f} "
                f"| {c.get('precision',0):.3f} | {c.get('recall',0):.3f} "
                f"| {c.get('f1',0):.3f} | {c.get('mean_iou_matched',0):.3f} |\n"
            )
        if data.get("overall"):
            r = data["overall"].get("raw",   {})
            c = data["overall"].get("clean", {})
            lines.append(
                f"| **Overall** "
                f"| **{r.get('precision',0):.3f}** | **{r.get('recall',0):.3f}** "
                f"| **{r.get('f1',0):.3f}** | **{r.get('mean_iou_matched',0):.3f}** "
                f"| **{c.get('precision',0):.3f}** | **{c.get('recall',0):.3f}** "
                f"| **{c.get('f1',0):.3f}** | **{c.get('mean_iou_matched',0):.3f}** |\n\n"
            )
            ov_raw = data["overall"].get("raw", {})
            if "by_size" in ov_raw:
                ov_clean = data["overall"].get("clean", {})
                lines.append("**Size-stratified recall** (fraction of GT buildings detected):\n\n")
                lines.append("| Size | GT buildings | Recall (raw) | Recall (clean) |\n")
                lines.append("|---|---|---|---|\n")
                for label, rd in ov_raw["by_size"].items():
                    cd = ov_clean.get("by_size", {}).get(label, {})
                    lines.append(
                        f"| {label} | {rd.get('n_gt',0)} "
                        f"| {rd.get('recall',0):.3f} | {cd.get('recall',0):.3f} |\n"
                    )
                lines.append("\n")

    # ── threshold analysis ────────────────────────────────────────────────
    thr_dir, thr_meta = _resolve_module_dir("threshold")
    thr_json = thr_dir / "threshold_analysis.json" if thr_dir else None
    if thr_json and thr_json.exists():
        lines.append("## Threshold Analysis\n\n")
        lines.append(
            "Sweeps the classification threshold from 0.05 to 0.95 and computes "
            "Precision, Recall, F1, and IoU at each step. The **optimal threshold** "
            "maximises F1. The default threshold (0.5) may not be optimal — "
            "compare F1 at optimal vs F1 at 0.5 to assess the gap. "
            "A lower optimal threshold means the model is conservative (misses buildings); "
            "higher means it over-predicts.\n\n"
        )
        lines.append(_eval_context_line(thr_meta))
        data = json.loads(thr_json.read_text())
        lines.append(
            f"| Optimal threshold | F1 at optimal | IoU at optimal |\n"
            f"|---|---|---|\n"
            f"| **{data.get('optimal_threshold', '—')}** "
            f"| **{data.get('optimal_f1', 0):.4f}** "
            f"| **{data.get('optimal_iou', 0):.4f}** |\n\n"
        )
        thr_png = thr_dir / "threshold_analysis.png"
        if thr_png.exists():
            lines.append(f"![Threshold Analysis]({thr_dir.name}/threshold_analysis.png)\n\n")

    # ── postprocessing sensitivity ────────────────────────────────────────
    pp_dir, pp_meta = _resolve_module_dir("postproc")
    pp_json = pp_dir / "postproc_sensitivity.json" if pp_dir else None
    if pp_json and pp_json.exists():
        lines.append("## Postprocessing Sensitivity\n\n")
        lines.append(
            "Sweeps Douglas-Peucker tolerance and minimum building area filter "
            "independently. Primary metrics are building-level (recall, F1, "
            "mean matched IoU) — these reveal whether postprocessing improves "
            "detection quality. Pixel IoU is shown as a secondary sanity check "
            "and is expected to drop slightly with aggressive simplification "
            "(straight-edge fitting moves boundary pixels) — this is not a concern "
            "unless the drop is large.\n\n"
        )
        lines.append(_eval_context_line(pp_meta))
        pp_png = pp_dir / "postproc_sensitivity.png"
        if pp_png.exists():
            lines.append(f"![Postprocessing Sensitivity]({pp_dir.name}/postproc_sensitivity.png)\n\n")

    # ── resolution robustness ─────────────────────────────────────────────
    res_dir, res_meta = _resolve_module_dir("resolution")
    res_json = res_dir / "resolution_robustness.json" if res_dir else None
    if res_json and res_json.exists():
        lines.append("## Resolution Robustness\n\n")
        lines.append(
            "Evaluates model performance under different resolution conditions. "
            "**native**: inference at the image's native resolution (no resampling). "
            "**resampled**: image resampled to training resolution (0.3 m/px) before inference. "
            "**coarse_0.6 / coarse_1.0**: simulated coarser inputs. "
            "The output mask is always upsampled back to original dimensions before "
            "computing metrics, so comparisons are fair. "
            "A large gap between native and resampled shows the value of the "
            "resolution correction pipeline.\n\n"
        )
        lines.append(_eval_context_line(res_meta))
        data = json.loads(res_json.read_text())
        lines.append("| Condition | IoU | Dice | F1 | Precision | Recall |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for label, m in data["per_condition"].items():
            lines.append(
                f"| {label} | {m.get('iou',0):.4f} | {m.get('dice',0):.4f} "
                f"| {m.get('f1',0):.4f} | {m.get('precision',0):.4f} "
                f"| {m.get('recall',0):.4f} |\n"
            )
        lines.append("\n")
        res_png = res_dir / "resolution_robustness.png"
        if res_png.exists():
            lines.append(f"![Resolution Robustness]({res_dir.name}/resolution_robustness.png)\n\n")

    # ── qualitative ───────────────────────────────────────────────────────
    qual_dir = ckpt_dir / "qualitative"
    grid_files = sorted(qual_dir.glob("predictions_grid_*.png")) if qual_dir.exists() else []
    if grid_files:
        lines.append("## Qualitative Results\n\n")
        lines.append(
            "Each column shows one patch: original image | ground truth mask | "
            "raw prediction overlay | clean prediction overlay.\n\n"
        )
        for gf in grid_files:
            city = gf.stem.replace("predictions_grid_", "")
            lines.append(f"### {city.capitalize()}\n\n")
            lines.append(f"![{city}](qualitative/{gf.name})\n\n")

    report_path = ckpt_dir / "report.md"
    report_path.write_text("".join(lines))
    print(f"  Report → {report_path}")


def _eval_context_line(meta: dict, extra_params: bool = False) -> str:
    """
    Generate a short italicised context line from module metadata.
    Placed at the top of each section so the reader knows exactly
    what data and params produced these results.

    Example:
      *Evaluated on: vienna | 200 samples | 2026-09-08 14:22 UTC*
    """
    if not meta:
        return ""

    parts = []
    cities = meta.get("eval_cities")
    if cities:
        parts.append(f"cities={', '.join(cities)}")
    n = meta.get("n_samples_evaluated")
    if n:
        parts.append(f"{n} samples")
    if extra_params:
        tol = meta.get("simplify_tolerance_m")
        area = meta.get("min_area_m2")
        if tol is not None:
            parts.append(f"simplify={tol}m")
        if area is not None:
            parts.append(f"min_area={area}m²")
    ts = meta.get("generated_at", "")
    if ts:
        parts.append(ts[:16].replace("T", " ") + " UTC")

    return f"*Evaluated on: {' | '.join(parts)}*\n\n" if parts else ""