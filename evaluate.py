"""
evaluate.py
 
Evaluation pipeline for the Inria building segmentation model.
 
Runs any combination of evaluation modules independently so each can be
executed separately on CPU in acceptable time.
 
Output directory structure (versioned, auto-managed)
-----------------------------------------------------
Results are written to a two-level versioned directory:
 
  out_dir/{checkpoint_hash}/{module}_{eval_params_hash}/
 
  checkpoint_hash   — SHA256[:8] of the checkpoint file.
                      Model identity: all results under a given hash come
                      from the same weights. Computed automatically — no
                      manual naming needed.
 
  eval_params_hash  — SHA256[:8] of the params that affect this module's
                      result (cities, max_per_city, min_area, etc.).
                      Different param combinations produce separate
                      subdirectories and never overwrite each other.
 
  Example:
    outputs/evaluation/
    └── abc123de/                       ← checkpoint hash
        ├── pixel_v1a2b3c4/             ← cities=vienna, max=200
        ├── pixel_ff91ab22/             ← cities=vienna, max=10 (sanity)
        ├── building_d5e6f7a8/          ← min_area=10
        ├── threshold_aa11bb22/
        ├── qualitative/                ← prediction grids (shared)
        ├── report_selection.yaml       ← controls which run per module
        └── report.md                   ← generated from selected runs
 
report_selection.yaml
---------------------
Auto-generated after each module run. Controls which eval run appears in
report.md for each module. Safe to edit manually.
 
  pixel:
      run: v1a2b3c4    # cities=['vienna'], max_per_city=200, 2026-09-08
      # run: ff91ab22  # cities=['vienna'], max_per_city=10,  2026-09-07
 
To roll back to a previous run: comment the active line, uncomment another.
To generate the report from current selection: --report-only.
 
Modes
-----
  inria   — Inria Aerial Image Labeling dataset (patched 512×512 PNG crops).
            GT is rasterized binary masks.
 
  custom  — SWISSIMAGE GeoTIFF tiles + swissTLM3D vector GT.
            Resolution auto-detected from GeoTIFF CRS. GT rasterized
            on-the-fly from TLM_GEBAEUDE_FOOTPRINT polygons.
 
Evaluation modules
------------------
  pixel       Pixel IoU, Dice, Precision, Recall — raw vs clean.
              Raw: direct model output. Clean: vectorized → rasterized.
 
  building    Object-level detection via polygon matching (IoU ≥ 0.5,
              COCO-style). Precision, Recall, F1, mean matched IoU,
              miss rate, false alarm rate — raw vs clean.
              Size-stratified recall (small/medium/large) when resolution known.
 
  threshold   Sweeps threshold 0.05→0.95. PR curve, F1/IoU vs threshold,
              optimal threshold. Answers: is 0.5 the best threshold?
 
  postproc    Sensitivity analysis for Douglas-Peucker tolerance and
              min_area filter. Primary: building recall/F1.
              Secondary: pixel IoU (expected small drop — sanity check).
 
  resolution  Resolution robustness: native / resampled (0.3m/px) /
              simulated coarser (0.6m/px, 1.0m/px).
              Requires --mode custom (images with known resolution).
 
Provenance — what is recorded per module run
--------------------------------------------
Each module writes a .meta_{module}.json sidecar with:
  checkpoint, checkpoint_hash, eval_params_hash, generated_at,
  training_run_id, eval_mode, eval_cities, max_per_city,
  n_samples_evaluated, resample, simplify_tolerance_m,
  min_area_m2, iou_threshold.
 
This is the complete provenance record — enough to reproduce any result exactly.
 
Usage examples
--------------
  # Sanity check — 10 samples, pixel only, no MLflow
  python evaluate.py \
      --checkpoint checkpoints/best_model.pth \
      --mode inria --patches data/patches --cities vienna \
      --max-per-city 10 --eval pixel --no-mlflow --out-dir outputs/evaluation
 
  # Recommended CPU workflow: run modules separately across sessions.
  # Each run is isolated in its own subdir — safe to run in any order.
  python evaluate.py --checkpoint checkpoints/best_model.pth \
      --mode inria --patches data/patches --cities vienna \
      --max-per-city 200 --eval pixel --out-dir outputs/evaluation
  python evaluate.py --checkpoint checkpoints/best_model.pth \
      --mode inria --patches data/patches --cities vienna \
      --max-per-city 200 --eval building --out-dir outputs/evaluation
  python evaluate.py --checkpoint checkpoints/best_model.pth \
      --mode inria --patches data/patches --cities vienna \
      --max-per-city 200 --eval threshold --out-dir outputs/evaluation
  python evaluate.py --checkpoint checkpoints/best_model.pth \
      --mode inria --patches data/patches --cities vienna \
      --max-per-city 50 --eval postproc --out-dir outputs/evaluation
 
  # Generate the full report from all completed module runs.
  # Reads report_selection.yaml (auto-generated) to pick which run per
  # module to include. Edit that file to switch between runs.
  # --mode is required for the report header.
  python evaluate.py \
      --report-only \
      --checkpoint checkpoints/best_model.pth \
      --mode inria \
      --out-dir outputs/evaluation
  # → reads  outputs/evaluation/abc123de/{module}_{hash}/ per selection
  # → writes outputs/evaluation/abc123de/report.md
 
  # Swisstopo custom — pixel + building + resolution
  python evaluate.py \
      --checkpoint checkpoints/best_model.pth \
      --mode custom \
      --images path/to/swissimage_tiles/ \
      --gt path/to/swissTLM3D_2026_LV95_LN02.gdb \
      --eval pixel building resolution \
      --out-dir outputs/evaluation
 
MLflow integration
------------------
Evaluation results are logged to a NEW MLflow run (separate from the
training run) and linked back to it via a tag. Training and evaluation
are cleanly separated in the MLflow UI while maintaining full traceability.
 
  Training run  [run_id: abc-123]
    params: encoder, lr, epochs, warmup_epochs, ...
    metrics: train_loss, val_iou, lr_encoder, ... (per epoch)
    artifacts: best_model.pth
 
  Evaluation run  [run_name: eval-20260901-143022]
    tag: training_run_id    = abc-123     ← links to training run
    tag: eval_run           = true        ← filter eval runs in UI
    tag: checkpoint         = checkpoints/best_model.pth
    tag: checkpoint_hash    = abc123de
    tag: model_epoch        = 42
    tag: model_encoder      = resnet34
    params: eval_mode, eval_cities, eval_modules, eval_max_per_city,
            eval_checkpoint_hash, eval_simplify_tol, eval_min_area
    metrics (eval.* prefix — no collision with training metrics):
      eval.pixel.vienna.iou_raw
      eval.pixel.overall.iou_raw / iou_clean
      eval.building.overall.precision_raw / recall_raw / f1_raw
      eval.building.by_size.small.recall_raw
      eval.threshold.optimal_threshold / optimal_f1 / at_0_5.f1
      eval.postproc.simplify.0_5.building_f1
      eval.postproc.min_area.10.building_recall
      eval.resolution.resampled.iou
    artifacts: metrics_pixel.json, threshold_analysis.png, report.md, ...
 
Use --no-mlflow to skip logging (fast local runs, no tracking server).
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

AVAILABLE_MODULES = ["pixel", "building", "threshold", "postproc", "resolution"]
INRIA_ONLY_MODULES = []
CUSTOM_ONLY_MODULES = ["resolution"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Building segmentation evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── required ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to best_model.pth",
    )
    parser.add_argument(
        "--mode", required=True, choices=["inria", "custom"],
        help="Data source: 'inria' (patched PNG crops) or 'custom' (SWISSIMAGE + TLM3D)",
    )

    # ── eval modules ──────────────────────────────────────────────────────
    parser.add_argument(
        "--eval", nargs="+",
        choices=AVAILABLE_MODULES,
        default=None,
        help=(
            "Evaluation modules to run. Default: all applicable. "
            f"Choices: {AVAILABLE_MODULES}. "
            "'resolution' requires --mode custom."
        ),
    )

    # ── inria mode ────────────────────────────────────────────────────────
    parser.add_argument("--patches", default="data/patches",
                        help="Patches directory (inria mode)")
    parser.add_argument("--cities", nargs="+", default=None,
                        help="Cities to evaluate. Default: all cities in patches dir.")
    parser.add_argument("--max-per-city", type=int, default=None,
                        help="Max samples per city (for faster runs)")

    # ── custom mode ───────────────────────────────────────────────────────
    parser.add_argument("--images", default=None,
                        help="Directory of SWISSIMAGE .tif tiles (custom mode)")
    parser.add_argument("--gt", default=None,
                        help="Path to swissTLM3D .gdb file (custom mode)")
    parser.add_argument("--gt-layer", default="TLM_GEBAEUDE_FOOTPRINT",
                        help="GDB layer name for building footprints")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max total samples (custom mode, for faster runs)")

    # ── postprocessing params ─────────────────────────────────────────────
    parser.add_argument("--simplify-tolerance", type=float, default=0.5,
                        help="Douglas-Peucker epsilon in metres (default 0.5)")
    parser.add_argument("--min-area", type=float, default=10.0,
                        help="Minimum building area in m² (default 10.0)")

    # ── output ────────────────────────────────────────────────────────────
    parser.add_argument("--out-dir", default="outputs/evaluation",
                        help="Output directory for results")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Config file path")

    # ── output ───────────────────────────────────────────────────────────
    # Note: the actual output subdirectory is determined automatically from
    # the checkpoint hash (outputs/evaluation/{hash8}/) — no naming needed.

    # ── report ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--report-only",
        action="store_true",
        default=False,
        help=(
            "Skip inference entirely. Read all JSON result files already "
            "present in --out-dir and generate report.md + qualitative grid "
            "from them. Use this after running modules separately across "
            "multiple sessions. No model or checkpoint needed."
        ),
    )

    # ── MLflow ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        default=False,
        help=(
            "Skip MLflow logging. Useful for quick local runs or when the "
            "tracking server is unavailable. Local outputs (JSON, PNG, CSV, "
            "report.md) are always written regardless of this flag."
        ),
    )

    return parser.parse_args()


def load_samples(args, cfg):
    """Build the sample iterable for the selected mode."""
    from evaluation.ground_truth import (
        load_inria_samples,
        load_swissimage_samples,
    )

    if args.mode == "inria":
        from data.dataset import InriaDataset
        patches_dir = Path(args.patches)

        if args.cities:
            cities = args.cities
        else:
            from evaluation.ground_truth import load_inria_samples
            cities = [
                d.name for d in patches_dir.iterdir() if d.is_dir()
            ]
            cities = sorted(cities)
            print(f"Auto-detected cities: {cities}")

        return load_inria_samples(
            patches_dir,
            cities=cities,
            max_per_city=args.max_per_city,
        )

    else:  # custom
        if not args.images:
            print("Error: --images is required for --mode custom", file=sys.stderr)
            sys.exit(1)
        if not args.gt:
            print("Error: --gt is required for --mode custom", file=sys.stderr)
            sys.exit(1)

        return load_swissimage_samples(
            images_dir=args.images,
            gdb_path=args.gt,
            gdb_layer=args.gt_layer,
            max_samples=args.max_samples,
        )


# ── MLflow helpers ───────────────────────────────────────────────────────────

def _resolve_mlflow_run(checkpoint_path: str, cfg: dict):
    """
    Read the training run_id from the checkpoint and return MLflow run kwargs.

    The checkpoint stores mlflow_run_id (added by train.py) which links this
    evaluation back to the exact training run that produced the model weights.

    Returns a dict to pass to mlflow.start_run() and a tags dict.
    """
    training_run_id  = None
    model_epoch      = None
    model_encoder    = None

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        training_run_id = ckpt.get("mlflow_run_id")
        model_epoch     = ckpt.get("epoch")
        model_encoder   = ckpt.get("model_config", {}).get("encoder", "unknown")
    except Exception as e:
        print(f"  Warning: could not read checkpoint metadata: {e}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name  = f"eval-{timestamp}"

    # Tags link this eval run to its training run and document the model used.
    # In the MLflow UI, filter eval runs by training_run_id tag to find all
    # evaluations of a specific model.
    tags = {
        "eval_run":         "true",                  # filter eval runs from training runs
        "checkpoint":       checkpoint_path,
        "model_epoch":      str(model_epoch) if model_epoch else "unknown",
        "model_encoder":    model_encoder or "unknown",
    }
    if training_run_id:
        tags["training_run_id"] = training_run_id    # core traceability link
    else:
        print("  Warning: checkpoint has no mlflow_run_id — eval run will not be "
              "linked to a training run. Re-train with the current train.py to "
              "enable full traceability.")

    return run_name, tags


def _log_to_mlflow(mlflow_run, module_name: str, results: dict, out_dir: Path) -> None:
    """
    Log evaluation results for one module into the active MLflow eval run.

    Metric naming convention:  eval.{module}.{scope}.{metric}_{condition}
      module    = pixel | building | threshold | postproc | resolution
      scope     = city name | overall | by_size | at_{threshold} | simplify | min_area
      condition = raw | clean  (for pixel and building)

    This namespace clearly separates eval metrics from training metrics
    (train_loss, val_iou, lr_encoder, ...) which live in the training run.

    All metrics are logged at step=0 (evaluation is a snapshot, not a time series).
    Artifacts (JSON, PNG, CSV) are always the local files written by each module.
    """
    import mlflow

    def safe_log_metric(key: str, value) -> None:
        """Log only if value is a real number — skip None and non-numeric."""
        if isinstance(value, (int, float)) and value == value:  # NaN check
            # MLflow metric keys must not contain spaces; replace dots with dots
            # (MLflow supports dots in metric names)
            mlflow.log_metric(key, float(value), step=0)

    def log_artifact_if_exists(filename: str) -> None:
        path = out_dir / filename
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=f"evaluation/{module_name}")

    # ── pixel ─────────────────────────────────────────────────────────────
    if module_name == "pixel":
        for city, data in results.get("per_city", {}).items():
            for condition in ("raw", "clean"):
                if condition not in data:
                    continue
                for metric, val in data[condition].items():
                    safe_log_metric(f"eval.pixel.{city}.{metric}_{condition}", val)

        for condition in ("raw", "clean"):
            if condition not in results.get("overall", {}):
                continue
            for metric, val in results["overall"][condition].items():
                safe_log_metric(f"eval.pixel.overall.{metric}_{condition}", val)

        log_artifact_if_exists("metrics_pixel.json")
        log_artifact_if_exists("metrics_pixel.csv")

    # ── building ──────────────────────────────────────────────────────────
    elif module_name == "building":
        for city, data in results.get("per_city", {}).items():
            for condition in ("raw", "clean"):
                if condition not in data:
                    continue
                for metric in ("precision", "recall", "f1", "mean_iou_matched",
                               "miss_rate", "false_alarm_rate"):
                    safe_log_metric(
                        f"eval.building.{city}.{metric}_{condition}",
                        data[condition].get(metric)
                    )

        for condition in ("raw", "clean"):
            ov = results.get("overall", {}).get(condition, {})
            if not ov:
                continue
            for metric in ("precision", "recall", "f1", "mean_iou_matched",
                           "miss_rate", "false_alarm_rate"):
                safe_log_metric(f"eval.building.overall.{metric}_{condition}",
                                ov.get(metric))
            # Size-stratified recall
            for size_label, size_data in ov.get("by_size", {}).items():
                safe_log_metric(
                    f"eval.building.by_size.{size_label}.recall_{condition}",
                    size_data.get("recall")
                )

        log_artifact_if_exists("metrics_building.json")

    # ── threshold ─────────────────────────────────────────────────────────
    elif module_name == "threshold":
        # Log the headline metrics
        safe_log_metric("eval.threshold.optimal_threshold",
                        results.get("optimal_threshold"))
        safe_log_metric("eval.threshold.optimal_f1",
                        results.get("optimal_f1"))
        safe_log_metric("eval.threshold.optimal_iou",
                        results.get("optimal_iou"))

        # Log metrics at key thresholds for easy comparison across models
        key_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        for row in results.get("sweep", []):
            if round(row["threshold"], 2) in key_thresholds:
                t_str = str(row["threshold"]).replace(".", "_")
                for metric in ("precision", "recall", "f1", "iou"):
                    safe_log_metric(f"eval.threshold.at_{t_str}.{metric}",
                                    row.get(metric))

        log_artifact_if_exists("threshold_analysis.json")
        log_artifact_if_exists("threshold_analysis.png")
        log_artifact_if_exists("threshold_table.csv")

    # ── postproc ──────────────────────────────────────────────────────────
    elif module_name == "postproc":
        # Simplify tolerance sweep — primary: building metrics, secondary: pixel IoU
        for row in results.get("simplify_tolerance", []):
            # Use underscore in key: 0.5 → "0_5" (dots allowed in MLflow but
            # underscores are cleaner in the UI filter)
            tol_str = str(row["simplify_tolerance_m"]).replace(".", "_")
            safe_log_metric(f"eval.postproc.simplify.{tol_str}.building_recall",
                            row.get("building_recall"))
            safe_log_metric(f"eval.postproc.simplify.{tol_str}.building_f1",
                            row.get("building_f1"))
            safe_log_metric(f"eval.postproc.simplify.{tol_str}.building_mean_iou",
                            row.get("building_mean_iou"))
            # Secondary sanity check — expected to drop slightly with simplification
            safe_log_metric(f"eval.postproc.simplify.{tol_str}.pixel_iou_secondary",
                            row.get("pixel_iou"))

        # Min area sweep
        for row in results.get("min_area_m2", []):
            area_str = str(int(row["min_area_m2"]))
            safe_log_metric(f"eval.postproc.min_area.{area_str}.building_recall",
                            row.get("building_recall"))
            safe_log_metric(f"eval.postproc.min_area.{area_str}.building_f1",
                            row.get("building_f1"))
            safe_log_metric(f"eval.postproc.min_area.{area_str}.building_mean_iou",
                            row.get("building_mean_iou"))
            safe_log_metric(f"eval.postproc.min_area.{area_str}.pixel_iou_secondary",
                            row.get("pixel_iou"))

        log_artifact_if_exists("postproc_sensitivity.json")
        log_artifact_if_exists("postproc_sensitivity.png")

    # ── resolution ────────────────────────────────────────────────────────
    elif module_name == "resolution":
        for condition, data in results.get("per_condition", {}).items():
            for metric in ("iou", "dice", "f1", "precision", "recall"):
                safe_log_metric(f"eval.resolution.{condition}.{metric}",
                                data.get(metric))

        log_artifact_if_exists("resolution_robustness.json")
        log_artifact_if_exists("resolution_robustness.png")
        log_artifact_if_exists("resolution_robustness.csv")


def _checkpoint_hash(checkpoint_path: str) -> str:
    """
    Compute first 8 chars of SHA256 of the checkpoint file contents.
    Used as a versioned subdirectory under out_dir so results from
    different model versions never silently overwrite each other.

    Example: outputs/evaluation/abc123de/metrics_pixel.json
    """
    import hashlib
    h = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        # Read in chunks — checkpoint files can be 100-500MB
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _eval_params_hash(module_name: str, args) -> str:
    """
    Compute first 8 chars of SHA256 of the eval params that affect this
    module's result. Used as a subdirectory suffix so different param
    combinations never overwrite each other.

    Params included per module:
      pixel:     cities, max_per_city, resample
      building:  cities, max_per_city, resample, iou_threshold,
                 simplify_tolerance, min_area
      threshold: cities, max_per_city, resample
      postproc:  cities, max_per_city, resample  (sweep defines own params)
      resolution: cities, max_per_city
    """
    import hashlib
    import json

    base = {
        "cities":       sorted(getattr(args, "cities", None) or []),
        "max_per_city": getattr(args, "max_per_city", None),
        "resample":     True,  # always True in current pipeline
    }

    if module_name == "building":
        base["iou_threshold"]       = 0.5   # fixed for now
        base["simplify_tolerance_m"] = args.simplify_tolerance
        base["min_area_m2"]          = args.min_area
    elif module_name == "postproc":
        pass  # sweep defines its own params — only data params matter
    elif module_name == "resolution":
        base.pop("resample")  # resolution module varies this itself

    payload = json.dumps(base, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:8]


def _write_module_metadata(
    module_dir: Path,
    module_name: str,
    eval_params_hash: str,
    ckpt_hash: str,
    args,
    n_samples: int = 0,
    training_run_id: str = None,
) -> None:
    """
    Write a complete provenance record alongside each module's output.

    Stored as .meta_{module}.json inside the module's subdirectory.
    Records everything needed to reproduce the result exactly and to
    interpret it correctly in the report.

    Parameters
    ----------
    module_dir        : the module's output subdirectory
                        (out_dir/{ckpt_hash}/{module}_{params_hash}/)
    module_name       : pixel | building | threshold | postproc | resolution
    eval_params_hash  : hash of the eval params affecting this result
    ckpt_hash         : hash of the checkpoint file (model identity)
    args              : parsed CLI args
    n_samples         : actual number of samples evaluated (after capping)
    training_run_id   : MLflow training run_id from the checkpoint
    """
    import json
    from datetime import datetime, timezone

    meta = {
        # ── identity ─────────────────────────────────────────────────────
        "module":               module_name,
        "checkpoint":           args.checkpoint,
        "checkpoint_hash":      ckpt_hash,
        "eval_params_hash":     eval_params_hash,

        # ── provenance ───────────────────────────────────────────────────
        "generated_at":         datetime.now(timezone.utc).isoformat(),
        "training_run_id":      training_run_id,

        # ── data params ───────────────────────────────────────────────────
        "eval_mode":            args.mode,
        "eval_cities":          getattr(args, "cities", None),
        "max_per_city":         getattr(args, "max_per_city", None),
        "n_samples_evaluated":  n_samples,

        # ── inference params ──────────────────────────────────────────────
        "resample":             True,

        # ── postprocessing params (only meaningful for building/postproc) ─
        "simplify_tolerance_m": args.simplify_tolerance,
        "min_area_m2":          args.min_area,
        "iou_threshold":        0.5,
    }

    meta_path = module_dir / f".meta_{module_name}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _update_report_selection(
    ckpt_dir: Path,
    module_name: str,
    eval_params_hash: str,
    args,
) -> None:
    """
    Maintain report_selection.yaml — the user-editable file that controls
    which eval run appears in the report for each module.

    After each module run:
      - The new run is set as the active selection (run: {hash})
      - The previous selection (if any) is kept as a commented-out line
        with a human-readable summary of its params, so the user can
        uncomment to roll back

    The YAML is intentionally simple — one key per module, one active
    run, previous runs as comments. No YAML library needed for writing
    since we control the exact format.

    Format:
      pixel:
        run: v1a2b3c4    # cities=['vienna'], max_per_city=200, 2026-09-08
        # run: ff91ab22  # cities=['vienna'], max_per_city=10,  2026-09-07
    """
    import json
    from datetime import datetime, timezone

    yaml_path = ckpt_dir / "report_selection.yaml"

    # Build human-readable comment for this run
    cities   = getattr(args, "cities", None) or ["all"]
    max_pc   = getattr(args, "max_per_city", None)
    ts       = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    simplify = getattr(args, "simplify_tolerance", None)
    min_area = getattr(args, "min_area", None)

    comment_parts = [f"cities={cities}", f"max_per_city={max_pc}"]
    if module_name == "building":
        comment_parts += [f"simplify={simplify}m", f"min_area={min_area}m²"]
    comment_parts.append(ts)
    comment = ", ".join(str(p) for p in comment_parts)

    new_active_line = f"    run: {eval_params_hash}    # {comment}"

    # Read existing YAML as plain text (we own the format)
    if yaml_path.exists():
        existing = yaml_path.read_text()
    else:
        existing = _report_selection_header()

    lines = existing.splitlines()
    new_lines = []
    in_module = False
    replaced  = False

    for line in lines:
        stripped = line.strip()

        # Detect module block start
        if stripped == f"{module_name}:":
            in_module = True
            new_lines.append(line)
            continue

        if in_module:
            # Active run line — demote to comment, insert new active
            if stripped.startswith("run:") and not stripped.startswith("# run:"):
                prev_hash    = stripped.split()[1]
                prev_comment = stripped[stripped.index("#"):] if "#" in stripped else ""
                new_lines.append(f"    {new_active_line.strip()}")
                new_lines.append(f"    # run: {prev_hash}    {prev_comment}".rstrip())
                replaced = True
                continue
            # Another module block starts — leave this one
            if stripped and not stripped.startswith("#") and stripped.endswith(":"):
                in_module = False

        new_lines.append(line)

    # Module not found in existing YAML — append it
    if not replaced:
        new_lines.append(f"\n{module_name}:")
        new_lines.append(f"    {new_active_line.strip()}")

    yaml_path.write_text("\n".join(new_lines) + "\n")


def _report_selection_header() -> str:
    return """# report_selection.yaml
#
# Controls which eval run appears in report.md for each module.
# Auto-updated after each evaluate.py run — safe to edit manually.
#
# To use a previous run: comment out the active 'run:' line and
# uncomment the one you want.
#
# Generate report:
#   python evaluate.py --report-only --checkpoint <path> --mode inria --out-dir outputs/evaluation
"""


def _read_report_selection(ckpt_dir: Path) -> dict[str, str | None]:
    """
    Parse report_selection.yaml and return {module_name: eval_params_hash}.
    Returns None for modules not yet evaluated.
    """
    yaml_path = ckpt_dir / "report_selection.yaml"
    if not yaml_path.exists():
        return {}

    selection = {}
    current_module = None

    for line in yaml_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Module block header: "pixel:" or "building:" etc.
        if stripped.endswith(":") and not stripped.startswith("run"):
            current_module = stripped[:-1]
            selection[current_module] = None
        # Active run line (not commented out)
        elif current_module and stripped.startswith("run:"):
            parts = stripped.split()
            if len(parts) >= 2:
                selection[current_module] = parts[1]

    return selection


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── resolve output directory ─────────────────────────────────────────
    # Two-level versioned directory structure:
    #   out_dir/{checkpoint_hash}/{module}_{eval_params_hash}/
    #
    # checkpoint_hash   — SHA256[:8] of the checkpoint file.
    #                     Guarantees model identity: all results under a
    #                     given checkpoint_hash come from the same weights.
    #
    # eval_params_hash  — SHA256[:8] of the eval params affecting each module.
    #                     Guarantees result identity: different param
    #                     combinations (cities, max_per_city, min_area, etc.)
    #                     produce separate subdirectories and never overwrite.
    #
    # Example:
    #   outputs/evaluation/
    #   └── abc123de/               ← checkpoint hash (model identity)
    #       ├── pixel_v1a2b3c4/     ← max-per-city=200, cities=vienna
    #       ├── pixel_ff91ab22/     ← max-per-city=50,  cities=vienna (sanity)
    #       ├── building_d5e6f7a8/  ← min-area=10
    #       ├── report_selection.yaml  ← controls which runs appear in report
    #       └── report.md
    ckpt_hash = _checkpoint_hash(args.checkpoint)
    ckpt_dir  = Path(args.out_dir) / ckpt_hash
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── report-only mode ──────────────────────────────────────────────────
    # Reads report_selection.yaml to determine which eval run to use per
    # module, then generates report.md from those runs.
    # Default (no selection file or null entry): most recent run per module.
    if args.report_only:
        from evaluation import visualisation
        print(f"\nCheckpoint hash : {ckpt_hash}")
        print(f"Checkpoint dir  : {ckpt_dir}")

        if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
            print(f"  No results found in {ckpt_dir}.")
            print(f"  Run evaluation modules first with --out-dir {Path(args.out_dir)}")
            return

        # Read report_selection.yaml to know which run per module to include
        selection = _read_report_selection(ckpt_dir)
        print(f"  Selection: {selection or 'auto (most recent per module)'}")

        visualisation.generate_report(
            ckpt_dir=ckpt_dir,
            mode=args.mode,
            checkpoint_path=args.checkpoint,
            selection=selection,
        )
        print("Done.")
        return

    # ── load model ────────────────────────────────────────────────────────
    print(f"\nLoading model from: {args.checkpoint}")
    from api.inference import SegmentationInference
    model = SegmentationInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device="auto",
    )

    # ── resolve modules to run ────────────────────────────────────────────
    if args.eval:
        modules = args.eval
    else:
        modules = [m for m in AVAILABLE_MODULES
                   if m not in CUSTOM_ONLY_MODULES or args.mode == "custom"]

    # Validate mode compatibility
    for m in modules:
        if m in CUSTOM_ONLY_MODULES and args.mode != "custom":
            print(f"Warning: module '{m}' requires --mode custom — skipping.")
            modules = [x for x in modules if x != m]

    print(f"Mode              : {args.mode}")
    print(f"Checkpoint hash   : {ckpt_hash}")
    print(f"Modules           : {modules}")
    print(f"Checkpoint dir    : {ckpt_dir}")
    print(f"MLflow            : {'disabled (--no-mlflow)' if args.no_mlflow else 'enabled'}\n")

    # ── resolve cities for inria mode (needed for MLflow params) ─────────
    if args.mode == "inria":
        if args.cities:
            cities = args.cities
        else:
            cities = sorted([
                d.name for d in Path(args.patches).iterdir() if d.is_dir()
            ])
    else:
        cities = ["custom"]

    # ── read training run_id from checkpoint (for metadata + MLflow) ────────
    training_run_id = None
    try:
        ckpt_data = torch.load(args.checkpoint, map_location="cpu")
        training_run_id = ckpt_data.get("mlflow_run_id")
    except Exception:
        pass

    # ── MLflow eval run setup ─────────────────────────────────────────────
    # A NEW run is created (separate from the training run) and tagged with
    # the training run_id for traceability. See _resolve_mlflow_run() for
    # the tagging strategy and _log_to_mlflow() for metric naming conventions.
    mlflow_run = None
    if not args.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

        run_name, tags = _resolve_mlflow_run(args.checkpoint, cfg)

        # Start the eval run — stays open for the duration of main()
        # so all modules log into the same run.
        mlflow_run = mlflow.start_run(run_name=run_name, tags=tags)

        # Log eval-level params — what data and setup was used
        mlflow.log_params({
            "eval_checkpoint_hash": ckpt_hash,
            "eval_mode":            args.mode,
            "eval_cities":          str(cities),
            "eval_modules":         str(modules),
            "eval_max_per_city":    str(args.max_per_city),
            "eval_simplify_tol":    args.simplify_tolerance,
            "eval_min_area":        args.min_area,
            "eval_checkpoint":      args.checkpoint,
        })
        print(f"MLflow eval run: {mlflow_run.info.run_id}")
        if "training_run_id" in tags:
            print(f"  linked to training run: {tags['training_run_id']}")

    # ── run modules ───────────────────────────────────────────────────────
    # Samples is a generator — each module that needs it gets a fresh one
    # (generators can only be consumed once, so we rebuild per module).
    # Results dict is captured from each module and passed to _log_to_mlflow.

    module_results: dict[str, dict] = {}

    for module_name in modules:
        print(f"\n{'='*60}")
        print(f"  Module: {module_name.upper()}")
        print(f"{'='*60}")

        # ── resolve module output directory ───────────────────────────────
        # Two-level: ckpt_dir/{module}_{eval_params_hash}/
        # Different eval params → different subdir → no overwrite.
        params_hash = _eval_params_hash(module_name, args)
        module_dir  = ckpt_dir / f"{module_name}_{params_hash}"
        module_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Eval params hash : {params_hash}")
        print(f"  Output dir       : {module_dir}")

        # Rebuild sample iterator for each module
        # (generators can only be consumed once)
        samples = load_samples(args, cfg)

        if module_name == "pixel":
            from evaluation import metrics_pixel
            results = metrics_pixel.run(
                samples, model, module_dir,
                postprocess=True,
                simplify_tolerance_m=args.simplify_tolerance,
                min_area_m2=args.min_area,
            )

        elif module_name == "building":
            from evaluation import metrics_building
            results = metrics_building.run(
                samples, model, module_dir,
                simplify_tolerance_m=args.simplify_tolerance,
                min_area_m2=args.min_area,
            )

        elif module_name == "threshold":
            from evaluation import threshold_analysis
            results = threshold_analysis.run(samples, model, module_dir)

        elif module_name == "postproc":
            from evaluation import postproc_sensitivity
            results = postproc_sensitivity.run(samples, model, module_dir)

        elif module_name == "resolution":
            from evaluation import resolution_robustness
            results = resolution_robustness.run(samples, model, module_dir)

        else:
            results = {}

        module_results[module_name] = results

        # Count samples evaluated (best effort from results dict)
        n_samples = 0
        if "per_city" in results:
            for city_data in results["per_city"].values():
                n_samples += city_data.get("n_samples", 0)

        # Write full provenance record alongside output
        _write_module_metadata(
            module_dir=module_dir,
            module_name=module_name,
            eval_params_hash=params_hash,
            ckpt_hash=ckpt_hash,
            args=args,
            n_samples=n_samples,
            training_run_id=training_run_id,
        )

        # Update report_selection.yaml — sets this run as active for this
        # module, demotes previous run to commented-out line.
        _update_report_selection(ckpt_dir, module_name, params_hash, args)

        # Log to MLflow immediately — partial results captured even if
        # a later module crashes.
        if mlflow_run is not None:
            _log_to_mlflow(mlflow_run, module_name, results, module_dir)

    # ── qualitative grid ──────────────────────────────────────────────────
    # Grid lives in ckpt_dir/qualitative/ (shared across module runs —
    # it's a visual aid, not a metrics output, so no per-params versioning)
    print(f"\n{'='*60}")
    print(f"  Module: QUALITATIVE GRID")
    print(f"{'='*60}")

    from evaluation import visualisation
    from api.vectorize import vectorize, polygons_to_mask

    qual_dir = ckpt_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)
    city_samples: dict[str, list] = {}

    for sample in load_samples(args, cfg):
        mask, _ = model.predict(
            sample.image,
            input_resolution=sample.resolution,
            resample=True,
        )
        geojson    = vectorize(mask, resolution=sample.resolution,
                               simplify_tolerance_m=args.simplify_tolerance,
                               min_area_m2=args.min_area)
        H, W       = sample.image.shape[:2]
        clean_mask = polygons_to_mask(geojson, height=H, width=W)

        city_samples.setdefault(sample.city, []).append({
            "image":      sample.image,
            "gt_mask":    sample.gt_mask,
            "raw_mask":   mask,
            "clean_mask": clean_mask,
            "name":       sample.name,
        })

    for city, s_list in city_samples.items():
        visualisation.save_prediction_grid(s_list, qual_dir, city=city)

    # ── report ────────────────────────────────────────────────────────────
    # Reads report_selection.yaml to pick which eval run per module to include.
    # Falls back to most recent run per module if selection file is missing.
    selection = _read_report_selection(ckpt_dir)
    visualisation.generate_report(
        ckpt_dir=ckpt_dir,
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        selection=selection,
    )

    # Log qualitative grids and report as artifacts
    if mlflow_run is not None:
        import mlflow
        for city in city_samples:
            grid_path = qual_dir / f"predictions_grid_{city}.png"
            if grid_path.exists():
                mlflow.log_artifact(str(grid_path),
                                    artifact_path="evaluation/qualitative")
        report_path = ckpt_dir / "report.md"
        if report_path.exists():
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")

        mlflow.end_run()
        print(f"\nMLflow eval run complete: {mlflow_run.info.run_id}")
        print(f"  View: mlflow ui --backend-store-uri {cfg['mlflow']['tracking_uri']} --port 5000")

    print(f"\n{'='*60}")
    print(f"  Evaluation complete.")
    print(f"  Results : {ckpt_dir}")
    print(f"  Report  : {ckpt_dir / 'report.md'}")
    print(f"  Selection: {ckpt_dir / 'report_selection.yaml'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()