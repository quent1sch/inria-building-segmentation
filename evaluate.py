"""
evaluate.py
 
Evaluation pipeline for the Inria building segmentation model.
 
Runs any combination of evaluation modules independently so each can be
executed separately on CPU in acceptable time.
 
Modes
-----
  inria   — Inria Aerial Image Labeling dataset test tiles (patched PNG crops)
  custom  — SWISSIMAGE GeoTIFF tiles + swissTLM3D vector ground truth
 
Evaluation modules
------------------
  pixel       Pixel-level IoU, Dice, Precision, Recall (raw vs clean)
  building    Object-level detection metrics (polygon matching)
  threshold   PR curve, ROC, optimal threshold sweep
  postproc    Postprocessing parameter sensitivity (simplify + min_area)
  resolution  Resolution robustness (native / resampled / simulated coarse)
              — custom mode only, requires images with known resolution
 
Usage examples
--------------
  # Inria — all modules
  python evaluate.py
      --checkpoint checkpoints/best_model.pth
      --mode inria
      --patches data/patches
      --cities vienna
      --out-dir outputs/evaluation
 
  # Inria — pixel metrics only (fast)
  python evaluate.py
      --checkpoint checkpoints/best_model.pth
      --mode inria
      --patches data/patches
      --cities vienna
      --eval pixel
      --out-dir outputs/evaluation
 
  # Swisstopo custom — pixel + building + resolution
  python evaluate.py
      --checkpoint checkpoints/best_model.pth
      --mode custom
      --images path/to/swissimage_tiles/
      --gt path/to/swissTLM3D_2026_LV95_LN02.gdb
      --max-samples 3
      --eval resolution
      --out-dir outputs/evaluation

  # Other...
  python evaluate.py 
        --checkpoint checkpoints/best_model.pth 
        --mode custom 
        --images data/swisstopo/SWISSIMAGE/ 
        --gt data/swisstopo/swissTLM3D/swissTLM3D_2026_LV95_LN02.gdb 
        --max-samples 1 
        --eval resolution
        --out-dir outputs/evaluation
 
  # Multiple cities, limit samples per city for speed
  python evaluate.py
      --checkpoint checkpoints/best_model.pth
      --mode inria
      --patches data/patches
      --cities vienna austin
      --max-per-city 50
      --eval pixel threshold
      --out-dir outputs/evaluation
  
      
  # Skip MLflow logging (quick local run)
  python evaluate.py 
       --checkpoint checkpoints/best_model.pth 
       --mode inria 
       --patches data/patches 
       --eval pixel 
       --no-mlflow
       --out-dir outputs/evaluation

View results
------------
# MLflow UI — see both training and eval runs side by side
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

# Quick check without MLflow
cat outputs/evaluation/metrics_pixel.json | python3 -m json.tool
cat outputs/evaluation/report.md

MLflow integration
------------------
Evaluation results are logged to a NEW MLflow run (separate from the
training run) and linked back to it via a tag:

  Training run  [run_id: abc-123]
    params: encoder, lr, epochs, ...
    metrics: train_loss, val_iou, ... (per epoch)
    artifacts: best_model.pth

  Evaluation run  [run_name: eval-inria-20260901-143022]
    tag: training_run_id = abc-123      ← links to training run
    tag: checkpoint = checkpoints/best_model.pth
    tag: model_epoch = 42
    params: eval_mode, eval_cities, eval_modules, ...
    metrics:
      eval.pixel.vienna.iou_raw         ← per city
      eval.pixel.overall.iou_raw        ← aggregated
      eval.pixel.overall.iou_clean
      eval.building.overall.precision_raw
      eval.threshold.optimal_threshold
      eval.postproc.simplify.0_5.building_f1
      eval.resolution.resampled.iou
    artifacts: metrics_pixel.json, threshold_analysis.png, report.md, ...

This design keeps training and evaluation cleanly separated in MLflow
while maintaining full traceability: every eval run tags the training
run_id so you always know which model produced a given result.

Use --no-mlflow to skip logging (fast local runs, CI, no tracking server).
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
        #InriaDataset,
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


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── load model ────────────────────────────────────────────────────────
    print(f"\\nLoading model from: {args.checkpoint}")
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

    print(f"Mode    : {args.mode}")
    print(f"Modules : {modules}")
    print(f"Out dir : {args.out_dir}")
    print(f"MLflow  : {'disabled (--no-mlflow)' if args.no_mlflow else 'enabled'}\\n")

    out_dir = Path(args.out_dir)

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
            "eval_mode":           args.mode,
            "eval_cities":         str(cities),
            "eval_modules":        str(modules),
            "eval_max_per_city":   str(args.max_per_city),
            "eval_simplify_tol":   args.simplify_tolerance,
            "eval_min_area":       args.min_area,
            "eval_checkpoint":     args.checkpoint,
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
        print(f"\\n{'='*60}")
        print(f"  Module: {module_name.upper()}")
        print(f"{'='*60}")

        # Rebuild sample iterator for each module
        samples = load_samples(args, cfg)

        if module_name == "pixel":
            from evaluation import metrics_pixel
            results = metrics_pixel.run(
                samples, model, out_dir,
                postprocess=True,
                simplify_tolerance_m=args.simplify_tolerance,
                min_area_m2=args.min_area,
            )

        elif module_name == "building":
            from evaluation import metrics_building
            results = metrics_building.run(
                samples, model, out_dir,
                simplify_tolerance_m=args.simplify_tolerance,
                min_area_m2=args.min_area,
            )

        elif module_name == "threshold":
            from evaluation import threshold_analysis
            results = threshold_analysis.run(samples, model, out_dir)

        elif module_name == "postproc":
            from evaluation import postproc_sensitivity
            results = postproc_sensitivity.run(samples, model, out_dir)

        elif module_name == "resolution":
            from evaluation import resolution_robustness
            results = resolution_robustness.run(samples, model, out_dir)

        else:
            results = {}

        module_results[module_name] = results

        # Log this module's results to MLflow immediately after it completes.
        # Logging per-module (not all at end) means partial results are
        # captured even if a later module crashes.
        if mlflow_run is not None:
            _log_to_mlflow(mlflow_run, module_name, results, out_dir)

    # ── qualitative grid ──────────────────────────────────────────────────
    print(f"\\n{'='*60}")
    print(f"  Module: QUALITATIVE GRID")
    print(f"{'='*60}")

    from evaluation import visualisation
    from api.vectorize import vectorize, polygons_to_mask

    qual_dir = out_dir / "qualitative"
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
    visualisation.generate_report(
        out_dir=out_dir,
        mode=args.mode,
        eval_modules=modules,
        checkpoint_path=args.checkpoint,
    )

    # Log qualitative grids and report as artifacts
    if mlflow_run is not None:
        import mlflow
        for city in city_samples:
            grid_path = qual_dir / f"predictions_grid_{city}.png"
            if grid_path.exists():
                mlflow.log_artifact(str(grid_path),
                                    artifact_path="evaluation/qualitative")
        report_path = out_dir / "report.md"
        if report_path.exists():
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")

        mlflow.end_run()
        print(f"\\nMLflow eval run complete: {mlflow_run.info.run_id}")
        print(f"  View: mlflow ui --backend-store-uri {cfg['mlflow']['tracking_uri']} --port 5000")

    print(f"\\n{'='*60}")
    print(f"  Evaluation complete. Results in: {out_dir}")
    print(f"{'='*60}\\n")


if __name__ == "__main__":
    main()