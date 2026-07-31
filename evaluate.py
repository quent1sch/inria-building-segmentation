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
 
  # Inria — pixel metrics only (fast)
  python evaluate.py
      --checkpoint checkpoints/best_model.pth
      --mode inria
      --patches data/patches
      --cities vienna
      --eval pixel
 
  # Swisstopo custom — pixel + building + resolution
  python evaluate.py
      --checkpoint checkpoints/best_model.pth
      --mode custom
      --images path/to/swissimage_tiles/
      --gt path/to/swissTLM3D_2026_LV95_LN02.gdb
      --max-samples 3
      --eval resolution

  # Other...
  # python evaluate.py --checkpoint checkpoints/best_model.pth --mode custom --images data/swisstopo/SWISSIMAGE/ --gt data/swisstopo/swissTLM3D/swissTLM3D_2026_LV95_LN02.gdb --max-samples 1 --eval resolution
 
  # Multiple cities, limit samples per city for speed
  python evaluate.py
      --checkpoint checkpoints/best_model.pth
      --mode inria
      --patches data/patches
      --cities vienna austin
      --max-per-city 50
      --eval pixel threshold
"""
 
import argparse
import sys
from pathlib import Path
 
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

def main():
    args = parse_args()
 
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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
 
    print(f"Mode: {args.mode}")
    print(f"Modules: {modules}")
    print(f"Out dir: {args.out_dir}\n")
 
    out_dir = Path(args.out_dir)


    # ── run modules ───────────────────────────────────────────────────────
    # Samples is a generator. Each module that needs it gets a fresh one
    # (generators can only be consumed once, so we rebuild per module)
 
    for module_name in modules:
        print(f"\n{'='*60}")
        print(f"  Module: {module_name.upper()}")
        print(f"{'='*60}")
 
        # Rebuild sample iterator for each module
        samples = load_samples(args, cfg)
        print("sample interator built")
 
        if module_name == "pixel":
            print(f"starting {module_name.upper()} evaluation module")
            from evaluation import metrics_pixel
            metrics_pixel.run(
                samples, model, out_dir,
                postprocess=True,
                simplify_tolerance_m=args.simplify_tolerance,
                min_area_m2=args.min_area,
            )
 
        elif module_name == "building":
            from evaluation import metrics_building
            metrics_building.run(
                samples, model, out_dir,
                simplify_tolerance_m=args.simplify_tolerance,
                min_area_m2=args.min_area,
            )

        elif module_name == "threshold":
            from evaluation import threshold_analysis
            threshold_analysis.run(samples, model, out_dir)

        elif module_name == "postproc":
            from evaluation import postproc_sensitivity
            postproc_sensitivity.run(samples, model, out_dir)

        elif module_name == "resolution":
            from evaluation import resolution_robustness
            resolution_robustness.run(samples, model, out_dir)
    
    # ── qualitative grid ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Module: QUALITATIVE GRID")
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
 
    print(f"\n{'='*60}")
    print(f"  Evaluation complete. Results in: {out_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()