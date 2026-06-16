"""
scripts/download_inria.py

Helper to remind users of the manual download steps for the Inria dataset.
The dataset requires free registration; direct programmatic download is not
officially supported, so this script validates the expected folder structure
and prints clear instructions.
"""

import argparse
import sys
from pathlib import Path


INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════╗
║          Inria Aerial Image Labeling Dataset — Download Guide        ║
╚══════════════════════════════════════════════════════════════════════╝

1. Register (free) at:
   https://project.inria.fr/aerialimagelabeling/download/

2. Download the archive:  AerialImageDataset.zip  (~20 GB)

3. Extract to:  {out_dir}/

   Expected layout after extraction:
   {out_dir}/
   ├── train/
   │   ├── images/         ← RGB GeoTIFFs  (austin1.tif … tyrol-w36.tif)
   │   └── gt/             ← Binary masks  (austin1.tif … tyrol-w36.tif)
   └── test/
       └── images/

4. Run patching:
   python scripts/patch_dataset.py \\
       --src {out_dir}/train \\
       --dst data/patches \\
       --size 512 \\
       --stride 256

Training cities : austin, chicago, kitsap, tyrol-w, vienna
Test  cities    : bellingham, bloomington, innsbruck, sfo, tyrol-e
"""


EXPECTED_TRAIN_CITIES = [
    "austin", "chicago", "kitsap", "tyrol-w", "vienna"
]


def check_structure(out_dir: Path) -> bool:
    """Return True if the expected directory structure is present."""
    images_dir = out_dir / "train" / "images"
    gt_dir = out_dir / "train" / "gt"

    if not images_dir.exists() or not gt_dir.exists():
        return False

    tif_files = list(images_dir.glob("*.tif"))
    return len(tif_files) > 0


def main():
    parser = argparse.ArgumentParser(description="Inria dataset download guide")
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw",
        help="Target directory where the dataset should be placed (default: data/raw)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(INSTRUCTIONS.format(out_dir=out_dir))

    if check_structure(out_dir):
        tif_count = len(list((out_dir / "train" / "images").glob("*.tif")))
        print(f"Dataset found at '{out_dir}' — {tif_count} training tiles detected.")
        print("Run patching next:")
        print(f"python scripts/patch_dataset.py --src {out_dir}/train --dst data/patches")
    else:
        print(f"Dataset NOT found at '{out_dir}/train/images/'.")
        print("Please follow the instructions above, then re-run this script to verify.")
        sys.exit(1)


if __name__ == "__main__":
    main()
