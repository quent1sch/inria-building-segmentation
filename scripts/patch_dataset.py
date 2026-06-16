"""
scripts/patch_dataset.py

Slice the large Inria 5000×5000 GeoTIFF tiles into fixed-size patches
suitable for training.  Patches that contain >1% building pixels are
kept; near-empty patches (pure background with no context) are dropped
to reduce class imbalance and disk usage.

Usage
-----
python scripts/patch_dataset.py \
    --src  data/raw/train \
    --dst  data/patches \
    --size 512 \
    --stride 256 \
    --min-building-ratio 0.01
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import rasterio
    USE_RASTERIO = True
except ImportError:
    USE_RASTERIO = False


# ── helpers ──────────────────────────────────────────────────────────────────

def read_image(path: Path) -> np.ndarray:
    """Read an RGB image as HWC uint8 numpy array."""
    if USE_RASTERIO and path.suffix.lower() in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read()          # (C, H, W)
            arr = np.moveaxis(arr, 0, -1)  # → (H, W, C)
            if arr.dtype != np.uint8:
                arr = (arr / arr.max() * 255).astype(np.uint8)
            return arr[:, :, :3]     # drop alpha if present
    else:
        return np.array(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    """Read a binary mask as HW uint8 (0 or 255)."""
    if USE_RASTERIO and path.suffix.lower() in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read(1)         # first band
            if arr.max() > 1:
                arr = (arr > 127).astype(np.uint8) * 255
            else:
                arr = (arr > 0).astype(np.uint8) * 255
            return arr
    else:
        img = Image.open(path).convert("L")
        arr = np.array(img)
        return (arr > 127).astype(np.uint8) * 255


def extract_city(name: str) -> str:
    """'austin1' → 'austin', 'tyrol-w3' → 'tyrol-w'."""
    for city in ["tyrol-w", "tyrol-e", "austin", "chicago", "kitsap", "vienna",
                 "bellingham", "bloomington", "innsbruck", "sfo"]:
        if name.startswith(city):
            return city
    return name.rstrip("0123456789")


# ── main ─────────────────────────────────────────────────────────────────────

def patch_tile(
    img: np.ndarray,
    mask: np.ndarray,
    tile_name: str,
    out_img_dir: Path,
    out_mask_dir: Path,
    size: int,
    stride: int,
    min_building_ratio: float,
) -> int:
    """Extract patches from one tile; return number of saved patches."""
    H, W = img.shape[:2]
    saved = 0

    for y in range(0, H - size + 1, stride):
        for x in range(0, W - size + 1, stride):
            patch_img = img[y:y + size, x:x + size]
            patch_mask = mask[y:y + size, x:x + size]

            building_ratio = (patch_mask > 127).mean()
            if building_ratio < min_building_ratio:
                continue

            stem = f"{tile_name}_y{y:04d}_x{x:04d}"
            Image.fromarray(patch_img).save(out_img_dir / f"{stem}.png")
            Image.fromarray(patch_mask).save(out_mask_dir / f"{stem}.png")
            saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="Patch Inria tiles for training")
    parser.add_argument("--src", required=True, help="Path to data/raw/train")
    parser.add_argument("--dst", required=True, help="Output directory for patches")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-building-ratio", type=float, default=0.01,
                        help="Minimum fraction of building pixels to keep a patch")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    img_src = src / "images"
    gt_src = src / "gt"

    assert img_src.exists(), f"Images dir not found: {img_src}"
    assert gt_src.exists(), f"GT dir not found: {gt_src}"

    tile_paths = sorted(img_src.glob("*.tif"))
    print(f"Found {len(tile_paths)} tiles in {img_src}")

    total_patches = 0

    for img_path in tqdm(tile_paths, desc="Tiling"):
        tile_name = img_path.stem
        city = extract_city(tile_name)

        out_img_dir = dst / city / "images"
        out_mask_dir = dst / city / "masks"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        gt_path = gt_src / img_path.name
        if not gt_path.exists():
            print(f"  ⚠️  GT not found for {tile_name}, skipping.")
            continue

        img = read_image(img_path)
        mask = read_mask(gt_path)

        n = patch_tile(
            img, mask, tile_name,
            out_img_dir, out_mask_dir,
            size=args.size,
            stride=args.stride,
            min_building_ratio=args.min_building_ratio,
        )
        total_patches += n

    print(f"\n✅  Done — {total_patches} patches saved to '{dst}'")
    print(f"   City subdirectories: {[d.name for d in sorted(dst.iterdir()) if d.is_dir()]}")


if __name__ == "__main__":
    main()
