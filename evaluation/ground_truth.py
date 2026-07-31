"""
evaluation/ground_truth.py

Ground truth loading for two data sources:

  Inria mode — loads pre-patched 512x512 PNG masks from the patches directory.
               GT is already a rasterized binary mask.

  Custom mode — loads SWISSIMAGE GeoTIFF tiles + swissTLM3D building footprints.
                GT is rasterized on-the-fly from vector polygons using the
                tile's affine transform so pixel alignment is exact.

Both modes yield (image_np, gt_mask_np, metadata_dict) tuples via a common
iterator interface so the evaluation modules don't need to know which mode
is active.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

try:
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


# ── sample dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """
    A single evaluation unit: image + GT mask + metadata.

    Attributes
    ----------
    image: (H, W, 3) uint8 RGB
    gt_mask: (H, W) bool - True = building
    name: identifier (tile filename stem or patch name)
    city: city/dataset label for grouping results
    resolution: pixel resolution in m/px, or None if unknown
    """
    image: np.ndarray
    gt_mask: np.ndarray
    name: str
    city: str
    resolution: Optional[float] = None
    extra: dict = field(default_factory=dict)


# ── Inria loader ──────────────────────────────────────────────────────────────

def load_inria_samples(
    patches_dir: str | Path,
    cities: list[str],
    max_per_city: Optional[int] = None,
) -> Iterator[EvalSample]:
    """
    Yield EvalSample objects from pre-patched Inria PNG files.

    Parameters
    ----------
    patches_dir: root directory with city subdirectories
    cities: list of city names to include
    max_per_city: cap samples per city (useful for fast runs)
    """
    from PIL import Image as PILImage

    patches_dir = Path(patches_dir)

    for city in cities:
        img_dir  = patches_dir / city / "images"
        mask_dir = patches_dir / city / "masks"

        if not img_dir.exists():
            warnings.warn(f"City directory not found: {img_dir} - skipping.")
            continue

        img_paths = sorted(img_dir.glob("*.png"))
        if max_per_city:
            img_paths = img_paths[:max_per_city]

        for img_path in img_paths:
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                continue

            image   = np.array(PILImage.open(img_path).convert("RGB"))
            gt_mask = np.array(PILImage.open(mask_path).convert("L")) > 127

            yield EvalSample(
                image=image,
                gt_mask=gt_mask,
                name=img_path.stem,
                city=city,
                resolution=0.3, # Inria dataset resolution is 0.3m/px
            )


# ── swisstopo loader ──────────────────────────────────────────────────────────

def _drop_z(geom):
    """Remove Z coordinate from a shapely geometry (2D required by rasterio)."""
    from shapely.ops import transform
    return transform(lambda x, y, *z: (x, y), geom)


def _rasterize_buildings(
    buildings_gdf,
    height: int,
    width: int,
    transform,
) -> np.ndarray:
    """
    Rasterize a GeoDataFrame of building polygons to a binary mask.

    Uses rasterio.features.rasterize with the tile's affine transform
    so pixel alignment is exact.

    Returns (H, W) bool array.
    """
    if buildings_gdf.empty:
        return np.zeros((height, width), dtype=bool)

    shapes = []
    for geom in buildings_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        # Drop Z coordinate - rasterio requires 2D geometries
        geom_2d = _drop_z(geom)
        # Flatten MultiPolygon to individual polygons
        if geom_2d.geom_type == "MultiPolygon":
            for part in geom_2d.geoms:
                shapes.append((part.__geo_interface__, 1))
        else:
            shapes.append((geom_2d.__geo_interface__, 1))

    if not shapes:
        return np.zeros((height, width), dtype=bool)

    mask = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False, # strict: only pixels whose centre is inside
    )
    return mask.astype(bool)


def load_swissimage_samples(
    images_dir: str | Path,
    gdb_path: str | Path,
    gdb_layer: str = "TLM_GEBAEUDE_FOOTPRINT",
    max_samples: Optional[int] = None,
) -> Iterator[EvalSample]:
    """
    Yield EvalSample objects from SWISSIMAGE GeoTIFF tiles with GT from swissTLM3D.

    For each tile:
      1. Open GeoTIFF -> extract CRS, affine transform, bounds, image array
      2. Load swissTLM3D buildings clipped to tile bounds (bbox)
      3. Rasterize buildings onto the tile's pixel grid
      4. Yield EvalSample

    Parameters
    ----------
    images_dir: directory containing SWISSIMAGE .tif tiles
    gdb_path: path to swissTLM3D_YYYY_LV95_LN02.gdb
    gdb_layer: layer name for building footprints
    max_samples: cap total samples (useful for fast runs)

    Notes
    -----
    Both SWISSIMAGE and swissTLM3D use LV95 (EPSG:2056) - no reprojection needed.
    The function asserts the CRS is projected (metres) before reading resolution.
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for custom mode. pip install rasterio")
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for custom mode. pip install geopandas")

    images_dir = Path(images_dir)
    tif_paths  = sorted(images_dir.glob("*.tif")) + sorted(images_dir.glob("*.tiff"))

    if not tif_paths:
        raise FileNotFoundError(f"No .tif files found in {images_dir}")

    if max_samples:
        tif_paths = tif_paths[:max_samples]

    for tif_path in tif_paths:
        with rasterio.open(tif_path) as src:
            # ── validate CRS ────────────────────────────────────────────
            if src.crs is None:
                warnings.warn(f"{tif_path.name}: no CRS found - skipping.")
                continue
            if not src.crs.is_projected:
                warnings.warn(
                    f"{tif_path.name}: CRS is geographic (degrees), not projected. "
                    "Resolution in metres cannot be determined - skipping."
                )
                continue

            # ── image metadata ───────────────────────────────────────────
            res_x, res_y = src.res
            resolution = float((res_x + res_y) / 2)
            affine_tf = src.transform
            bounds = src.bounds
            H, W = src.height, src.width

            # ── read image ───────────────────────────────────────────────
            arr = src.read() # (C, H, W)

        image = _bands_to_rgb_uint8(arr)

        # ── load GT buildings clipped to tile bbox ────────────────────
        bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)
        try:
            buildings = gpd.read_file(str(gdb_path), layer=gdb_layer, bbox=bbox)
        except Exception as e:
            warnings.warn(f"{tif_path.name}: failed to load GT - {e}. Skipping.")
            continue

        gt_mask = _rasterize_buildings(buildings, H, W, affine_tf)

        yield EvalSample(
            image=image,
            gt_mask=gt_mask,
            name=tif_path.stem,
            city="swisstopo",
            resolution=resolution,
            extra={
                "bounds": bounds,
                "crs": str(src.crs),
                "n_gt_buildings": len(buildings),
            },
        )


def _bands_to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """(C, H, W) any dtype -> (H, W, 3) uint8."""
    C = arr.shape[0]
    if C == 1:
        arr = np.concatenate([arr, arr, arr], axis=0)
    elif C > 3:
        arr = arr[:3]

    if arr.dtype == np.uint16:
        arr = (arr.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        for c in range(arr.shape[0]):
            mn, mx = arr[c].min(), arr[c].max()
            if mx > mn:
                arr[c] = (arr[c] - mn) / (mx - mn) * 255.0
        arr = arr.astype(np.uint8)

    return arr.transpose(1, 2, 0)