"""
tests/conftest.py

Shared pytest fixtures used across unit tests.

All fixtures produce synthetic data — no model weights, no DB, no filesystem
access required. Tests that import these fixtures run in milliseconds.

Naming conventions
------------------
  binary_mask_*   bool (H, W) arrays representing segmentation masks
  rgb_image_*     uint8 (H, W, 3) arrays representing aerial images
  geojson_*       GeoJSON FeatureCollection dicts
"""

import numpy as np
import pytest


# ── image fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def rgb_image_small():
    """64×64 RGB image — solid grey background."""
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def rgb_image_with_buildings():
    """
    128×128 RGB image with two visually distinct regions:
      - Background: dark grey (50, 50, 50)
      - Buildings:  light grey (200, 200, 200) in a 30×30 block at (20,20)
                    and a 20×20 block at (80,80)
    """
    img = np.full((128, 128, 3), 50, dtype=np.uint8)
    img[20:50, 20:50] = 200   # building 1: 30×30
    img[80:100, 80:100] = 200  # building 2: 20×20
    return img


@pytest.fixture
def uint16_image_small():
    """64×64 3-band uint16 image in CHW layout (as rasterio returns it)."""
    arr = np.random.randint(0, 65535, (3, 64, 64), dtype=np.uint16)
    return arr


@pytest.fixture
def multiband_image_4band():
    """64×64 4-band uint8 image in CHW layout (RGBN — common in aerial imagery)."""
    return np.random.randint(0, 255, (4, 64, 64), dtype=np.uint8)


@pytest.fixture
def grayscale_image():
    """64×64 single-band uint8 image in CHW layout."""
    return np.random.randint(0, 255, (1, 64, 64), dtype=np.uint8)


# ── mask fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def empty_mask():
    """128×128 all-False mask — no buildings detected."""
    return np.zeros((128, 128), dtype=bool)


@pytest.fixture
def full_mask():
    """128×128 all-True mask — everything is a building (edge case)."""
    return np.ones((128, 128), dtype=bool)


@pytest.fixture
def binary_mask_two_buildings():
    """
    128×128 binary mask with two rectangular building blobs:
      - Building 1: rows 20-49, cols 20-49  (30×30 = 900 px)
      - Building 2: rows 80-99, cols 80-99  (20×20 = 400 px)
    """
    mask = np.zeros((128, 128), dtype=bool)
    mask[20:50, 20:50] = True   # building 1
    mask[80:100, 80:100] = True  # building 2
    return mask


@pytest.fixture
def binary_mask_one_building():
    """128×128 mask with a single 40×40 building blob."""
    mask = np.zeros((128, 128), dtype=bool)
    mask[10:50, 10:50] = True
    return mask


@pytest.fixture
def perfect_pred_mask(binary_mask_two_buildings):
    """Prediction mask identical to GT — IoU=1.0, perfect score."""
    return binary_mask_two_buildings.copy()


@pytest.fixture
def shifted_pred_mask():
    """
    128×128 mask with buildings shifted 5 pixels right — partial overlap with GT.
    Used to test that metrics are < 1.0 but > 0.0.
    """
    mask = np.zeros((128, 128), dtype=bool)
    mask[20:50, 25:55] = True   # shifted building 1
    mask[80:100, 85:105] = True  # shifted building 2 (partially out of bounds OK)
    return mask


@pytest.fixture
def no_overlap_pred_mask():
    """Prediction mask with buildings in completely different locations from GT."""
    mask = np.zeros((128, 128), dtype=bool)
    mask[0:10, 0:10] = True   # tiny blob far from GT buildings
    return mask


# ── GeoJSON fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def simple_geojson():
    """
    Minimal valid GeoJSON FeatureCollection with one rectangular polygon.
    Coordinates are in pixel space (col, row).
    """
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[20, 20], [50, 20], [50, 50], [20, 50], [20, 20]]
                    ],
                },
                "properties": {"area_px": 900.0},
            }
        ],
        "metadata": {"n_buildings": 1},
    }


@pytest.fixture
def empty_geojson():
    """GeoJSON FeatureCollection with no features."""
    return {
        "type": "FeatureCollection",
        "features": [],
        "metadata": {"n_buildings": 0},
    }


# ── spatial ref fixture ───────────────────────────────────────────────────────

@pytest.fixture
def mock_spatial_ref():
    """
    Minimal SpatialRef-like object for tests that need spatial metadata
    without a real rasterio CRS.

    Uses a simple identity-like transform at SWISSIMAGE scale:
      - Resolution: 0.1m/px
      - Origin: (2600000, 1200000) — approximate LV95 coordinates for Switzerland
      - CRS: mocked with .to_epsg() returning 2056

    We use a plain namespace object rather than importing SpatialRef to
    keep tests decoupled from the inference module.
    """
    from types import SimpleNamespace
    from affine import Affine

    crs = SimpleNamespace(
        is_projected=True,
        to_epsg=lambda: 2056,
    )
    # Affine(x_scale, x_rot, x_origin, y_rot, y_scale, y_origin)
    # y_scale is negative for north-up rasters
    transform = Affine(0.1, 0.0, 2600000.0,
                       0.0, -0.1, 1200000.0)

    return SimpleNamespace(
        crs=crs,
        transform=transform,
        width=128,
        height=128,
    )