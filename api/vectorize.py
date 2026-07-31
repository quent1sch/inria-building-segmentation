"""
api/vectorize.py

GIS post-processing pipeline for building segmentation masks.

Pipeline
--------
  binary mask
    -> polygonization (contour extraction via OpenCV)
    -> shapely conversion (handles holes / building courtyards)
    -> Douglas-Peucker simplification (removes staircase raster artefacts)
    -> small area filtering (removes spurious detections)
    -> GeoJSON FeatureCollection

Mask reconstruction
-------------------
  polygons -> rasterize back to pixel mask (for evaluation / comparison)

Why this pipeline?
------------------
Raw segmentation masks have two classes of defects:
  1. Staircase edges - rasterized contours at 0.3m/px look jagged at display
     resolution. Douglas-Peucker fits straight line segments to straight
     building walls, which is geometrically correct.
  2. Spurious blobs - small detections below a physically meaningful area
     threshold (e.g. 10 m2) are almost certainly noise, shadows, or cars
     rather than buildings. Filtering by real-world area (pixels x res2)
     requires the input resolution to be known.

Dependencies: shapely, opencv-python-headless (in requirements.txt)
"""

import json
from typing import Any, Optional

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.validation import make_valid


# ── contour extraction ────────────────────────────────────────────────────────

def _mask_to_shapely_polygons(mask: np.ndarray) -> list[Polygon]:
    """
    Extract connected building blobs from a binary mask as Shapely Polygons.

    Uses RETR_CCOMP to capture two-level hierarchy:
      level 0 = outer contour (building footprint)
      level 1 = inner contour (courtyard / hole)

    Parameters
    ----------
    mask : (H, W) bool or uint8 array

    Returns
    -------
    list of shapely Polygon objects (may include holes)
    """
    uint8_mask = (mask > 0).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(
        uint8_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours is None or len(contours) == 0:
        return []

    # hierarchy shape: (1, N, 4) - [next, prev, first_child, parent]
    hierarchy = hierarchy[0]
    polygons = []

    for i, contour in enumerate(contours):
        # Only process outer contours (parent == -1)
        if hierarchy[i][3] != -1:
            continue

        if len(contour) < 3:
            continue

        exterior = contour.reshape(-1, 2).tolist()

        # Collect holes: all contours whose parent is this contour
        holes = []
        child_idx = hierarchy[i][2] # first child index
        while child_idx != -1:
            hole_contour = contours[child_idx]
            if len(hole_contour) >= 3:
                holes.append(hole_contour.reshape(-1, 2).tolist())
            child_idx = hierarchy[child_idx][0] # next sibling

        try:
            poly = Polygon(exterior, holes)
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.is_empty:
                continue
            # make_valid can return MultiPolygon - flatten
            if isinstance(poly, MultiPolygon):
                polygons.extend(poly.geoms)
            else:
                polygons.append(poly)
        except Exception:
            continue

    return polygons


# ── main pipeline ─────────────────────────────────────────────────────────────

def vectorize(
    mask: np.ndarray,
    resolution: Optional[float] = None,
    simplify_tolerance_m: float = 0.5,
    min_area_m2: float = 10.0,
) -> dict[str, Any]:
    """
    Convert a binary building mask to a GeoJSON FeatureCollection.

    Pipeline:
      mask -> polygonization -> Douglas-Peucker simplification -> area filtering

    Parameters
    ----------
    mask : (H, W) bool or uint8
        Binary building mask at the original image resolution.
    resolution : float or None
        Pixel resolution in metres/pixel.
        Required for real-world area filtering and metric simplification.
        If None, tolerance is treated as pixels and area filter is skipped.
    simplify_tolerance_m : float
        Douglas-Peucker epsilon in metres (converted to pixels internally).
        Controls the trade-off between edge fidelity and smoothness.
        0.5m is a good default for building outlines at 0.3m/px.
        Set to 0.0 to disable simplification.
    min_area_m2 : float
        Minimum building footprint area in square metres.
        Detections below this threshold are discarded as noise.
        Ignored if resolution is None.

    Returns
    -------
    GeoJSON FeatureCollection dict with pixel-coordinate geometries.
    Each feature carries a "area_px" property (and "area_m2" if resolution
    is known).
    """
    polygons = _mask_to_shapely_polygons(mask)

    if not polygons:
        return _empty_feature_collection()

    # Convert tolerance from metres to pixels
    if resolution is not None and resolution > 0:
        tolerance_px = simplify_tolerance_m / resolution
        min_area_px  = min_area_m2 / (resolution ** 2)
    else:
        tolerance_px = simplify_tolerance_m   # treat as pixels
        min_area_px  = None                   # skip area filter

    features = []
    for poly in polygons:
        # ── Douglas-Peucker simplification ───────────────────────────────
        if tolerance_px > 0:
            poly = poly.simplify(tolerance_px, preserve_topology=True)

        if poly.is_empty:
            continue

        # ── area filtering ────────────────────────────────────────────────
        area_px = poly.area
        if min_area_px is not None and area_px < min_area_px:
            continue

        # ── build GeoJSON feature ─────────────────────────────────────────
        props: dict[str, Any] = {"area_px": round(area_px, 2)}
        if resolution is not None:
            props["area_m2"] = round(area_px * resolution ** 2, 2)

        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": props,
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "n_buildings":           len(features),
            "resolution_m_per_px":   resolution,
            "simplify_tolerance_m":  simplify_tolerance_m if resolution else None,
            "simplify_tolerance_px": round(tolerance_px, 4),
            "min_area_m2":           min_area_m2 if resolution else None,
        },
    }


# ── mask reconstruction ───────────────────────────────────────────────────────

def polygons_to_mask(
    geojson: dict[str, Any],
    height: int,
    width: int,
) -> np.ndarray:
    """
    Rasterize a GeoJSON FeatureCollection back to a binary pixel mask.

    Useful for:
      - Comparing the vectorized (clean) mask against the raw prediction
      - Computing IoU between raw and vectorized outputs
      - Generating clean mask PNGs from vector results

    Parameters
    ----------
    geojson : dict
        GeoJSON FeatureCollection as returned by vectorize().
        Coordinates must be in pixel space (col, row).
    height, width : int
        Output mask dimensions — should match the original image.

    Returns
    -------
    np.ndarray  shape (H, W)  dtype bool
    """
    canvas = np.zeros((height, width), dtype=np.uint8)

    for feature in geojson.get("features", []):
        geom = feature.get("geometry", {})
        geom_type = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if geom_type == "Polygon":
            _draw_polygon(canvas, coords)
        elif geom_type == "MultiPolygon":
            for poly_coords in coords:
                _draw_polygon(canvas, poly_coords)

    return canvas.astype(bool)


def _draw_polygon(canvas: np.ndarray, coords: list) -> None:
    """
    Fill a GeoJSON polygon (with optional holes) onto a uint8 canvas.
    coords follows GeoJSON spec: [ exterior_ring, hole1, hole2, ... ]
    Each ring is a list of [x, y] pairs.
    """
    if not coords:
        return

    # Draw exterior (filled white)
    exterior = np.array(coords[0], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [exterior], color=255)

    # Punch out holes (filled black)
    for hole in coords[1:]:
        hole_pts = np.array(hole, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [hole_pts], color=0)


# ── serialisation helper ──────────────────────────────────────────────────────

def geojson_to_bytes(geojson: dict[str, Any]) -> bytes:
    """Serialise a GeoJSON dict to UTF-8 bytes."""
    return json.dumps(geojson, ensure_ascii=False).encode("utf-8")


def _empty_feature_collection() -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [],
        "metadata": {"n_buildings": 0},
    }