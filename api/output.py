"""
api/output.py

Format-aware output serialisation.

All result bytes generation lives here so main.py and job_runner.py
don't need to know about file formats.

Why this exists
---------------
Previously all output was PNG regardless of input format. This threw away
the CRS and affine transform from GeoTIFF inputs, making results useless
in GIS tools (QGIS, GeoPandas, etc.).

Now:
  - GeoTIFF input with CRS → GeoTIFF output by default (CRS + transform preserved)
  - Plain image input       → PNG by default
  - User can override with output_format="png" | "tif"

For vector output:
  - GeoTIFF input with CRS → GeoJSON in world coordinates by default
    (coordinates in metres/degrees matching the input CRS, loadable in QGIS)
  - Plain image input       → GeoJSON in pixel coordinates (previous behaviour)
  - User can override with coords="world" | "pixel"

Public API
----------
  mask_to_bytes(mask, spatial_ref, output_format)
      → (bytes, media_type, file_extension)

  overlay_to_bytes(image, mask, processing, geojson, spatial_ref, output_format)
      → (bytes, media_type, file_extension)

  vector_to_bytes(mask, spatial_ref, coords, simplify_tolerance, min_area, resolution)
      → (bytes, media_type, file_extension)

  resolve_output_format(output_format, spatial_ref)
      → "tif" | "png"   — resolves "auto" to the correct default
"""

from __future__ import annotations

import io
import json
from typing import Optional, Tuple

import numpy as np
from PIL import Image as PILImage, ImageDraw


# ── format resolution ─────────────────────────────────────────────────────────

def resolve_output_format(
    output_format: str,
    spatial_ref,
) -> str:
    """
    Resolve "auto" to the concrete format based on whether spatial metadata
    is available.

    "auto" + spatial_ref available → "tif"  (preserve geospatial metadata)
    "auto" + no spatial_ref        → "png"  (no metadata to preserve)
    Explicit "tif" or "png"        → returned as-is
    """
    if output_format != "auto":
        return output_format
    return "tif" if spatial_ref is not None else "png"


def resolve_coords(coords: str, spatial_ref) -> str:
    """
    Resolve "auto" to "world" if spatial_ref is available, "pixel" otherwise.
    Only relevant for result_type=vector.
    """
    if coords != "auto":
        return coords
    return "world" if spatial_ref is not None else "pixel"


def _media_type_and_ext(fmt: str) -> Tuple[str, str]:
    if fmt == "tif":
        return "image/tiff", ".tif"
    return "image/png", ".png"


# ── GeoTIFF writing ───────────────────────────────────────────────────────────

def _write_geotiff(
    arrays: np.ndarray,
    spatial_ref,
    dtype=np.uint8,
) -> bytes:
    """
    Write one or more bands to an in-memory GeoTIFF with the original
    CRS and affine transform.

    Parameters
    ----------
    arrays      : (H, W) for single-band (mask), (H, W, 3) for RGB (overlay)
    spatial_ref : SpatialRef from inference.py — carries CRS + transform
    dtype       : output dtype (uint8 for masks and overlays)

    Returns
    -------
    bytes of a valid GeoTIFF readable by rasterio, QGIS, GDAL, etc.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        raise ImportError("rasterio is required for GeoTIFF output.")

    buf = io.BytesIO()

    if arrays.ndim == 2:
        # Single-band mask: (H, W) → write as 1-band GeoTIFF
        H, W  = arrays.shape
        count = 1
        data  = arrays[np.newaxis, ...]   # (1, H, W)
    else:
        # RGB overlay: (H, W, 3) → write as 3-band GeoTIFF
        H, W, _ = arrays.shape
        count   = 3
        data    = arrays.transpose(2, 0, 1)   # (3, H, W)

    # If output dims differ from original (shouldn't happen — mask is always
    # returned at input resolution), we'd need to adjust the transform.
    # Assert here to catch any future pipeline changes early.
    assert H == spatial_ref.height and W == spatial_ref.width, (
        f"Output size ({H}×{W}) does not match original input "
        f"({spatial_ref.height}×{spatial_ref.width}). "
        "Spatial transform would be incorrect — check upsampling logic."
    )

    with rasterio.open(
        buf, "w",
        driver="GTiff",
        height=H,
        width=W,
        count=count,
        dtype=dtype,
        crs=spatial_ref.crs,
        transform=spatial_ref.transform,
        compress="lzw",      # lossless, standard for masks
    ) as dst:
        for i in range(count):
            dst.write(data[i], i + 1)

    return buf.getvalue()


# ── mask output ───────────────────────────────────────────────────────────────

def mask_to_bytes(
    mask: np.ndarray,
    spatial_ref,
    output_format: str,
) -> Tuple[bytes, str, str]:
    """
    Serialise a binary mask to bytes.

    Parameters
    ----------
    mask          : (H, W) bool or uint8
    spatial_ref   : SpatialRef or None
    output_format : "auto" | "tif" | "png"

    Returns
    -------
    (bytes, media_type, file_extension)
    """
    fmt = resolve_output_format(output_format, spatial_ref)

    if fmt == "tif" and spatial_ref is not None:
        # Georeferenced GeoTIFF — CRS + transform preserved
        data = _write_geotiff(
            (mask.astype(np.uint8) * 255),
            spatial_ref,
        )
    elif fmt == "tif":
        # Plain TIFF (no CRS — input had no spatial metadata or user forced tif)
        img  = PILImage.fromarray((mask.astype(np.uint8) * 255), mode="L")
        buf  = io.BytesIO()
        img.save(buf, format="TIFF")
        data = buf.getvalue()
    else:
        # PNG
        img  = PILImage.fromarray((mask.astype(np.uint8) * 255), mode="L")
        buf  = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()

    media_type, ext = _media_type_and_ext(fmt)
    return data, media_type, ext


# ── overlay output ────────────────────────────────────────────────────────────

def overlay_to_bytes(
    image: np.ndarray,
    mask: np.ndarray,
    processing: str,
    geojson: Optional[dict],
    spatial_ref,
    output_format: str,
) -> Tuple[bytes, str, str]:
    """
    Render overlay and serialise to bytes.

    Parameters
    ----------
    image         : (H, W, 3) uint8 original image
    mask          : (H, W) bool prediction mask
    processing    : "raw" | "clean" | "vectorized"
    geojson       : precomputed GeoJSON dict (required for vectorized/clean)
    spatial_ref   : SpatialRef or None
    output_format : "auto" | "tif" | "png"

    Returns
    -------
    (bytes, media_type, file_extension)
    """
    fmt = resolve_output_format(output_format, spatial_ref)

    if processing == "vectorized" and geojson is not None:
        rgb = _vector_overlay_rgb(image, geojson)
    else:
        # raw and clean both use a pixel fill — clean just has a cleaner mask
        rgb = _raw_overlay_rgb(image.copy(), mask)

    if fmt == "tif" and spatial_ref is not None:
        data = _write_geotiff(rgb, spatial_ref)
    elif fmt == "tif":
        buf = io.BytesIO()
        PILImage.fromarray(rgb, mode="RGB").save(buf, format="TIFF")
        data = buf.getvalue()
    else:
        buf = io.BytesIO()
        PILImage.fromarray(rgb, mode="RGB").save(buf, format="PNG")
        data = buf.getvalue()

    media_type, ext = _media_type_and_ext(fmt)
    return data, media_type, ext


def _raw_overlay_rgb(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply semi-transparent red fill over building pixels. Returns RGB array."""
    image[mask > 0] = np.clip(
        image[mask > 0].astype(int) * 0.5 + np.array([255, 50, 50]) * 0.5,
        0, 255,
    ).astype(np.uint8)
    return image


def _vector_overlay_rgb(image: np.ndarray, geojson: dict) -> np.ndarray:
    """Draw simplified polygon outlines over the image. Returns RGB array."""
    pil_img = PILImage.fromarray(image, mode="RGB").convert("RGBA")
    overlay = PILImage.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for feature in geojson.get("features", []):
        geom   = feature.get("geometry", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") == "Polygon":
            _draw_poly(draw, coords)
        elif geom.get("type") == "MultiPolygon":
            for ring in coords:
                _draw_poly(draw, ring)

    result = PILImage.alpha_composite(pil_img, overlay).convert("RGB")
    return np.array(result)


def _draw_poly(draw: ImageDraw.Draw, coords: list) -> None:
    if not coords:
        return
    exterior = [tuple(pt) for pt in coords[0]]
    if len(exterior) >= 3:
        draw.polygon(exterior, fill=(255, 50, 50, 120), outline=(255, 50, 50, 220))
    for hole in coords[1:]:
        pts = [tuple(pt) for pt in hole]
        if len(pts) >= 3:
            draw.polygon(pts, fill=(0, 0, 0, 0))


# ── vector output ─────────────────────────────────────────────────────────────

def vector_to_bytes(
    mask: np.ndarray,
    spatial_ref,
    coords: str,
    simplify_tolerance_m: float,
    min_area_m2: float,
    resolution: Optional[float],
) -> Tuple[bytes, str, str]:
    """
    Vectorize mask and serialise to GeoJSON bytes.

    Parameters
    ----------
    mask                 : (H, W) bool prediction mask
    spatial_ref          : SpatialRef or None
    coords               : "auto" | "world" | "pixel"
    simplify_tolerance_m : Douglas-Peucker epsilon in metres
    min_area_m2          : minimum building area filter
    resolution           : m/px — used by vectorize() for area filtering

    Returns
    -------
    (bytes, media_type, file_extension)
    Always returns ("application/json", ".geojson").
    """
    from api.vectorize import vectorize

    resolved_coords = resolve_coords(coords, spatial_ref)

    # vectorize() always works in pixel space
    geojson = vectorize(
        mask,
        resolution=resolution,
        simplify_tolerance_m=simplify_tolerance_m,
        min_area_m2=min_area_m2,
    )

    if resolved_coords == "world" and spatial_ref is not None:
        # Convert pixel coordinates to world coordinates using the affine transform.
        # shapely.affinity.affine_transform applies a 2D affine matrix to a geometry.
        # Rasterio's transform maps (col, row) → (x, y) in CRS units.
        geojson = _pixels_to_world(geojson, spatial_ref)

    data = json.dumps(geojson, ensure_ascii=False).encode("utf-8")
    return data, "application/json", ".geojson"


def _pixels_to_world(geojson: dict, spatial_ref) -> dict:
    """
    Convert all polygon coordinates in a GeoJSON FeatureCollection from
    pixel (col, row) space to world (x, y) coordinates using the affine
    transform from spatial_ref.

    Shapely's affine_transform takes a 6-element matrix [a, b, d, e, xoff, yoff]
    matching rasterio's Affine object coefficients:
        | a  b  c |       a = pixel width (m/px)
        | d  e  f |       e = pixel height (negative for north-up rasters)
        | 0  0  1 |       c, f = top-left corner coordinates

    The output GeoJSON includes a "crs" member pointing to the input CRS
    so QGIS and GeoPandas can load it with correct georeferencing.
    """
    from shapely.geometry import shape, mapping
    from shapely.affinity import affine_transform

    t = spatial_ref.transform
    # Shapely matrix: [a, b, d, e, xoff, yoff]
    # Rasterio Affine: a=col_scale, b=col_rot, c=x_origin,
    #                  d=row_rot, e=row_scale, f=y_origin
    matrix = [t.a, t.b, t.d, t.e, t.c, t.f]

    new_features = []
    for feature in geojson.get("features", []):
        try:
            geom       = shape(feature["geometry"])
            world_geom = affine_transform(geom, matrix)
            new_features.append({
                **feature,
                "geometry": mapping(world_geom),
            })
        except Exception:
            # Keep original if transform fails for any polygon
            new_features.append(feature)

    # Add CRS member so GIS tools know the coordinate system
    # GeoJSON RFC 7946 technically deprecates the "crs" member but QGIS
    # and most GIS tools still read it for non-WGS84 data.
    crs_name = str(spatial_ref.crs.to_epsg()) if spatial_ref.crs.to_epsg() else str(spatial_ref.crs)

    return {
        **geojson,
        "features": new_features,
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs_name}"},
        },
        "metadata": {
            **geojson.get("metadata", {}),
            "coords": "world",
            "crs": crs_name,
        },
    }