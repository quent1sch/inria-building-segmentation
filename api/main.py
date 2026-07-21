"""
api/main.py

FastAPI application exposing the building segmentation model as a REST API.

Endpoints
---------
POST /predict              — building mask PNG
POST /predict/overlay      — original image with buildings highlighted PNG
POST /predict/vector       — GeoJSON FeatureCollection of building polygons

Output processing levels
-------------------------
  raw         Direct model output, thresholded.
  clean       Vectorized (polygonized + simplified + filtered) then
              rasterized back to a pixel mask.
  vectorized  Vectorized polygon outlines — only meaningful for overlay.
 
Applies to:
  /predict        → ?processing=raw (default) | clean
  /predict/overlay → ?processing=raw (default) | vectorized | clean
  /predict/vector  → always vectorized (no param)

Resolution parameters (both endpoints)
---------------------------------------
  ?resolution=0.15   metres/pixel of the input image.
                     If omitted and the file is a GeoTIFF with embedded CRS,
                     resolution is read automatically.
 
  ?resample=false    Disable resampling for finer-than-training images.
                     Default: true (resample when beneficial).
 

Vectorization parameters (/predict/vector and /predict/overlay?vectorized=true)
---------------------------------------------------------------------------------
  ?simplify_tolerance=0.5   Douglas-Peucker epsilon in metres. Default 0.5.
  ?min_area=10.0            Minimum building area in m². Default 10.0.
                            Requires resolution to be known.
 
Response headers (when resolution is known)
-------------------------------------------
  X-Input-Resolution   : detected or supplied resolution in m/px
  X-Resampling-Applied : "true" / "false"
  X-Resampled-To       : target resolution used (only when resampling applied)
  X-Resolution-Warning : human-readable warning (only when applicable)
  X-Building-Count      number of polygons (/predict/vector only)

Usage
-----
uvicorn api.main:app --reload --port 8000

Examples
--------
  # Raw mask
  curl -X POST http://localhost:8000/predict \\
       -F "file=@tile.tif" --output mask.png
 
  # Clean mask (vectorized → rasterized)
  curl -X POST "http://localhost:8000/predict?processing=clean&resolution=0.3" \\
       -F "file=@tile.tif" --output clean_mask.png
 
  # Raw overlay
  curl -X POST http://localhost:8000/predict/overlay \\
       -F "file=@tile.tif" --output overlay.png
 
  # Overlay with clean polygon outlines
  curl -X POST "http://localhost:8000/predict/overlay?processing=vectorized&resolution=0.3" \\
       -F "file=@tile.tif" --output overlay_vector.png
 
  # Overlay with clean rasterized fill
  curl -X POST "http://localhost:8000/predict/overlay?processing=clean&resolution=0.3" \\
       -F "file=@tile.tif" --output overlay_clean.png
 
  # GeoJSON polygons
  curl -X POST "http://localhost:8000/predict/vector?resolution=0.3" \\
       -F "file=@tile.tif"
"""

import io
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw

from api.inference import ResolutionInfo, SegmentationInference, read_image_bytes
from api.vectorize import geojson_to_bytes, polygons_to_mask, vectorize


# ── enums ─────────────────────────────────────────────────────────────────────
 
class MaskProcessing(str, Enum):
    raw   = "raw"
    clean = "clean"
 
 
class OverlayProcessing(str, Enum):
    raw        = "raw"
    vectorized = "vectorized"
    clean      = "clean"


# ── app setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Inria Building Segmentation API",
    description=(
        "Binary building segmentation from aerial RGB imagery. "
        "Upload a JPEG/PNG/GeoTIFF and receive a binary mask, overlay or GeoJSON vector polygons.\n\n"
        "The model was trained on the Inria Aerial Image Labeling dataset at **0.3m/pixel**. "
        "Supply `?resolution=` or use a georeferenced GeoTIFF for automatic resolution handling."
    ),
    version="1.3.0",
)

# ── model singleton ──────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
CONFIG_PATH = Path("configs/config.yaml")

_inference: Optional[SegmentationInference] = None


def get_inference() -> SegmentationInference:
    global _inference
    if _inference is None:
        if not CHECKPOINT_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model checkpoint not found at '{CHECKPOINT_PATH}'. "
                    "Train the model first with: python train.py"
                ),
            )
        _inference = SegmentationInference(
            checkpoint_path=str(CHECKPOINT_PATH),
            config_path=str(CONFIG_PATH),
        )
    return _inference


# ── shared helpers ──────────────────────────────────────────────────────────────────

async def load_upload(file: UploadFile) -> tuple[np.ndarray, Optional[float]]:
    """
    Read an uploaded file -> (HWC uint8 RGB array, auto_resolution or None).
    Raises HTTP 422 on unsupported format or decode error.
    """
    data = await file.read()
    try:
        image, auto_resolution = read_image_bytes(data, filename=file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {e}")
    return image, auto_resolution
 
 
def resolution_headers(info: ResolutionInfo) -> dict:
    """Build response headers from a ResolutionInfo object."""
    headers = {}
    if info.input_resolution is not None:
        headers["X-Input-Resolution"] = f"{info.input_resolution:.4f}"
    headers["X-Resampling-Applied"] = "true" if info.resampled else "false"
    if info.resampled and info.resampled_to is not None:
        headers["X-Resampled-To"] = f"{info.resampled_to:.4f}"
    if info.warning:
        headers["X-Resolution-Warning"] = info.warning
    return headers
    

def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Binary mask (H, W bool/uint8) -> PNG bytes."""
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
    

def _raw_overlay(image: np.ndarray, mask: np.ndarray) -> bytes:
    """Original image with raw mask pixels highlighted in red."""
    overlay = image.copy()
    overlay[mask > 0] = np.clip(
        overlay[mask > 0].astype(int) * 0.5 + np.array([255, 50, 50]) * 0.5,
        0, 255,
    ).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(overlay, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()
 
 
def _vector_overlay(image: np.ndarray, geojson: dict) -> bytes:
    """
    Original image with simplified building polygon outlines drawn in red.
    Uses PIL ImageDraw so polygon edges are clean vectors, not raster fills.
    """
    pil_img = Image.fromarray(image, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
 
    for feature in geojson.get("features", []):
        geom      = feature.get("geometry", {})
        geom_type = geom.get("type", "")
        coords    = geom.get("coordinates", [])
 
        if geom_type == "Polygon":
            _draw_polygon_pil(draw, coords)
        elif geom_type == "MultiPolygon":
            for ring in coords:
                _draw_polygon_pil(draw, ring)
 
    result = Image.alpha_composite(pil_img, overlay).convert("RGB")
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()
 
 
def _draw_polygon_pil(draw: ImageDraw.Draw, coords: list) -> None:
    """Fill exterior in semi-transparent red, then punch holes in black."""
    if not coords:
        return
    exterior = [tuple(pt) for pt in coords[0]]
    if len(exterior) >= 3:
        draw.polygon(exterior, fill=(255, 50, 50, 120), outline=(255, 50, 50, 220))

    for hole in coords[1:]:
        pts = [tuple(pt) for pt in hole]
        if len(pts) >= 3:
            draw.polygon(pts, fill=(0, 0, 0, 0))


def _clean_overlay(image: np.ndarray, geojson: dict, H: int, W: int) -> bytes:
    """Rasterized clean mask pixels filled red on original image."""
    clean_mask = polygons_to_mask(geojson, height=H, width=W)
    return _raw_overlay(image, clean_mask)


# ── shared query param docs ───────────────────────────────────────────────────
 
_RES_DOC = (
    "Pixel resolution in **metres/pixel** (e.g. 0.15). "
    "Auto-detected for GeoTIFF with embedded projected CRS. "
    "Omit for JPEG/PNG if unknown."
)
_RESAMPLE_DOC = (
    "Resample to training resolution (0.3m/px) if image is finer. Default true."
)
_SIMPLIFY_DOC = (
    "Douglas-Peucker simplification tolerance in **metres**. "
    "Controls straight-edge fitting on building outlines. Default 0.5m."
)
_MIN_AREA_DOC = (
    "Minimum building footprint in **m2**. "
    "Detections below this are discarded as noise. "
    "Requires `resolution` to be known. Default 10.0 m2."
)
 

# ── /predict ──────────────────────────────────────────────────────────────────
 
@app.post(
    "/predict",
    summary="Building mask PNG",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def predict(
    file: UploadFile = File(..., description="Aerial image (JPEG, PNG, GeoTIFF)"),
    processing: MaskProcessing = Query(
        MaskProcessing.raw,
        description=(
            "**raw** — direct model output. "
            "**clean** — vectorized (polygonized + simplified + filtered) "
            "then rasterized back to pixels."
        ),
    ),
    resolution: Optional[float] = Query(None, description=_RES_DOC, gt=0),
    resample: bool = Query(True, description=_RESAMPLE_DOC),
    simplify_tolerance: float = Query(0.5, description=_SIMPLIFY_DOC, gt=0),
    min_area: float = Query(10.0, description=_MIN_AREA_DOC, gt=0),
):
    """
    Returns a **binary mask PNG** (white = building, black = background).
 
    - `processing=raw` (default): direct model output
    - `processing=clean`: polygonized → simplified → rasterized back to pixels
    """
    image, auto_res = await load_upload(file)
    eff_res = resolution if resolution is not None else auto_res
 
    model = get_inference()
    mask, info = model.predict(image, input_resolution=eff_res, resample=resample)
 
    if processing == MaskProcessing.clean:
        geojson    = vectorize(mask, resolution=eff_res,
                               simplify_tolerance_m=simplify_tolerance,
                               min_area_m2=min_area)
        H, W       = image.shape[:2]
        mask       = polygons_to_mask(geojson, height=H, width=W)
 
    return Response(
        content=mask_to_png_bytes(mask),
        media_type="image/png",
        headers=resolution_headers(info),
    )
 
 
# ── /predict/overlay ──────────────────────────────────────────────────────────
 
@app.post(
    "/predict/overlay",
    summary="Building overlay PNG",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def predict_overlay(
    file: UploadFile = File(..., description="Aerial image (JPEG, PNG, GeoTIFF)"),
    processing: OverlayProcessing = Query(
        OverlayProcessing.raw,
        description=(
            "**raw** — raw mask pixels filled red. "
            "**vectorized** — clean simplified polygon outlines. "
            "**clean** — vectorized then rasterized, filled red."
        ),
    ),
    resolution: Optional[float] = Query(None, description=_RES_DOC, gt=0),
    resample: bool = Query(True, description=_RESAMPLE_DOC),
    simplify_tolerance: float = Query(0.5, description=_SIMPLIFY_DOC, gt=0),
    min_area: float = Query(10.0, description=_MIN_AREA_DOC, gt=0),
):
    """
    Returns the original image with buildings highlighted.
 
    - `processing=raw` (default): raw mask pixels filled red
    - `processing=vectorized`: clean simplified polygon outlines drawn
    - `processing=clean`: vectorized → rasterized → filled red
    """
    image, auto_res = await load_upload(file)
    eff_res = resolution if resolution is not None else auto_res
 
    model = get_inference()
    mask, info = model.predict(image, input_resolution=eff_res, resample=resample)
 
    H, W = image.shape[:2]
 
    if processing == OverlayProcessing.raw:
        png_bytes = _raw_overlay(image, mask)
 
    elif processing == OverlayProcessing.vectorized:
        geojson   = vectorize(mask, resolution=eff_res,
                              simplify_tolerance_m=simplify_tolerance,
                              min_area_m2=min_area)
        png_bytes = _vector_overlay(image, geojson)
 
    else:  # clean
        geojson   = vectorize(mask, resolution=eff_res,
                              simplify_tolerance_m=simplify_tolerance,
                              min_area_m2=min_area)
        png_bytes = _clean_overlay(image, geojson, H, W)
 
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers=resolution_headers(info),
    )
 
 
# ── /predict/vector ───────────────────────────────────────────────────────────
 
@app.post(
    "/predict/vector",
    summary="Building polygons GeoJSON",
    response_class=JSONResponse,
)
async def predict_vector(
    file: UploadFile = File(..., description="Aerial image (JPEG, PNG, GeoTIFF)"),
    resolution: Optional[float] = Query(None, description=_RES_DOC, gt=0),
    resample: bool = Query(True, description=_RESAMPLE_DOC),
    simplify_tolerance: float = Query(0.5, description=_SIMPLIFY_DOC, gt=0),
    min_area: float = Query(10.0, description=_MIN_AREA_DOC, gt=0),
):
    """
    Returns a **GeoJSON FeatureCollection** of building polygons.
 
    Coordinates are in **pixel space** (col, row) of the original input image.
 
    Each feature has properties:
    - `area_px`: polygon area in pixels²
    - `area_m2`: real-world area in m² (only when resolution is known)
 
    The response `metadata` block records the parameters used
    (n_buildings, resolution, simplify_tolerance, min_area).
    """
    image, auto_res = await load_upload(file)
    eff_res = resolution if resolution is not None else auto_res
 
    model = get_inference()
    mask, info = model.predict(image, input_resolution=eff_res, resample=resample)
 
    geojson = vectorize(
        mask,
        resolution=eff_res,
        simplify_tolerance_m=simplify_tolerance,
        min_area_m2=min_area,
    )
 
    headers = resolution_headers(info)
    headers["X-Building-Count"] = str(geojson["metadata"]["n_buildings"])
 
    return JSONResponse(content=geojson, headers=headers)
 
 
# ── health ────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "model_loaded": _inference is not None}
 
@app.get("/", include_in_schema=False)
def root():
    return {"message": "Inria Building Segmentation API — see /docs for usage."}
 