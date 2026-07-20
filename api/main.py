"""
api/main.py

FastAPI application exposing the building segmentation model as a REST API.

Endpoints
---------
GET  /health          — liveness check
POST /predict         — upload an image, receive a binary mask PNG
POST /predict/overlay — upload an image, receive a coloured overlay PNG

Resolution parameters (both endpoints)
---------------------------------------
  ?resolution=0.15   metres/pixel of the input image.
                     If omitted and the file is a GeoTIFF with embedded CRS,
                     resolution is read automatically.
 
  ?resample=false    Disable resampling even if image is finer than 0.3m/px.
                     Default: true (resample when beneficial).
 
Resolution logic
----------------
  Finer than 0.3m/px   -> resampled to 0.3m/px (unless resample=false)
  Coarser than 0.3m/px -> not resampled; warning returned in headers
  Unknown              -> no action
 
Response headers (when resolution is known)
-------------------------------------------
  X-Input-Resolution   : detected or supplied resolution in m/px
  X-Resampling-Applied : "true" / "false"
  X-Resampled-To       : target resolution used (only when resampling applied)
  X-Resolution-Warning : human-readable warning (only when applicable)

Usage
-----
uvicorn api.main:app --reload --port 8000

Examples
--------
  # GeoTIFF with embedded resolution — auto-detected
  curl -X POST http://localhost:8000/predict -F "file=@tile.tif" --output mask.png
 
  # JPEG with known resolution
  curl -X POST "http://localhost:8000/predict?resolution=0.15" \
       -F "file=@image.jpg" --output mask.png
 
  # Skip resampling
  curl -X POST "http://localhost:8000/predict?resolution=0.15&resample=false" \
       -F "file=@image.jpg" --output mask.png
"""

import io
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image

from api.inference import ResolutionInfo, SegmentationInference, read_image_bytes

# ── app setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Inria Building Segmentation API",
    description=(
        "Binary building segmentation from aerial RGB imagery. "
        "Upload a JPEG/PNG/GeoTIFF and receive a binary mask or overlay.\n\n"
        "The model was trained on the Inria Aerial Image Labeling dataset at **0.3m/pixel**. "
        "Supply `?resolution=` or use a georeferenced GeoTIFF for automatic resolution handling."
    ),
    version="1.0.0",
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
    

def overlay_to_png_bytes(image: np.ndarray, mask: np.ndarray) -> bytes:
    """Original image with buildings highlighted in semi-transparent red overlay -> PNG bytes."""
    overlay = image.copy()
    overlay[mask > 0] = np.clip(
        overlay[mask > 0].astype(int) * 0.5 + np.array([255, 50, 50]) * 0.5,
        0, 255,
    ).astype(np.uint8)
    img = Image.fromarray(overlay, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── shared query param docs ───────────────────────────────────────────────────
 
_RESOLUTION_DOC = (
    "Pixel resolution of the input image in **metres per pixel** "
    "(e.g. 0.15 for 15cm/px). "
    "For GeoTIFF files with an embedded projected CRS, this is read automatically "
    "and this parameter is ignored. "
    "For JPEG/PNG, omit if resolution is unknown."
)
 
_RESAMPLE_DOC = (
    "If `true` (default), images finer than the training resolution (0.3m/px) "
    "are resampled to 0.3m/px before inference. "
    "Set to `false` to skip resampling."
)
 

# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "model_loaded": _inference is not None}


@app.post(
    "/predict",
    summary="Predict binary building mask",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def predict(
    file: UploadFile = File(..., description="Aerial RGB image (JPEG, PNG, GeoTIFF)"),
    resolution: Optional[float] = Query(None, description=_RESOLUTION_DOC, gt=0),
    resample: bool = Query(True, description=_RESAMPLE_DOC),
    ):
    """
    Upload an aerial RGB image and receive a **binary mask** PNG back.

    - White pixels (255) = building
    - Black pixels (0)   = background

    The output mask is always aligned to the **original input pixel grid**,
    even when resampling was applied internally.
    """
    image, auto_resolution = await load_upload(file)
 
    # User-supplied resolution takes precedence; fall back to auto-detected
    effective_resolution = resolution if resolution is not None else auto_resolution
 
    model = get_inference()
    mask, info = model.predict(image, input_resolution=effective_resolution, resample=resample)
 
    return Response(
        content=mask_to_png_bytes(mask),
        media_type="image/png",
        headers=resolution_headers(info),
    )


@app.post(
    "/predict/overlay",
    summary="Predict with coloured overlay",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def predict_overlay(
    file: UploadFile = File(..., description="Aerial RGB image (JPEG, PNG, GeoTIFF)"),
    resolution: Optional[float] = Query(None, description=_RESOLUTION_DOC, gt=0),
    resample: bool = Query(True, description=_RESAMPLE_DOC),
):
    """
    Upload an aerial RGB image and receive the original image with buildings
    highlighted in **red overlay** as a PNG.

    The overlay is always rendered at the **original input image resolution**.
    """
    image, auto_resolution = await load_upload(file)
    effective_resolution = resolution if resolution is not None else auto_resolution
 
    model = get_inference()
    mask, info = model.predict(image, input_resolution=effective_resolution, resample=resample)
 
    return Response(
        content=overlay_to_png_bytes(image, mask),
        media_type="image/png",
        headers=resolution_headers(info),
    )


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Inria Building Segmentation API. See /docs for usage."}
