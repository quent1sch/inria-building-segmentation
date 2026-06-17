"""
api/main.py

FastAPI application exposing the building segmentation model as a REST API.

Endpoints
---------
GET  /health          — liveness check
POST /predict         — upload an image, receive a binary mask PNG
POST /predict/overlay — upload an image, receive a coloured overlay PNG

Usage
-----
uvicorn api.main:app --reload --port 8000
"""

import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from api.inference import SegmentationInference

# ── app setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Inria Building Segmentation API",
    description=(
        "Binary building segmentation from aerial RGB imagery. "
        "Upload a JPEG/PNG/GeoTIFF and receive a binary mask or overlay."
    ),
    version="1.0.0",
)

# ── model singleton ──────────────────────────────────────────────────────────

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
CONFIG_PATH = Path("configs/config.yaml")

_inference: SegmentationInference | None = None


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


# ── helpers ──────────────────────────────────────────────────────────────────

async def read_image_upload(file: UploadFile) -> np.ndarray:
    """Read uploaded file → HWC uint8 numpy (RGB)."""
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {e}")


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Binary mask (H, W bool/uint8) → PNG bytes."""
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def overlay_to_png_bytes(image: np.ndarray, mask: np.ndarray) -> bytes:
    """Return a semi-transparent red overlay PNG."""
    overlay = image.copy()
    overlay[mask > 0] = np.clip(
        overlay[mask > 0].astype(int) * 0.5 + np.array([255, 50, 50]) * 0.5,
        0, 255,
    ).astype(np.uint8)
    img = Image.fromarray(overlay, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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
async def predict(file: UploadFile = File(..., description="RGB image (JPEG/PNG)")):
    """
    Upload an aerial RGB image and receive a **binary mask** PNG back.

    - White pixels (255) = building
    - Black pixels (0)   = background
    """
    image_np = await read_image_upload(file)
    model = get_inference()
    mask = model.predict(image_np)
    return Response(content=mask_to_png_bytes(mask), media_type="image/png")


@app.post(
    "/predict/overlay",
    summary="Predict with coloured overlay",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def predict_overlay(
    file: UploadFile = File(..., description="RGB image (JPEG/PNG)")
):
    """
    Upload an aerial RGB image and receive the original image with buildings
    highlighted in **red overlay** as a PNG.
    """
    image_np = await read_image_upload(file)
    model = get_inference()
    mask = model.predict(image_np)
    return Response(content=overlay_to_png_bytes(image_np, mask), media_type="image/png")


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Inria Building Segmentation API. See /docs for usage."}
