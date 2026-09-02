"""
api/main.py
 
FastAPI application — building segmentation as a service.
 
The model was trained on the Inria Aerial Image Labeling dataset at 0.3m/pixel.
Input images are resampled to that resolution before inference when needed.
Results are written in the same format as the input by default (GeoTIFF in →
GeoTIFF out, preserving CRS and affine transform for direct GIS use).
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENDPOINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
Sync — result returned directly in the response body:
  POST /predict/upload          multipart file upload (small/medium images)
  POST /predict/from-path       open file directly from container filesystem
                                (best for large tiles — zero upload buffering,
                                rasterio reads straight from disk)
  POST /predict/from-url        download from URL then run inference
                                (Azure Blob SAS, public URL — streamed in chunks)
 
Async — returns job_id immediately, result fetched separately:
  POST /predict/async/upload
  POST /predict/async/from-path
  POST /predict/async/from-url
  → 202 Accepted: {"job_id": "...", "status": "queued", "poll_url": "/jobs/{id}"}
 
Job management:
  GET  /jobs/{job_id}           poll status; result_url populated when done
  GET  /jobs/{job_id}/result    stream result (local) or redirect to SAS URL (Azure)
  GET  /jobs/                   list recent jobs for the current user
 
Local result streaming (STORAGE_BACKEND=local only):
  GET  /results/{key}           stream a stored result file by key
                                (Azure mode: clients use the presigned SAS URL directly)
 
Health:
  GET  /health                  DB + storage backend + model reachability
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETERS  (JSON body for from-path/from-url; form fields for upload)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
Resolution:
  resolution=<float>       Input resolution in metres/pixel.
                           Auto-detected from GeoTIFF CRS metadata when available.
                           Omit for JPEG/PNG if unknown — inference runs as-is.
 
  resample=true|false      Resample finer-than-training images to 0.3m/px before
                           inference. Default: true. Set false to skip resampling.
                           Images coarser than 0.3m/px are never resampled — a
                           warning is returned in the X-Resolution-Warning header.
 
Result type (what you get back):
  result_type=mask         Binary mask: white=building, black=background.  [default]
  result_type=overlay      Original image with buildings highlighted in red.
  result_type=vector       GeoJSON FeatureCollection of building polygons.
 
Processing level (how the prediction is post-processed):
  processing=raw           Direct model output, thresholded. Fast, pixel-accurate,
                           but may have jagged edges and small spurious detections.
                           Valid for: mask, overlay.                        [default]
 
  processing=clean         Polygonize → Douglas-Peucker simplify → area filter →
                           rasterize back to pixels. Straight building edges,
                           noise removed. Slower than raw.
                           Valid for: mask, overlay.
 
  processing=vectorized    Simplified polygon outlines drawn over the image.
                           Valid for: overlay only.
 
  result_type=vector always runs the full vectorization pipeline regardless of
  the processing param (which is ignored for vector output).
 
Vectorization (used when processing=clean, processing=vectorized, or result_type=vector):
  simplify_tolerance=0.5   Douglas-Peucker epsilon in metres. Controls how
                           aggressively straight lines are fitted to building
                           outlines. Higher = more simplified. Default: 0.5m.
 
  min_area=10.0            Minimum building footprint in m². Detections below
                           this are discarded as noise. Default: 10.0 m².
                           Requires resolution to be known.
 
Output file format:
  output_format=auto       GeoTIFF input with CRS → GeoTIFF output (CRS +
                           affine transform preserved, loadable in QGIS etc.).
                           Plain JPEG/PNG input → PNG output.              [default]
  output_format=tif        Force GeoTIFF output regardless of input.
  output_format=png        Force PNG output (spatial metadata discarded).
  (Not applicable for result_type=vector which always returns GeoJSON.)
 
Vector coordinate space (result_type=vector only):
  coords=auto              GeoTIFF with CRS → world coordinates (metres in
                           LV95, degrees in WGS84, etc.) with a CRS member in
                           the GeoJSON — loadable directly in QGIS/GeoPandas.
                           Plain image → pixel coordinates (col, row).     [default]
  coords=world             Force world coordinates (requires CRS in input).
  coords=pixel             Force pixel (col, row) coordinates.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE HEADERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
  X-Input-Resolution       detected or supplied resolution in m/px
  X-Resampling-Applied     "true" / "false"
  X-Resampled-To           target resolution used (only when resampling applied)
  X-Resolution-Warning     set when input is coarser than training resolution
  X-Output-Format          actual format used: "tif" or "png"
  X-Output-CRS             EPSG code of output CRS (e.g. "EPSG:2056") when georeferenced
  X-Job-Id                 DB job ID logged for every sync request
  X-Cache                  "HIT" — result served from cache / "MISS" — freshly computed
  X-Building-Count         number of polygons (async result headers only)
  Content-Disposition      filename hint with correct extension (result.tif / result.png)
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
  Local single-user:  API_KEY env var empty (default) → no auth required.
                      All requests attributed to DEFAULT_USER_ID="local".
  Multi-user / cloud: set API_KEY env var → requests must include header:
                      X-API-Key: <value>
                      (Replace with JWT/Azure AD token validation in production.)
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLOUD MIGRATION — env vars only, zero code changes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
  Local (defaults, no config needed):
    STORAGE_BACKEND=local
    DATABASE_URL=sqlite+aiosqlite:///jobs.db
    WORKER_MODE=thread          # jobs run in-process thread pool
 
  Azure production:
    STORAGE_BACKEND=azure
    AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
    AZURE_STORAGE_CONTAINER=segmentation-results
    DATABASE_URL=postgresql+asyncpg://user:pass@host/dbname
    WORKER_MODE=queue           # jobs run in separate worker container
    AZURE_QUEUE_CONNECTION_STRING=...
    AZURE_QUEUE_NAME=inference-jobs
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
  uvicorn api.main:app --reload --port 8000   # local dev
  docker compose up                           # Docker
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
The async/from-path pattern is the recommended approach for large GeoTIFF tiles
(e.g. SWISSIMAGE 10k×10k at 0.1m/px). It returns immediately, processes in the
background, and the result is a georeferenced GeoTIFF loadable in QGIS.
 
-- Async: raw mask GeoTIFF (default output_format=auto → tif since input has CRS)
  curl -s -X POST http://localhost:8000/predict/async/from-path \
       -H "Content-Type: application/json" \
       -d '{"path": "/data/swissimage_2494-1114.tif"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['poll_url'])"
  → /jobs/abc-123
 
  curl http://localhost:8000/jobs/abc-123
  → {"status": "done", "result_url": "/results/abc-123.tif", ...}
 
  curl http://localhost:8000/jobs/abc-123/result --output mask.tif
 
-- Async: clean mask GeoTIFF (postprocessed — straight edges, noise removed)
  curl -X POST http://localhost:8000/predict/async/from-path \
       -H "Content-Type: application/json" \
       -d '{"path": "/data/swissimage_2494-1114.tif",
            "resolution": 0.1, "processing": "clean"}' \
  → poll /jobs/{id}, download → mask_clean.tif
 
-- Async: overlay GeoTIFF with raw mask fill
  curl -X POST http://localhost:8000/predict/async/from-path \
       -H "Content-Type: application/json" \
       -d '{"path": "/data/swissimage_2494-1114.tif",
            "result_type": "overlay", "resolution": 0.1}' \
  → poll /jobs/{id}, download → overlay_raw.tif
 
-- Async: overlay GeoTIFF with clean polygon outlines
  curl -X POST http://localhost:8000/predict/async/from-path \
       -H "Content-Type: application/json" \
       -d '{"path": "/data/swissimage_2494-1114.tif",
            "result_type": "overlay", "processing": "vectorized", "resolution": 0.1}' \
  → poll /jobs/{id}, download → overlay_vector.tif
 
-- Async: GeoJSON in LV95 world coordinates (QGIS-ready, no pixel coords)
  curl -X POST http://localhost:8000/predict/async/from-path \
       -H "Content-Type: application/json" \
       -d '{"path": "/data/swissimage_2494-1114.tif",
            "result_type": "vector", "resolution": 0.1}' \
  → poll /jobs/{id}, download → buildings_lv95.geojson
  (coords=auto → world because input has CRS EPSG:2056)
 
-- Async: GeoJSON in pixel coordinates (explicit override)
  curl -X POST http://localhost:8000/predict/async/from-path \
       -H "Content-Type: application/json" \
       -d '{"path": "/data/swissimage_2494-1114.tif",
            "result_type": "vector", "resolution": 0.1, "coords": "pixel"}' \
  → poll /jobs/{id}, download → buildings_pixels.geojson
 
-- Async: force PNG output even though input is GeoTIFF
  curl -X POST http://localhost:8000/predict/async/from-path \
       -H "Content-Type: application/json" \
       -d '{"path": "/data/swissimage_2494-1114.tif", "output_format": "png"}' \
  → poll /jobs/{id}, download → mask.png
 
-- Sync upload (small images, external clients without filesystem access)
  curl -X POST http://localhost:8000/predict/upload \
       -F "file=@small_tile.tif" \
       -F "result_type=overlay" -F "processing=clean" -F "resolution=0.3" \
       --output overlay_clean.tif --dump-header -
  → X-Output-Format: tif
    X-Output-CRS: EPSG:2056
    X-Resampling-Applied: false
    X-Cache: MISS
 
-- Check if a result was cached (second identical request → HIT)
  curl -X POST http://localhost:8000/predict/async/from-path \
       -H "Content-Type: application/json" \
       -d '{"path": "/data/swissimage_2494-1114.tif", "resolution": 0.1}'
  → X-Cache: HIT  (result returned immediately, no inference)
 
-- List recent jobs
  curl http://localhost:8000/jobs/?limit=5
"""

import io
import json
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from PIL import Image as PILImage, ImageDraw
from sqlalchemy.ext.asyncio import AsyncSession

from api.inference import (
    SegmentationInference,
    SpatialRef,
    _bands_to_rgb_uint8,
    read_image_bytes,
    ResolutionInfo,
)
from api.output import (
    mask_to_bytes,
    overlay_to_bytes,
    vector_to_bytes,
    resolve_output_format,
)
from api.schemas import (
    AsyncJobAccepted,
    FromPathRequest,
    FromUrlRequest,
    HealthResponse,
    InferenceParams,
    JobResponse,
)
from api.vectorize import vectorize, polygons_to_mask
from config import get_settings
from db.crud import (
    compute_input_hash,
    create_job,
    get_cached_result,
    get_job,
    hash_bytes,
    list_jobs,
    store_cached_result,
    mark_done,
)
from db.models import InputMode, JobStatus, ResultType
from db.session import db_session_dependency, init_db, close_db
from storage import get_storage
from worker.job_runner import run_job_async
from worker.queue import get_queue


# ── app lifespan ──────────────────────────────────────────────────────────────

_queue = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global _queue
    settings = get_settings()

    # Initialise DB tables
    await init_db()

    # Start worker queue (thread mode: starts background task;
    # queue mode: worker runs in separate container, this is a no-op)
    if settings.worker_mode == "thread":
        _queue = get_queue()
        await _queue.start(lambda job_id: __import__("asyncio").run(run_job_async(job_id)))

    print(f"API started — storage={settings.storage_backend} "
          f"db={settings.database_url[:40]}... "
          f"worker={settings.worker_mode}")
    yield

    # Shutdown
    if _queue:
        await _queue.stop()
    await close_db()


# ── app ───────────────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Binary building segmentation from aerial imagery.\n\n"
        "Three input modes: **upload** (multipart), **from-path** (local/mounted), "
        "**from-url** (Azure Blob SAS or public URL).\n\n"
        "Each has a sync variant (result returned directly) and an async variant "
        "(returns job_id, poll `/jobs/{job_id}` for result).\n\n"
        "Trained on Inria Aerial Image Labeling dataset at **0.3 m/pixel**."
    ),
    lifespan=lifespan,
)


# ── singletons ────────────────────────────────────────────────────────────────

_model:   Optional[SegmentationInference] = None
_storage = None


def get_model() -> SegmentationInference:
    global _model
    if _model is None:
        _model = SegmentationInference(
            checkpoint_path=settings.checkpoint_path,
            config_path=settings.config_path,
        )
    return _model


def get_storage_backend():
    global _storage
    if _storage is None:
        _storage = get_storage()
    return _storage


# ── auth dependency ───────────────────────────────────────────────────────────

async def get_user_id(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Extract user_id from request.

    Local single-user: API_KEY is empty → always returns DEFAULT_USER_ID.
    Production: API_KEY is set → validate header, return key as user_id.

    In a real multi-user system you'd validate a JWT here instead and
    extract the user ID from the token claims.
    """
    if settings.api_key:
        if x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
        return x_api_key   # use key as user_id; replace with JWT sub claim in production
    return settings.default_user_id


# ── inference helpers ─────────────────────────────────────────────────────────

def _build_params(p: InferenceParams) -> dict:
    return {
        "resolution":         p.resolution,
        "resample":           p.resample,
        "processing":         p.processing,
        "result_type":        p.result_type,
        "simplify_tolerance": p.simplify_tolerance,
        "min_area":           p.min_area,
    }


def _run_inference_sync(
    image: np.ndarray,
    params: dict,
    auto_resolution: Optional[float] = None,
    spatial_ref=None,
) -> tuple[bytes, ResolutionInfo, str, str, str]:
    """
    Run the full inference pipeline synchronously.

    Parameters
    ----------
    spatial_ref : SpatialRef or None — passed through to api/output.py so
                  results can be written as georeferenced GeoTIFF when the
                  input was a GeoTIFF with embedded CRS.

    Returns
    -------
    (result_bytes, resolution_info, result_type, media_type, file_extension)
    media_type and file_extension are determined by output_format param and
    whether spatial_ref is available — callers don't need to know what
    format was actually used.
    """
    model = get_model()

    eff_res       = params.get("resolution") or auto_resolution
    result_type   = params.get("result_type", ResultType.MASK)
    processing    = params.get("processing", "raw")
    simplify      = params.get("simplify_tolerance", 0.5)
    min_area      = params.get("min_area", 10.0)
    output_format = params.get("output_format", "auto")
    coords        = params.get("coords", "auto")

    mask, info = model.predict(
        image,
        input_resolution=eff_res,
        resample=params.get("resample", True),
    )

    H, W = image.shape[:2]

    if result_type == ResultType.VECTOR:
        # vector_to_bytes handles pixel→world coordinate conversion
        # when spatial_ref is available and coords="auto"|"world"
        data, media_type, ext = vector_to_bytes(
            mask, spatial_ref, coords, simplify, min_area, eff_res
        )
        return data, info, result_type, media_type, ext

    geojson = None
    if processing in ("clean", "vectorized"):
        geojson = vectorize(mask, resolution=eff_res,
                            simplify_tolerance_m=simplify, min_area_m2=min_area)
    if processing == "clean":
        mask = polygons_to_mask(geojson, height=H, width=W)

    if result_type == ResultType.OVERLAY:
        # overlay_to_bytes writes GeoTIFF if output_format="auto"|"tif"
        # and spatial_ref is available
        data, media_type, ext = overlay_to_bytes(
            image, mask, processing, geojson, spatial_ref, output_format
        )
        return data, info, result_type, media_type, ext

    # Default: mask — mask_to_bytes writes GeoTIFF when appropriate
    data, media_type, ext = mask_to_bytes(mask, spatial_ref, output_format)
    return data, info, result_type, media_type, ext


def _mask_to_png(mask: np.ndarray) -> bytes:
    img = PILImage.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _raw_overlay(image: np.ndarray, mask: np.ndarray) -> bytes:
    overlay = image.copy()
    overlay[mask > 0] = np.clip(
        overlay[mask > 0].astype(int) * 0.5 + np.array([255, 50, 50]) * 0.5,
        0, 255,
    ).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(overlay, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _vector_overlay(image: np.ndarray, geojson: dict) -> bytes:
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
    buf = io.BytesIO()
    PILImage.alpha_composite(pil_img, overlay).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _clean_overlay(image: np.ndarray, geojson: dict, H: int, W: int) -> bytes:
    clean_mask = polygons_to_mask(geojson, height=H, width=W)
    return _raw_overlay(image.copy(), clean_mask)


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


def _resolution_headers(info: ResolutionInfo) -> dict:
    h = {}
    if info.input_resolution is not None:
        h["X-Input-Resolution"]   = f"{info.input_resolution:.4f}"
    h["X-Resampling-Applied"]     = "true" if info.resampled else "false"
    if info.resampled and info.resampled_to:
        h["X-Resampled-To"]       = f"{info.resampled_to:.4f}"
    if info.warning:
        h["X-Resolution-Warning"] = info.warning
    return h


def _result_media_type(result_type: str) -> str:
    return "application/json" if result_type == ResultType.VECTOR else "image/png"


def _result_key(job_id: str, result_type: str) -> str:
    ext = "json" if result_type == ResultType.VECTOR else "png"
    return f"{job_id}.{ext}"


# ── cache helper ──────────────────────────────────────────────────────────────

async def _check_and_serve_cache(
    session: AsyncSession,
    input_hash: str,
    result_type: str,
) -> Optional[Response]:
    """Return a cached Response if available, else None."""
    if not settings.cache_enabled:
        return None
    cached = await get_cached_result(session, input_hash)
    if not cached:
        return None
    storage = get_storage_backend()
    data    = await storage.read(cached.result_path)
    return Response(
        content=data,
        media_type=_result_media_type(result_type),
        headers={"X-Cache": "HIT"},
    )


# ── sync inference + store ────────────────────────────────────────────────────

async def _sync_predict_and_store(
    session: AsyncSession,
    image: np.ndarray,
    params: dict,
    auto_resolution: Optional[float],
    user_id: str,
    input_mode: str,
    input_ref: str,
    input_hash: Optional[str],
    spatial_ref=None,
) -> Response:
    """
    Run inference, store result, log job, return Response.

    spatial_ref is passed through to _run_inference_sync so output.py
    can write georeferenced GeoTIFF results when the input was a GeoTIFF.
    """
    import uuid
    from datetime import datetime, timezone

    job_id = str(uuid.uuid4())
    t0     = time.monotonic()

    result_bytes, info, result_type, media_type, ext = _run_inference_sync(
        image, params, auto_resolution, spatial_ref
    )
    duration = time.monotonic() - t0

    storage     = get_storage_backend()
    result_key  = _result_key(job_id, result_type)
    await storage.write(result_key, result_bytes)

    # Log job
    job = await create_job(
        session,
        user_id=user_id,
        input_mode=input_mode,
        input_ref=input_ref,
        params=params,
        result_type=result_type,
        input_hash=input_hash,
    )
    await mark_done(session, job.id, result_key, duration)

    # Store cache entry
    if input_hash and settings.cache_enabled:
        await store_cached_result(session, input_hash, result_key, result_type)

    headers = _resolution_headers(info)
    headers["X-Job-Id"]            = job.id
    headers["X-Cache"]             = "MISS"
    headers["X-Output-Format"]     = ext.lstrip(".")
    # Hint the correct filename extension so curl/browsers save with right extension
    headers["Content-Disposition"] = f'attachment; filename="result{ext}"'
    if spatial_ref is not None:
        epsg = spatial_ref.crs.to_epsg()
        if epsg:
            headers["X-Output-CRS"] = f"EPSG:{epsg}"

    return Response(
        content=result_bytes,
        media_type=media_type,
        headers=headers,
    )


# ── input loading ─────────────────────────────────────────────────────────────

async def _load_from_path(path: str) -> tuple[np.ndarray, Optional[float], Optional[SpatialRef]]:
    """
    Open a file directly — zero copy, no upload buffering.
    Returns (image, resolution, spatial_ref).
    spatial_ref is populated for GeoTIFF with projected CRS, None otherwise.
    """
    if not Path(path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found at path '{path}'. "
                   "Ensure the file is mounted into the container at this path.",
        )
    try:
        import rasterio
        spatial_ref = None
        with rasterio.open(path) as src:
            arr = src.read()
            resolution = None
            try:
                if src.crs and src.crs.is_projected:
                    res_x, res_y = src.res
                    resolution   = float((res_x + res_y) / 2)
                    # Capture CRS + transform for georeferenced output
                    spatial_ref  = SpatialRef(
                        crs=src.crs,
                        transform=src.transform,
                        width=src.width,
                        height=src.height,
                    )
            except Exception:
                pass
        image = _bands_to_rgb_uint8(arr)
        del arr
        return image, resolution, spatial_ref
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot open file: {e}")


async def _load_from_url(url: str) -> tuple[np.ndarray, Optional[float], Optional[SpatialRef]]:
    """
    Download URL in chunks to temp file — never fully in RAM.
    Returns (image, resolution, spatial_ref).
    spatial_ref populated for GeoTIFF with projected CRS, None otherwise.
    """
    try:
        import httpx
        import rasterio
        from api.inference import SpatialRef

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(1024 * 1024):
                            f.write(chunk)

            spatial_ref = None
            with rasterio.open(tmp_path) as src:
                arr = src.read()
                resolution = None
                try:
                    if src.crs and src.crs.is_projected:
                        res_x, res_y = src.res
                        resolution   = float((res_x + res_y) / 2)
                        # Capture CRS + transform for georeferenced output
                        spatial_ref  = SpatialRef(
                            crs=src.crs,
                            transform=src.transform,
                            width=src.width,
                            height=src.height,
                        )
                except Exception:
                    pass

            image = _bands_to_rgb_uint8(arr)
            del arr
            return image, resolution, spatial_ref
        finally:
            os.unlink(tmp_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot download or open URL: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SYNC ENDPOINTS — result returned directly
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/predict/upload", summary="Sync — multipart upload")
async def predict_upload(
    file:       UploadFile = File(...),
    params:     InferenceParams = Depends(),
    user_id:    str = Depends(get_user_id),
    db:         AsyncSession = Depends(db_session_dependency),
):
    """Upload an image file and receive the result directly."""
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, detail=f"File exceeds {settings.max_upload_bytes // 1024**2}MB limit.")

    try:
        image, auto_res, spatial_ref = read_image_bytes(data, filename=file.filename or "")
    except ValueError as e:
        raise HTTPException(422, detail=str(e))

    p = _build_params(params)
    eff_res    = p.get("resolution") or auto_res
    input_hash = compute_input_hash(hash_bytes(data), p)
    del data

    cached_response = await _check_and_serve_cache(db, input_hash, p["result_type"])
    if cached_response:
        return cached_response

    return await _sync_predict_and_store(
        db, image, p, auto_res, user_id,
        InputMode.UPLOAD, file.filename or "upload", input_hash,
        spatial_ref=spatial_ref,
    )


@app.post("/predict/from-path", summary="Sync — local file path")
async def predict_from_path(
    body:    FromPathRequest,
    user_id: str = Depends(get_user_id),
    db:      AsyncSession = Depends(db_session_dependency),
):
    """
    Open a file directly from the container filesystem.
    Best for large tiles — zero upload buffering, rasterio reads straight from disk.

    The file must be accessible at `body.path` inside the container.
    Mount your data directory via the `volumes` section in docker-compose.
    """
    p = _build_params(body)

    # Cache key uses path + mtime so cache invalidates when file changes
    try:
        mtime = str(Path(body.path).stat().st_mtime)
    except Exception:
        mtime = ""
    input_hash = compute_input_hash(f"{body.path}:{mtime}", p)

    cached_response = await _check_and_serve_cache(db, input_hash, p["result_type"])
    if cached_response:
        return cached_response

    image, auto_res, spatial_ref = await _load_from_path(body.path)
    return await _sync_predict_and_store(
        db, image, p, auto_res, user_id,
        InputMode.PATH, body.path, input_hash,
        spatial_ref=spatial_ref,
    )


@app.post("/predict/from-url", summary="Sync — download from URL")
async def predict_from_url(
    body:    FromUrlRequest,
    user_id: str = Depends(get_user_id),
    db:      AsyncSession = Depends(db_session_dependency),
):
    """
    Download an image from a URL and return the result directly.
    Suitable for Azure Blob SAS URLs or any publicly accessible image.
    Downloaded in chunks to minimise peak RAM.
    """
    p = _build_params(body)
    input_hash = compute_input_hash(body.url, p)

    cached_response = await _check_and_serve_cache(db, input_hash, p["result_type"])
    if cached_response:
        return cached_response

    image, auto_res, spatial_ref = await _load_from_url(body.url)
    return await _sync_predict_and_store(
        db, image, p, auto_res, user_id,
        InputMode.URL, body.url, input_hash,
        spatial_ref=spatial_ref,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ASYNC ENDPOINTS — returns job_id immediately
# ══════════════════════════════════════════════════════════════════════════════

async def _enqueue(
    session:    AsyncSession,
    user_id:    str,
    input_mode: str,
    input_ref:  str,
    params:     dict,
    input_hash: Optional[str] = None,
) -> AsyncJobAccepted:
    """Create job in DB, enqueue, return accepted response."""
    job = await create_job(
        session,
        user_id=user_id,
        input_mode=input_mode,
        input_ref=input_ref,
        params=params,
        result_type=params.get("result_type", ResultType.MASK),
        input_hash=input_hash,
    )
    await session.commit()   # commit before enqueue so worker can find the job

    if _queue:
        await _queue.enqueue(job.id)
    else:
        # Fallback: run inline (shouldn't happen in normal operation)
        import asyncio
        asyncio.create_task(run_job_async(job.id))

    return AsyncJobAccepted(
        job_id=job.id,
        poll_url=f"/jobs/{job.id}",
    )


@app.post("/predict/async/upload",
          response_model=AsyncJobAccepted,
          summary="Async — multipart upload",
          status_code=202)
async def predict_async_upload(
    file:    UploadFile = File(...),
    params:  InferenceParams = Depends(),
    user_id: str = Depends(get_user_id),
    db:      AsyncSession = Depends(db_session_dependency),
):
    """Upload a file and receive a job_id. Poll /jobs/{job_id} for result."""
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, detail="File too large.")

    # Store upload in storage so worker can retrieve it
    storage     = get_storage_backend()
    upload_key  = f"uploads/{hash_bytes(data)}/{file.filename or 'upload'}"
    await storage.write(upload_key, data)

    p = _build_params(params)
    return await _enqueue(
        db, user_id, InputMode.UPLOAD, upload_key, p, hash_bytes(data)
    )


@app.post("/predict/async/from-path",
          response_model=AsyncJobAccepted,
          summary="Async — local file path",
          status_code=202)
async def predict_async_from_path(
    body:    FromPathRequest,
    user_id: str = Depends(get_user_id),
    db:      AsyncSession = Depends(db_session_dependency),
):
    """Queue inference on a local path. Returns job_id immediately."""
    if not Path(body.path).exists():
        raise HTTPException(404, detail=f"File not found: {body.path}")
    p = _build_params(body)
    return await _enqueue(db, user_id, InputMode.PATH, body.path, p)


@app.post("/predict/async/from-url",
          response_model=AsyncJobAccepted,
          summary="Async — download from URL",
          status_code=202)
async def predict_async_from_url(
    body:    FromUrlRequest,
    user_id: str = Depends(get_user_id),
    db:      AsyncSession = Depends(db_session_dependency),
):
    """Queue inference on a URL. Returns job_id immediately."""
    p = _build_params(body)
    return await _enqueue(db, user_id, InputMode.URL, body.url, p)


# ══════════════════════════════════════════════════════════════════════════════
# JOB MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/jobs/{job_id}", response_model=JobResponse, summary="Get job status")
async def get_job_status(
    job_id:  str,
    user_id: str = Depends(get_user_id),
    db:      AsyncSession = Depends(db_session_dependency),
):
    """Poll this endpoint after submitting an async job."""
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(404, detail=f"Job {job_id} not found.")

    result_url = None
    if job.status == JobStatus.DONE and job.result_path:
        storage    = get_storage_backend()
        result_url = await storage.get_url(job.result_path)

    d = job.to_dict()
    d["result_url"] = result_url
    d["cached"]     = False
    return JobResponse(**d)


@app.get("/jobs/{job_id}/result", summary="Download job result")
async def get_job_result(
    job_id:  str,
    user_id: str = Depends(get_user_id),
    db:      AsyncSession = Depends(db_session_dependency),
):
    """
    Download the result for a completed job.

    Local storage: streams the file from disk.
    Azure storage: redirects to a time-limited SAS URL (client downloads
                   directly from Azure — API never touches the bytes again).
    """
    from fastapi.responses import RedirectResponse

    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(404, detail=f"Job {job_id} not found.")
    if job.status != JobStatus.DONE:
        raise HTTPException(
            409,
            detail=f"Job is {job.status}, not done yet. Poll /jobs/{job_id} for status.",
        )

    storage = get_storage_backend()
    url     = await storage.get_url(job.result_path)

    if settings.storage_backend == "azure":
        # Azure: redirect to SAS URL — zero bytes through the API
        return RedirectResponse(url=url)

    # Local: stream from disk
    data = await storage.read(job.result_path)
    return Response(
        content=data,
        media_type=_result_media_type(job.result_type),
    )


@app.get("/jobs/", summary="List recent jobs")
async def list_recent_jobs(
    limit:   int = Query(20, ge=1, le=100),
    offset:  int = Query(0, ge=0),
    user_id: str = Depends(get_user_id),
    db:      AsyncSession = Depends(db_session_dependency),
):
    """List the most recent jobs for the current user."""
    jobs = await list_jobs(db, user_id=user_id, limit=limit, offset=offset)
    return [j.to_dict() for j in jobs]


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL RESULT STREAMING
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/results/{key}", summary="Stream result file (local storage only)")
async def get_result_file(key: str):
    """
    Stream a result file from local storage.
    Only relevant when STORAGE_BACKEND=local.
    In Azure mode, clients use the SAS URL returned by /jobs/{job_id}.
    """
    storage = get_storage_backend()
    if not await storage.exists(key):
        raise HTTPException(404, detail=f"Result '{key}' not found.")
    data = await storage.read(key)
    ext  = Path(key).suffix.lower()
    media_type = "application/json" if ext == ".json" else "image/png"
    return Response(content=data, media_type=media_type)


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health(db: AsyncSession = Depends(db_session_dependency)):
    """Check DB, storage, and model reachability."""
    # DB
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {e}"

    # Storage
    try:
        storage        = get_storage_backend()
        storage_status = "ok" if await storage.health_check() else "error"
    except Exception as e:
        storage_status = f"error: {e}"

    # Model
    try:
        get_model()
        model_status = "ok"
    except Exception as e:
        model_status = f"error: {e}"

    overall = "ok" if all(
        s == "ok" for s in [db_status, storage_status, model_status]
    ) else "degraded"

    return HealthResponse(
        status=overall,
        database=db_status,
        storage=storage_status,
        model=model_status,
    )


@app.get("/", include_in_schema=False)
def root():
    return {"message": f"{settings.app_name} — see /docs for usage."}