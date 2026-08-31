"""
worker/job_runner.py

Job runner — the actual inference pipeline executed per job.

Called by:
  - LocalThreadQueue: run_job(job_id) called in a thread pool
  - AzureQueueStorage: run_job(job_id) called in the worker container

The runner:
  1. Loads the job from DB
  2. Marks it as processing
  3. Reads the input (from path, URL, or stored upload)
  4. Runs inference
  5. Writes result to storage
  6. Stores cache entry
  7. Marks job as done (or failed)

Can also be run as a standalone process (worker container):
    python -m worker.job_runner
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from config import get_settings
from db.crud import (
    compute_input_hash,
    get_job,
    mark_done,
    mark_failed,
    mark_processing,
    store_cached_result,
)
from db.models import InputMode, ResultType
from db.session import get_session
from storage import get_storage


# ── result key generation ─────────────────────────────────────────────────────

def _result_key(job_id: str, result_type: str) -> str:
    ext = {"mask": "png", "overlay": "png", "vector": "json"}.get(result_type, "bin")
    return f"{job_id}.{ext}"


# ── input loading ─────────────────────────────────────────────────────────────

async def _load_input_from_path(path: str):
    """Open a local file directly with rasterio — zero copy overhead."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read()
        resolution = None
        try:
            if src.crs and src.crs.is_projected:
                res_x, res_y = src.res
                resolution   = float((res_x + res_y) / 2)
        except Exception:
            pass

    from api.inference import _bands_to_rgb_uint8
    image = _bands_to_rgb_uint8(arr)
    return image, resolution


async def _load_input_from_url(url: str):
    """Download URL in chunks to a temp file, open with rasterio."""
    import httpx
    import numpy as np
    import rasterio
    from api.inference import _bands_to_rgb_uint8

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)

        with rasterio.open(tmp_path) as src:
            arr = src.read()
            resolution = None
            try:
                if src.crs and src.crs.is_projected:
                    res_x, res_y = src.res
                    resolution   = float((res_x + res_y) / 2)
            except Exception:
                pass

        image = _bands_to_rgb_uint8(arr)
        return image, resolution
    finally:
        os.unlink(tmp_path)


async def _load_input_from_storage(storage, input_ref: str):
    """Load a previously uploaded file from storage."""
    data = await storage.read(input_ref)
    from api.inference import read_image_bytes
    return read_image_bytes(data, filename=input_ref)


# ── inference ──────────────────────────────────────────────────────────────────

def _run_inference(image, resolution, params: dict) -> bytes:
    """
    Run the full inference pipeline and return result as bytes.
    This is CPU-bound and runs in a thread pool in local mode.
    """
    from api.inference import SegmentationInference
    from api.vectorize import vectorize, polygons_to_mask
    import json

    settings = get_settings()
    model = _get_model(settings.checkpoint_path, settings.config_path)

    # Extract params
    input_resolution     = params.get("resolution", resolution)
    resample             = params.get("resample", True)
    processing           = params.get("processing", "raw")
    simplify_tolerance   = params.get("simplify_tolerance", 0.5)
    min_area             = params.get("min_area", 10.0)
    result_type          = params.get("result_type", ResultType.MASK)

    mask, info = model.predict(
        image,
        input_resolution=input_resolution,
        resample=resample,
    )

    H, W = image.shape[:2]

    if result_type == ResultType.VECTOR:
        geojson = vectorize(
            mask, resolution=input_resolution,
            simplify_tolerance_m=simplify_tolerance,
            min_area_m2=min_area,
        )
        return json.dumps(geojson).encode()

    if processing == "clean":
        geojson = vectorize(
            mask, resolution=input_resolution,
            simplify_tolerance_m=simplify_tolerance,
            min_area_m2=min_area,
        )
        mask = polygons_to_mask(geojson, height=H, width=W)

    # Mask or overlay → PNG
    from PIL import Image as PILImage
    import numpy as np

    if result_type == ResultType.OVERLAY:
        from api.main import _raw_overlay, _vector_overlay, _clean_overlay
        if processing == "raw":
            png_bytes = _raw_overlay(image.copy(), mask)
        elif processing == "vectorized":
            geojson   = vectorize(mask, resolution=input_resolution,
                                  simplify_tolerance_m=simplify_tolerance,
                                  min_area_m2=min_area)
            png_bytes = _vector_overlay(image, geojson)
        else:
            geojson   = vectorize(mask, resolution=input_resolution,
                                  simplify_tolerance_m=simplify_tolerance,
                                  min_area_m2=min_area)
            png_bytes = _clean_overlay(image, geojson, H, W)
        return png_bytes

    # Default: mask PNG
    img = PILImage.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Model singleton — loaded once per worker process
_model_instance = None

def _get_model(checkpoint_path: str, config_path: str):
    global _model_instance
    if _model_instance is None:
        from api.inference import SegmentationInference
        _model_instance = SegmentationInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
        )
    return _model_instance


# ── main runner ───────────────────────────────────────────────────────────────

async def run_job_async(job_id: str) -> None:
    """
    Async version of the job runner. Used by the async endpoints.
    """
    storage = get_storage()
    t0      = time.monotonic()

    async with get_session() as session:
        job = await get_job(session, job_id)
        if job is None:
            print(f"Worker: job {job_id} not found in DB")
            return
        await mark_processing(session, job_id)

    try:
        # ── load input ────────────────────────────────────────────────────
        async with get_session() as session:
            job = await get_job(session, job_id)

        if job.input_mode == InputMode.PATH:
            image, resolution = await _load_input_from_path(job.input_ref)
        elif job.input_mode == InputMode.URL:
            image, resolution = await _load_input_from_url(job.input_ref)
        else:  # upload — stored in storage layer
            image, resolution = await _load_input_from_storage(storage, job.input_ref)

        # ── run inference (in thread to avoid blocking event loop) ────────
        loop       = asyncio.get_running_loop()
        result_bytes = await loop.run_in_executor(
            None, _run_inference, image, resolution, job.params
        )
        del image

        # ── store result ──────────────────────────────────────────────────
        result_key = _result_key(job_id, job.result_type)
        await storage.write(result_key, result_bytes)
        duration = time.monotonic() - t0

        async with get_session() as session:
            await mark_done(session, job_id, result_key, duration)
            if job.input_hash:
                await store_cached_result(
                    session, job.input_hash, result_key, job.result_type
                )

        print(f"Worker: job {job_id} done in {duration:.1f}s")

    except Exception as e:
        async with get_session() as session:
            await mark_failed(session, job_id, str(e))
        print(f"Worker: job {job_id} failed — {e}")
        raise


def run_job(job_id: str) -> None:
    """
    Sync wrapper for thread pool execution (LocalThreadQueue).
    Creates its own event loop since thread pool threads don't have one.
    """
    asyncio.run(run_job_async(job_id))


# ── standalone worker entry point ─────────────────────────────────────────────

async def _worker_main() -> None:
    """
    Standalone worker process entry point.
    Used when WORKER_MODE=queue (separate container).

    CMD in docker-compose for worker service:
        python -m worker.job_runner
    """
    from worker.queue import get_queue
    from db.session import init_db

    await init_db()
    queue = get_queue()
    print("Worker started. Waiting for jobs...")
    await queue.start(run_job)


if __name__ == "__main__":
    asyncio.run(_worker_main())