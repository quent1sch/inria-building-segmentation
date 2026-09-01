"""
api/schemas.py

Pydantic models for API request bodies and responses.
Keeps main.py clean and makes the API contract explicit.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl


# ── inference params shared across all input modes ────────────────────────────

class InferenceParams(BaseModel):
    resolution:          Optional[float] = Field(None,  gt=0, description="Input resolution in m/px. Auto-detected for GeoTIFF.")
    resample:            bool            = Field(True,        description="Resample to training resolution (0.3m/px) if finer.")
    processing:          Literal["raw", "clean", "vectorized"] = Field("raw", description="Output processing level.")
    result_type:         Literal["mask", "overlay", "vector"]  = Field("mask", description="Output format.")
    simplify_tolerance:  float           = Field(0.5,   gt=0, description="Douglas-Peucker tolerance in metres.")
    min_area:            float           = Field(10.0,  gt=0, description="Minimum building area in m².")

    # ── output format ─────────────────────────────────────────────────────
    # "auto": GeoTIFF if input was a GeoTIFF with embedded CRS, PNG otherwise.
    # Explicit "tif" forces GeoTIFF output regardless of input format.
    # Explicit "png" forces PNG even when spatial metadata is available.
    # Note: GeoTIFF output preserves the input CRS and affine transform so
    # the mask can be loaded directly in QGIS or any GIS tool.
    output_format: Literal["auto", "png", "tif"] = Field(
        "auto",
        description=(
            "Output file format. "
            "'auto' = GeoTIFF if input had embedded CRS, PNG otherwise. "
            "'tif' = always GeoTIFF (no CRS if input had none). "
            "'png' = always PNG (spatial metadata discarded)."
        ),
    )

    # ── vector coordinate space ───────────────────────────────────────────
    # Only applies when result_type=vector.
    # "auto": world coordinates if spatial_ref available, pixel otherwise.
    # "world": CRS coordinates (metres in LV95, degrees in WGS84, etc.)
    #          → GeoJSON loadable directly in QGIS, GeoPandas, etc.
    # "pixel": (col, row) pixel coordinates (previous default behaviour).
    coords: Literal["auto", "world", "pixel"] = Field(
        "auto",
        description=(
            "Coordinate space for vector output (result_type=vector only). "
            "'auto' = world coordinates if input had CRS, pixel otherwise. "
            "'world' = real-world CRS coordinates. "
            "'pixel' = pixel (col, row) coordinates."
        ),
    )

# ── input mode request bodies ─────────────────────────────────────────────────

class FromPathRequest(InferenceParams):
    path: str = Field(..., description="Absolute path to image file inside the container (e.g. /data/tile.tif).")


class FromUrlRequest(InferenceParams):
    url: str = Field(..., description="Publicly accessible URL to image file (e.g. Azure Blob SAS URL).")


# ── responses ─────────────────────────────────────────────────────────────────

class JobResponse(BaseModel):
    job_id:      str
    status:      str
    user_id:     str
    input_mode:  str
    input_ref:   str
    result_type: str
    params:      dict
    result_url:  Optional[str] = None
    error:       Optional[str] = None
    created_at:  Optional[str] = None
    started_at:  Optional[str] = None
    completed_at: Optional[str] = None
    duration_s:  Optional[float] = None
    cached:      bool = False


class AsyncJobAccepted(BaseModel):
    job_id:      str
    status:      str = "queued"
    poll_url:    str
    message:     str = "Job queued. Poll poll_url for status."


class HealthResponse(BaseModel):
    status:   str
    database: str
    storage:  str
    model:    str