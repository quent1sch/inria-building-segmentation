"""
tests/unit/test_output.py

Unit tests for api/output.py.

Tested:
  - resolve_output_format: "auto" logic based on spatial_ref presence
  - resolve_coords: "auto" logic for vector coordinate space
  - mask_to_bytes: correct format, media type, extension, content
  - overlay_to_bytes: PNG/TIFF output, correct shape
  - _pixels_to_world: coordinate transformation using affine transform

Not tested here (require rasterio CRS objects for full GeoTIFF writing):
  - GeoTIFF output with actual CRS — tested implicitly via mock_spatial_ref
    in integration tests
"""

import io
import json

import numpy as np
import pytest
from PIL import Image as PILImage

from api.output import (
    _pixels_to_world,
    mask_to_bytes,
    overlay_to_bytes,
    resolve_coords,
    resolve_output_format,
    vector_to_bytes,
)


# ── resolve_output_format ─────────────────────────────────────────────────────

class TestResolveOutputFormat:

    def test_auto_with_spatial_ref_gives_tif(self, mock_spatial_ref):
        assert resolve_output_format("auto", mock_spatial_ref) == "tif"

    def test_auto_without_spatial_ref_gives_png(self):
        assert resolve_output_format("auto", None) == "png"

    def test_explicit_png_always_png(self, mock_spatial_ref):
        """Explicit png overrides even when spatial_ref is present."""
        assert resolve_output_format("png", mock_spatial_ref) == "png"
        assert resolve_output_format("png", None)             == "png"

    def test_explicit_tif_always_tif(self):
        """Explicit tif regardless of spatial_ref."""
        assert resolve_output_format("tif", None)  == "tif"


# ── resolve_coords ────────────────────────────────────────────────────────────

class TestResolveCoords:

    def test_auto_with_spatial_ref_gives_world(self, mock_spatial_ref):
        assert resolve_coords("auto", mock_spatial_ref) == "world"

    def test_auto_without_spatial_ref_gives_pixel(self):
        assert resolve_coords("auto", None) == "pixel"

    def test_explicit_world(self):
        assert resolve_coords("world", None) == "world"

    def test_explicit_pixel(self, mock_spatial_ref):
        assert resolve_coords("pixel", mock_spatial_ref) == "pixel"


# ── mask_to_bytes ─────────────────────────────────────────────────────────────

class TestMaskToBytes:

    def test_png_output_format_and_media_type(self, binary_mask_two_buildings):
        """output_format=png → PNG bytes, image/png, .png extension."""
        data, media_type, ext = mask_to_bytes(
            binary_mask_two_buildings, spatial_ref=None, output_format="png"
        )
        assert media_type == "image/png"
        assert ext        == ".png"
        # Verify bytes are a valid PNG
        img = PILImage.open(io.BytesIO(data))
        assert img.format == "PNG"

    def test_tif_output_without_spatial_ref(self, binary_mask_two_buildings):
        """output_format=tif, no spatial_ref → plain TIFF (no CRS)."""
        data, media_type, ext = mask_to_bytes(
            binary_mask_two_buildings, spatial_ref=None, output_format="tif"
        )
        assert media_type == "image/tiff"
        assert ext        == ".tif"
        img = PILImage.open(io.BytesIO(data))
        assert img.format == "TIFF"

    def test_auto_without_spatial_ref_gives_png(self, binary_mask_two_buildings):
        data, media_type, ext = mask_to_bytes(
            binary_mask_two_buildings, spatial_ref=None, output_format="auto"
        )
        assert media_type == "image/png"
        assert ext        == ".png"

    def test_png_mask_pixel_values(self, binary_mask_two_buildings):
        """
        Building pixels should be white (255) and background black (0) in the PNG.
        """
        data, _, _ = mask_to_bytes(
            binary_mask_two_buildings, spatial_ref=None, output_format="png"
        )
        img = PILImage.open(io.BytesIO(data)).convert("L")
        arr = np.array(img)
        # Where GT is True, output should be 255
        assert arr[25, 25] == 255  # centre of building 1 (rows 20-49, cols 20-49)
        assert arr[0, 0]   == 0    # background

    def test_returns_bytes(self, binary_mask_two_buildings):
        data, _, _ = mask_to_bytes(
            binary_mask_two_buildings, spatial_ref=None, output_format="png"
        )
        assert isinstance(data, bytes)
        assert len(data) > 0


# ── overlay_to_bytes ──────────────────────────────────────────────────────────

class TestOverlayToBytes:

    def test_raw_overlay_png(self, rgb_image_with_buildings, binary_mask_two_buildings):
        """Raw overlay → PNG, buildings should be reddish."""
        data, media_type, ext = overlay_to_bytes(
            rgb_image_with_buildings,
            binary_mask_two_buildings,
            processing="raw",
            geojson=None,
            spatial_ref=None,
            output_format="png",
        )
        assert media_type == "image/png"
        assert ext        == ".png"
        img = PILImage.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img)
        # Building pixel (25,25) should have more red than the background pixel (0,0)
        assert arr[25, 25, 0] > arr[0, 0, 0]   # R channel higher on building

    def test_overlay_output_same_spatial_size(
        self, rgb_image_with_buildings, binary_mask_two_buildings
    ):
        """Output image should have same H×W as input."""
        H, W = rgb_image_with_buildings.shape[:2]
        data, _, _ = overlay_to_bytes(
            rgb_image_with_buildings,
            binary_mask_two_buildings,
            processing="raw",
            geojson=None,
            spatial_ref=None,
            output_format="png",
        )
        img = PILImage.open(io.BytesIO(data))
        assert img.size == (W, H)   # PIL size is (W, H)

    def test_clean_overlay_uses_geojson_mask(
        self, rgb_image_with_buildings, binary_mask_two_buildings, simple_geojson
    ):
        """
        processing=clean with a geojson → rasterizes geojson back to mask.
        Should not crash and should return valid PNG bytes.
        """
        data, media_type, _ = overlay_to_bytes(
            rgb_image_with_buildings,
            binary_mask_two_buildings,
            processing="clean",
            geojson=simple_geojson,
            spatial_ref=None,
            output_format="png",
        )
        assert media_type == "image/png"
        img = PILImage.open(io.BytesIO(data))
        assert img.size == (128, 128)

    def test_tif_output_without_spatial_ref(
        self, rgb_image_with_buildings, binary_mask_two_buildings
    ):
        """output_format=tif, no spatial_ref → plain TIFF."""
        data, media_type, ext = overlay_to_bytes(
            rgb_image_with_buildings,
            binary_mask_two_buildings,
            processing="raw",
            geojson=None,
            spatial_ref=None,
            output_format="tif",
        )
        assert media_type == "image/tiff"
        assert ext        == ".tif"


# ── _pixels_to_world ──────────────────────────────────────────────────────────

class TestPixelsToWorld:

    def test_coordinate_transform_applied(self, simple_geojson, mock_spatial_ref):
        """
        Pixel coordinate (col=20, row=20) should map to world coordinate
        (x = x_origin + col * scale, y = y_origin + row * scale).

        mock_spatial_ref has:
          transform = Affine(0.1, 0, 2600000, 0, -0.1, 1200000)
          so (col=20, row=20) → x = 2600000 + 20*0.1 = 2600002.0
                                  y = 1200000 + 20*(-0.1) = 1199998.0
        """
        result = _pixels_to_world(simple_geojson, mock_spatial_ref)
        coords = result["features"][0]["geometry"]["coordinates"][0]
        # First coordinate of the polygon exterior was [20, 20] in pixel space
        x, y = coords[0][0], coords[0][1]
        assert x == pytest.approx(2600002.0, abs=0.01)
        assert y == pytest.approx(1199998.0, abs=0.01)

    def test_crs_member_added(self, simple_geojson, mock_spatial_ref):
        """Output GeoJSON should include a 'crs' member with the EPSG code."""
        result = _pixels_to_world(simple_geojson, mock_spatial_ref)
        assert "crs" in result
        assert "2056" in result["crs"]["properties"]["name"]

    def test_feature_count_preserved(self, simple_geojson, mock_spatial_ref):
        """Number of features should not change after coordinate transform."""
        result = _pixels_to_world(simple_geojson, mock_spatial_ref)
        assert len(result["features"]) == len(simple_geojson["features"])

    def test_empty_geojson_no_crash(self, empty_geojson, mock_spatial_ref):
        """Empty feature collection should transform without error."""
        result = _pixels_to_world(empty_geojson, mock_spatial_ref)
        assert result["features"] == []

    def test_metadata_coords_updated(self, simple_geojson, mock_spatial_ref):
        """metadata.coords should be updated to 'world'."""
        result = _pixels_to_world(simple_geojson, mock_spatial_ref)
        assert result["metadata"]["coords"] == "world"


# ── vector_to_bytes ───────────────────────────────────────────────────────────

class TestVectorToBytes:

    def test_pixel_coords_returns_geojson(self, binary_mask_two_buildings):
        """coords=pixel → valid GeoJSON bytes."""
        data, media_type, ext = vector_to_bytes(
            binary_mask_two_buildings,
            spatial_ref=None,
            coords="pixel",
            simplify_tolerance_m=0.5,
            min_area_m2=0.0,   # no area filter so both buildings appear
            resolution=1.0,
        )
        assert media_type == "application/json"
        assert ext        == ".geojson"
        geojson = json.loads(data)
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) >= 1   # at least one building detected

    def test_world_coords_without_spatial_ref_falls_back_to_pixel(
        self, binary_mask_two_buildings
    ):
        """
        coords=world but no spatial_ref → resolve_coords returns "world" but
        _pixels_to_world is not called (spatial_ref is None).
        Should return pixel-space GeoJSON without CRS member.
        """
        data, _, _ = vector_to_bytes(
            binary_mask_two_buildings,
            spatial_ref=None,
            coords="world",
            simplify_tolerance_m=0.5,
            min_area_m2=0.0,
            resolution=1.0,
        )
        geojson = json.loads(data)
        # No CRS member because spatial_ref was None
        assert "crs" not in geojson

    def test_empty_mask_returns_empty_features(self, empty_mask):
        """Empty mask → GeoJSON with no features."""
        data, _, _ = vector_to_bytes(
            empty_mask,
            spatial_ref=None,
            coords="pixel",
            simplify_tolerance_m=0.5,
            min_area_m2=0.0,
            resolution=1.0,
        )
        geojson = json.loads(data)
        assert geojson["features"] == []