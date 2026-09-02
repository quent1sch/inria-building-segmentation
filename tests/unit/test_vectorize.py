"""
tests/unit/test_vectorize.py

Unit tests for api/vectorize.py.

Tested:
  - vectorize: polygon extraction, area filtering, simplification
  - polygons_to_mask: roundtrip rasterization from GeoJSON back to mask
  - geojson_to_bytes: serialisation

Key property tested: polygons_to_mask(vectorize(mask)) ≈ mask
(not exact due to simplification, but coverage should be high)
"""

import json

import numpy as np
import pytest
from shapely.geometry import shape

from api.vectorize import (
    _mask_to_shapely_polygons,
    geojson_to_bytes,
    polygons_to_mask,
    vectorize,
)


# ── vectorize ─────────────────────────────────────────────────────────────────

class TestVectorize:

    def test_empty_mask_returns_no_features(self, empty_mask):
        result = vectorize(empty_mask, resolution=1.0,
                           simplify_tolerance_m=0.0, min_area_m2=0.0)
        assert result["type"] == "FeatureCollection"
        assert result["features"] == []
        assert result["metadata"]["n_buildings"] == 0

    def test_two_buildings_detected(self, binary_mask_two_buildings):
        """Two distinct blobs should produce two polygon features."""
        result = vectorize(binary_mask_two_buildings, resolution=1.0,
                           simplify_tolerance_m=0.0, min_area_m2=0.0)
        assert len(result["features"]) == 2

    def test_area_filter_removes_small_blob(self, binary_mask_two_buildings):
        """
        binary_mask_two_buildings has:
          building 1: 30×30 = 900 px → 900 m² at res=1.0
          building 2: 20×20 = 400 px → 400 m²
        min_area=500 → only building 1 survives.
        """
        result = vectorize(binary_mask_two_buildings, resolution=1.0,
                           simplify_tolerance_m=0.0, min_area_m2=500.0)
        assert len(result["features"]) == 1

    def test_area_filter_requires_resolution(self, binary_mask_two_buildings):
        """
        Without resolution, area filter is skipped — both buildings returned
        even with a large min_area_m2.
        """
        result = vectorize(binary_mask_two_buildings, resolution=None,
                           simplify_tolerance_m=0.0, min_area_m2=9999.0)
        # min_area skipped because resolution is None
        assert len(result["features"]) == 2

    def test_feature_has_area_px_property(self, binary_mask_two_buildings):
        """Every feature should have area_px in its properties."""
        result = vectorize(binary_mask_two_buildings, resolution=1.0,
                           simplify_tolerance_m=0.0, min_area_m2=0.0)
        for feature in result["features"]:
            assert "area_px" in feature["properties"]
            assert feature["properties"]["area_px"] > 0

    def test_feature_has_area_m2_when_resolution_known(self, binary_mask_two_buildings):
        """area_m2 should be present when resolution is provided."""
        result = vectorize(binary_mask_two_buildings, resolution=0.1,
                           simplify_tolerance_m=0.0, min_area_m2=0.0)
        for feature in result["features"]:
            assert "area_m2" in feature["properties"]
            # area_m2 = area_px * resolution² = area_px * 0.01
            expected = feature["properties"]["area_px"] * 0.1 ** 2
            assert feature["properties"]["area_m2"] == pytest.approx(expected, rel=1e-4)

    def test_no_area_m2_without_resolution(self, binary_mask_two_buildings):
        """area_m2 should NOT be present when resolution is None."""
        result = vectorize(binary_mask_two_buildings, resolution=None,
                           simplify_tolerance_m=0.0, min_area_m2=0.0)
        for feature in result["features"]:
            assert "area_m2" not in feature["properties"]

    def test_metadata_contains_n_buildings(self, binary_mask_two_buildings):
        result = vectorize(binary_mask_two_buildings, resolution=1.0,
                           simplify_tolerance_m=0.0, min_area_m2=0.0)
        assert result["metadata"]["n_buildings"] == 2

    def test_geometry_type_is_polygon_or_multipolygon(self, binary_mask_two_buildings):
        """All features should have Polygon or MultiPolygon geometry."""
        result = vectorize(binary_mask_two_buildings, resolution=1.0,
                           simplify_tolerance_m=0.0, min_area_m2=0.0)
        for feature in result["features"]:
            assert feature["geometry"]["type"] in ("Polygon", "MultiPolygon")

    def test_full_mask_produces_one_polygon(self, full_mask):
        """All-True mask → one large polygon covering the image."""
        result = vectorize(full_mask, resolution=1.0,
                           simplify_tolerance_m=0.0, min_area_m2=0.0)
        assert len(result["features"]) >= 1


# ── polygons_to_mask ──────────────────────────────────────────────────────────

class TestPolygonsToMask:

    def test_roundtrip_approximate(self, binary_mask_two_buildings):
        """
        vectorize → polygons_to_mask should reproduce the original mask
        with high overlap (IoU > 0.9) when simplification is disabled.
        """
        H, W = binary_mask_two_buildings.shape
        geojson = vectorize(binary_mask_two_buildings, resolution=1.0,
                            simplify_tolerance_m=0.0, min_area_m2=0.0)
        recovered = polygons_to_mask(geojson, height=H, width=W)

        pred = recovered.ravel()
        gt   = binary_mask_two_buildings.ravel()
        tp   = (pred & gt).sum()
        fp   = (pred & ~gt).sum()
        fn   = (~pred & gt).sum()
        iou  = tp / (tp + fp + fn + 1e-6)
        assert iou > 0.90, f"Roundtrip IoU too low: {iou:.3f}"

    def test_empty_geojson_gives_empty_mask(self, empty_geojson):
        """Empty GeoJSON → all-False mask."""
        result = polygons_to_mask(empty_geojson, height=64, width=64)
        assert result.shape == (64, 64)
        assert not result.any()

    def test_output_shape_matches_requested(self, binary_mask_two_buildings):
        """Output mask dimensions should match the requested H×W."""
        H, W = binary_mask_two_buildings.shape
        geojson = vectorize(binary_mask_two_buildings, resolution=1.0,
                            simplify_tolerance_m=0.0, min_area_m2=0.0)
        result = polygons_to_mask(geojson, height=H, width=W)
        assert result.shape == (H, W)

    def test_output_is_bool(self, binary_mask_two_buildings):
        """polygons_to_mask should return a bool array."""
        H, W = binary_mask_two_buildings.shape
        geojson = vectorize(binary_mask_two_buildings, resolution=1.0,
                            simplify_tolerance_m=0.0, min_area_m2=0.0)
        result = polygons_to_mask(geojson, height=H, width=W)
        assert result.dtype == bool

    def test_simple_geojson_fills_correctly(self, simple_geojson):
        """
        simple_geojson has one polygon: rows 20-50, cols 20-50 (pixel coords).
        The rasterized mask should be True in that region.
        """
        result = polygons_to_mask(simple_geojson, height=128, width=128)
        # Centre of the polygon
        assert result[35, 35] == True
        # Far corner — background
        assert result[0, 0]   == False


# ── _mask_to_shapely_polygons ─────────────────────────────────────────────────

class TestMaskToShapelyPolygons:

    def test_two_blobs_two_polygons(self, binary_mask_two_buildings):
        polys = _mask_to_shapely_polygons(binary_mask_two_buildings)
        assert len(polys) == 2

    def test_empty_mask_no_polygons(self, empty_mask):
        polys = _mask_to_shapely_polygons(empty_mask)
        assert polys == []

    def test_polygons_are_valid_shapely(self, binary_mask_two_buildings):
        """All returned polygons should be valid Shapely geometries."""
        polys = _mask_to_shapely_polygons(binary_mask_two_buildings)
        for p in polys:
            assert p.is_valid
            assert not p.is_empty
            assert p.area > 0

    def test_polygon_area_matches_blob_size(self, binary_mask_one_building):
        """
        binary_mask_one_building has a 40×40 blob = 1600 pixels.
        Polygon area should be close to 1600 (pixel² units).
        """
        polys = _mask_to_shapely_polygons(binary_mask_one_building)
        assert len(polys) == 1
        # Contour approximation may lose a few border pixels
        assert polys[0].area == pytest.approx(1600, rel=0.05)


# ── geojson_to_bytes ──────────────────────────────────────────────────────────

class TestGeojsonToBytes:

    def test_returns_valid_utf8_json(self, simple_geojson):
        data = geojson_to_bytes(simple_geojson)
        assert isinstance(data, bytes)
        parsed = json.loads(data.decode("utf-8"))
        assert parsed["type"] == "FeatureCollection"

    def test_empty_geojson_serialises(self, empty_geojson):
        data = geojson_to_bytes(empty_geojson)
        parsed = json.loads(data)
        assert parsed["features"] == []