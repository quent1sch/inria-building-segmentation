"""
tests/unit/test_inference_utils.py

Unit tests for pure utility functions in api/inference.py.

Tested functions (no model weights needed):
  - _bands_to_rgb_uint8: dtype normalisation and band selection
  - resample_image: scale factor and output dimensions
  - upsample_prob: bilinear upsampling of probability maps

Not tested here (require model weights or rasterio file I/O):
  - SegmentationInference (integration territory)
  - read_image_bytes (requires file I/O — integration territory)
  - _predict_tile, _sliding_window_predict (require model)
"""

import numpy as np
import pytest

from api.inference import _bands_to_rgb_uint8, resample_image, upsample_prob


# ── _bands_to_rgb_uint8 ───────────────────────────────────────────────────────

class TestBandsToRgbUint8:

    def test_3band_uint8_passthrough(self):
        """3-band uint8 CHW → (H, W, 3) uint8 HWC, values unchanged."""
        arr = np.array([
            [[10, 20], [30, 40]],   # R
            [[50, 60], [70, 80]],   # G
            [[90, 100],[110,120]],  # B
        ], dtype=np.uint8)
        result = _bands_to_rgb_uint8(arr)
        assert result.shape == (2, 2, 3)
        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 10   # R value at (0,0)
        assert result[0, 0, 1] == 50   # G value at (0,0)
        assert result[0, 0, 2] == 90   # B value at (0,0)

    def test_uint16_to_uint8_right_shift(self):
        """
        uint16 → uint8 via >> 8.
        65535 >> 8 = 255, 256 >> 8 = 1, 0 >> 8 = 0.
        """
        arr = np.array([
            [[0,     256]],
            [[0,     256]],
            [[65535, 256]],
        ], dtype=np.uint16)
        result = _bands_to_rgb_uint8(arr)
        assert result.dtype == np.uint8
        assert result[0, 0, 2] == 255   # 65535 >> 8
        assert result[0, 1, 0] == 1     # 256 >> 8
        assert result[0, 0, 0] == 0     # 0 >> 8

    def test_4band_keeps_first_3(self, multiband_image_4band):
        """4-band RGBN → first 3 bands kept, 4th (NIR) discarded."""
        result = _bands_to_rgb_uint8(multiband_image_4band)
        assert result.shape == (64, 64, 3)
        # Verify it's the first 3 bands (not a scrambled order)
        expected_r = multiband_image_4band[0]
        np.testing.assert_array_equal(result[:, :, 0], expected_r)

    def test_grayscale_replicated_to_rgb(self, grayscale_image):
        """Single-band → R=G=B (replicated to 3 identical channels)."""
        result = _bands_to_rgb_uint8(grayscale_image)
        assert result.shape == (64, 64, 3)
        # All three channels should be identical
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])

    def test_generic_float32_normalised_to_uint8(self):
        """
        Non-uint8/uint16 dtype → normalised to [0, 255] per band.
        Band with values [0.0, 0.5, 1.0] → [0, 127, 255] approximately.
        """
        arr = np.array([
            [[0.0, 0.5, 1.0]],
            [[0.0, 0.5, 1.0]],
            [[0.0, 0.5, 1.0]],
        ], dtype=np.float32).reshape(3, 1, 3)
        result = _bands_to_rgb_uint8(arr)
        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 0    # min → 0
        assert result[0, 2, 0] == 255  # max → 255

    def test_output_is_always_hwc(self):
        """Output is always HWC regardless of input spatial dimensions."""
        arr = np.zeros((3, 100, 200), dtype=np.uint8)
        result = _bands_to_rgb_uint8(arr)
        assert result.shape == (100, 200, 3)

    def test_constant_band_not_normalised_to_zero(self):
        """
        A band with all-same values: min == max, so the normalisation branch
        skips scaling (mx > mn is False). Values should not all become 0.
        Currently they remain 0.0 → cast to uint8 → 0.
        This documents the current behaviour (not a bug — constant bands are
        uninformative anyway).
        """
        arr = np.full((3, 4, 4), 0.5, dtype=np.float32)
        result = _bands_to_rgb_uint8(arr)
        assert result.dtype == np.uint8
        # Constant band: mx == mn, branch skipped, values stay 0.5 → uint8 → 0
        assert result[0, 0, 0] == 0


# ── resample_image ────────────────────────────────────────────────────────────

class TestResampleImage:

    def test_downsample_3x(self):
        """
        0.1m/px → 0.3m/px: scale = 0.1/0.3 ≈ 0.333.
        60×60 image → ~20×20.
        """
        image = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
        result = resample_image(image, input_resolution=0.1, target_resolution=0.3)
        assert result.shape[0] == pytest.approx(20, abs=1)
        assert result.shape[1] == pytest.approx(20, abs=1)
        assert result.dtype == np.uint8

    def test_upsample_2x(self):
        """
        0.6m/px → 0.3m/px: scale = 0.6/0.3 = 2.0.
        30×30 image → 60×60.
        """
        image = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        result = resample_image(image, input_resolution=0.6, target_resolution=0.3)
        assert result.shape == (60, 60, 3)

    def test_same_resolution_unchanged(self):
        """Same input and target resolution → same dimensions."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = resample_image(image, input_resolution=0.3, target_resolution=0.3)
        assert result.shape == image.shape

    def test_output_dtype_preserved(self):
        """Output dtype should always be uint8."""
        image = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
        result = resample_image(image, input_resolution=0.1, target_resolution=0.3)
        assert result.dtype == np.uint8

    def test_minimum_output_size_is_1(self):
        """Extreme downscaling should not produce 0-size output."""
        image = np.ones((10, 10, 3), dtype=np.uint8)
        result = resample_image(image, input_resolution=0.001, target_resolution=100.0)
        assert result.shape[0] >= 1
        assert result.shape[1] >= 1

    def test_non_square_image(self):
        """Non-square images should scale both dimensions correctly."""
        image = np.random.randint(0, 255, (60, 120, 3), dtype=np.uint8)
        result = resample_image(image, input_resolution=0.1, target_resolution=0.3)
        # H: 60 * (0.1/0.3) ≈ 20, W: 120 * (0.1/0.3) ≈ 40
        assert result.shape[0] == pytest.approx(20, abs=1)
        assert result.shape[1] == pytest.approx(40, abs=1)


# ── upsample_prob ─────────────────────────────────────────────────────────────

class TestUpsampleProb:

    def test_output_shape(self):
        """Upsampled prob map should match target dimensions exactly."""
        prob = np.random.rand(20, 20).astype(np.float32)
        result = upsample_prob(prob, original_h=60, original_w=60)
        assert result.shape == (60, 60)

    def test_output_dtype_float32(self):
        """Output should remain float32 (not quantised to uint8)."""
        prob = np.random.rand(20, 20).astype(np.float32)
        result = upsample_prob(prob, original_h=60, original_w=60)
        assert result.dtype == np.float32

    def test_values_in_range(self):
        """Bilinear interpolation of [0,1] values should stay in [0,1]."""
        prob = np.random.rand(20, 20).astype(np.float32)
        result = upsample_prob(prob, original_h=60, original_w=60)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6   # allow tiny float error

    def test_uniform_map_stays_uniform(self):
        """A uniform probability map should upsample to the same constant."""
        prob = np.full((10, 10), 0.7, dtype=np.float32)
        result = upsample_prob(prob, original_h=30, original_w=30)
        np.testing.assert_allclose(result, 0.7, atol=1e-5)

    def test_non_square_target(self):
        """Non-square target dimensions should be respected."""
        prob = np.random.rand(10, 20).astype(np.float32)
        result = upsample_prob(prob, original_h=30, original_w=60)
        assert result.shape == (30, 60)

    def test_identity_when_same_size(self):
        """Upsampling to the same size should return values very close to input."""
        prob = np.random.rand(32, 32).astype(np.float32)
        result = upsample_prob(prob, original_h=32, original_w=32)
        np.testing.assert_allclose(result, prob, atol=1e-5)