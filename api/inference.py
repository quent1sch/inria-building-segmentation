"""
api/inference.py

Inference wrapper with embedded formatting pipeline, resolution-aware preprocessing 
and sliding-window support for large images.

The model was trained on 512x512 patches; for larger inputs (e.g. a full
5000x5000 Inria tile) we tile the image, run inference on each tile, and
stitch the predictions back with overlap-based averaging to avoid
boundary artefacts.

Memory management for large inputs
------------------------------------
Large GeoTIFF tiles (e.g., 10kx10k) require careful memory handling. 
Key design decisions:
 
  - Large TIFFs are written to a temp file instead of io.BytesIO so the
    OS can page them rather than holding everything in RAM simultaneously.
  - uint16 -> uint8 conversion uses right-shift (>> 8) instead of a float32
    intermediate, saving ~1.2 GB peak RAM for a 10kx10k 3-band image.
  - Padded image and intermediate arrays are explicitly deleted after use.
  - TTA is automatically skipped for images above TTA_MAX_PIXELS to avoid
    doubling inference memory on large tiles.

Resolution handling
-------------------
The model was trained on the Inria dataset at 0.3m/pixel. For best results,
input images should match this resolution. This module handles three cases:
 
  1. Image is finer than 0.3m/px (e.g. 0.1m/px):
       Resampled DOWN to 0.3m/px before inference (default).
       The output mask is upsampled back to original pixel dimensions.
       Can be skipped by passing resample=False.
 
  2. Image is coarser than 0.3m/px (e.g. 1.0m/px):
       Never resampled - predictions may be suboptimal at coarser resolution.
       A warning is returned to the caller via ResolutionInfo.
 
  3. Resolution unknown:
       No resampling. Proceeds as-is.
 
Resolution can be supplied in two ways:
  - Automatically: read from GeoTIFF file metadata (rasterio).
  - Manually: passed as the `input_resolution` argument to predict().
 
Image format handling
---------------------
  - 16-bit images (uint16): normalised to uint8 before inference.
  - Multi-band images (>3 bands, e.g. RGBN or RGBA): first 3 bands kept.
  - Single-band (grayscale): replicated to 3 channels.
  - Plain images (JPEG, PNG, BMP, TIFF): read via Pillow.
  - GeoTIFF: read via rasterio (preserves 16-bit, reads metadata).

Spatial metadata passthrough
-----------------------------
GeoTIFF files carry a CRS (coordinate reference system) and an affine
transform (pixel → world coordinate mapping). These are captured in a
SpatialRef dataclass and returned alongside the image array. They are
NOT consumed by the inference pipeline — the model only sees uint8 RGB
pixels. SpatialRef is passed through to the output layer (api/output.py)
so results can be written as georeferenced GeoTIFFs with the same CRS
and transform as the input, preserving spatial alignment for GIS use.

Plain images (JPEG/PNG) return SpatialRef=None. The output layer falls
back to plain PNG in that case unless the user explicitly requests TIFF.
"""

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from models.unet import UNetBuilding

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
 
# Formats Pillow can reliably open
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
 
TRAINING_RESOLUTION = 0.3 # metres per pixel

# Images above this pixel count skip TTA to avoid doubling memory.
# 4096×4096 =~ 16M pixels. Adjust in config if needed.
TTA_MAX_PIXELS = 4096 * 4096
 
 
# ── result dataclass ──────────────────────────────────────────────────────────
 
@dataclass
class ResolutionInfo:
    """
    Carries resolution metadata back to the caller (API response headers).
 
    Fields
    ------
    input_resolution  : detected or user-supplied resolution (m/px), or None
    resampled         : True if the image was resampled before inference
    resampled_to      : target resolution used for resampling, or None
    warning           : human-readable warning string, or None
    """
    input_resolution: Optional[float] = None
    resampled: bool = False
    resampled_to: Optional[float] = None
    warning: Optional[str] = None


# ── spatial reference dataclass ──────────────────────────────────────────────

@dataclass
class SpatialRef:
    """
    CRS and affine transform extracted from a GeoTIFF input.

    Passed through the pipeline unchanged — the model never sees it.
    Consumed by api/output.py to write georeferenced GeoTIFF results.

    Fields
    ------
    crs       : rasterio.crs.CRS — coordinate reference system (e.g. EPSG:2056)
    transform : affine.Affine    — pixel (col, row) → world (x, y) mapping
    width     : int              — original image width in pixels
    height    : int              — original image height in pixels

    Note: width/height are the ORIGINAL input dimensions. After resampling
    the image passed to the model is smaller, but output.py needs the
    original dimensions to write the result at the correct scale with the
    correct transform.
    """
    crs:       object   # rasterio.crs.CRS
    transform: object   # affine.Affine
    width:     int
    height:    int
 
# ── image reading ─────────────────────────────────────────────────────────────
 
def read_image_bytes(
    data: bytes,
    filename: str = "",
) -> Tuple[np.ndarray, Optional[float], Optional["SpatialRef"]]:
    """
    Read raw file bytes -> (HWC uint8 RGB numpy array, resolution_m_per_px or None, 
    SpatialRef or None).

    SpatialRef is populated for GeoTIFF files with a projected CRS.
    It is None for plain images (JPEG/PNG) or TIFFs without CRS metadata.
 
    Handles:
      - 16-bit -> 8-bit normalisation
      - Multi-band (>3) -> first 3 bands kept
      - Single-band -> replicated to 3 channels

    Large TIFF handling
    -------------------
    For TIFFs, writes bytes to a temp file before opening with rasterio.
    io.BytesIO pins the entire compressed file in RAM while rasterio also
    holds the decompressed array — for a 10k×10k 16-bit tile that can
    exceed 2 GB simultaneously. A temp file lets the OS page the compressed
    data out, reducing peak RAM significantly.
    The temp file is always deleted in the finally block.
    """
    suffix = Path(filename).suffix.lower()
 
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{suffix}'. "
            f"Accepted: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
 
    auto_resolution: Optional[float] = None
 
    # ── GeoTIFF: use rasterio for metadata + 16-bit support ──────────────
    if suffix in (".tif", ".tiff") and HAS_RASTERIO:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        del data # release compressed bytes before rasterio reads
 
        try:
            with rasterio.open(tmp_path) as src:
                arr = src.read() # (C, H, W)
                try:
                    if src.crs and src.crs.is_projected:
                        res_x, res_y = src.res
                        auto_resolution = float((res_x + res_y) / 2)
                        # Capture spatial metadata for georeferenced output.
                        # width/height are original dims — needed by output.py
                        # even if the image is later resampled for inference.
                        spatial_ref = SpatialRef(
                            crs=src.crs,
                            transform=src.transform,
                            width=src.width,
                            height=src.height,
                        )
                except Exception:
                    pass
        finally:
            os.unlink(tmp_path)
 
        image = _bands_to_rgb_uint8(arr)
        del arr
 
    else:
        pil_img = Image.open(io.BytesIO(data))
        arr = np.array(pil_img)
        del data
 
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)
 
        image = _bands_to_rgb_uint8(arr)
        del arr
 
    return image, auto_resolution, spatial_ref
 
 
def _bands_to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """
    (C, H, W) array of any dtype -> (H, W, 3) uint8 RGB.
 
    uint16 → uint8 via right-shift (>> 8) instead of float32 cast.
    For a 10kx10k 3-band image this avoids a ~1.2 GB float32 intermediate.
    >> 8 divides by 256 (vs / 65535 * 255), error is =< 0.4% - negligible
    for inference.
    """
    C = arr.shape[0]
 
    # Band selection
    if C == 1:
        arr = np.concatenate([arr, arr, arr], axis=0) # grayscale -> RGB
    elif C > 3:
        arr = arr[:3] # keep R, G, B
 
    # dtype normalisation
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype != np.uint8:
        # generic: clip and scale to 0-255
        arr = arr.astype(np.float32)
        for c in range(arr.shape[0]):
            mn, mx = arr[c].min(), arr[c].max()
            if mx > mn:
                arr[c] = (arr[c] - mn) / (mx - mn) * 255.0
        arr = arr.astype(np.uint8)
 
    # CHW → HWC
    return arr.transpose(1, 2, 0)
 
 
# ── resampling ────────────────────────────────────────────────────────────────
 
def resample_image(
    image: np.ndarray,
    input_resolution: float,
    target_resolution: float,
) -> np.ndarray:
    """
    Resample HWC uint8 image from input_resolution to target_resolution.
 
    scale = input_resolution / target_resolution
      > 1 -> image is finer, we scale DOWN in pixels (fewer pixels needed)
      < 1 -> image is coarser (caller should not call this for coarser images)
    """
    H, W = image.shape[:2]
    scale = input_resolution / target_resolution
    new_H = max(1, int(round(H * scale)))
    new_W = max(1, int(round(W * scale)))
 
    # Use INTER_AREA for downscaling (best for aerial), INTER_CUBIC for up
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_W, new_H), interpolation=interp)
 
 
def upsample_prob(
    prob: np.ndarray,
    original_h: int,
    original_w: int,
) -> np.ndarray:
    """
    Upsample a float32 probability map back to (original_h, original_w).
    Uses bilinear interpolation to preserve smooth probability gradients
    near boundaries — thresholding after this produces sharper edges than
    nearest-neighbour upsampling of a binary mask.
    """
    return cv2.resize(
        prob,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR,
    )
    
 
 
# ── main inference class ──────────────────────────────────────────────────────
 
class SegmentationInference:
    """
    Loads a trained model and exposes a simple predict() interface.

    Parameters
    ----------
    checkpoint_path : str | Path
    config_path     : str | Path
    device          : str   "cpu" | "cuda" | "auto"
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config_path: Union[str, Path] = "configs/config.yaml",
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.tile_size = self.cfg["inference"]["tile_size"]
        self.overlap = self.cfg["inference"]["overlap"]
        self.threshold = self.cfg["inference"]["threshold"]
        self.use_tta = self.cfg["inference"].get("tta", False)
        self.target_resolution = self.cfg["inference"].get("target_resolution", TRAINING_RESOLUTION)
        self.mean = np.array(self.cfg["data"]["mean"], dtype=np.float32)
        self.std = np.array(self.cfg["data"]["std"], dtype=np.float32)

        self.model = UNetBuilding.load(str(checkpoint_path), device=device)
        print(f"Model loaded on {device}.")

    # ── normalisation ─────────────────────────────────────────────────────

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """HWC uint8 -> HWC float32 normalised with ImageNet stats."""
        img = image.astype(np.float32) / 255.0
        return (img - self.mean) / self.std

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """HWC float32 -> 1CHW torch.Tensor."""
        return torch.from_numpy(
            self._normalize(image).transpose(2, 0, 1)
            ).unsqueeze(0).to(self.device)

    # ── single-tile inference ─────────────────────────────────────────────

    @torch.no_grad()
    def _predict_tile(self, tile: np.ndarray) -> np.ndarray:
        """Predict probability map for a single HWC tile."""
        logit = self.model(self._to_tensor(tile))
        return torch.sigmoid(logit).squeeze().cpu().numpy() # (H, W) float32 in [0, 1]

    # ── sliding-window inference ──────────────────────────────────────────

    def _sliding_window_predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference over a large image using a sliding window.

        Returns a probability map (H, W) with the same spatial resolution
        as the input.  Overlapping tiles are averaged.
        """
        H, W, _ = image.shape
        size = self.tile_size
        step = size - self.overlap

        # Pad image so every position is covered
        pad_h = max(0, size - H % step if H % step != 0 else 0)
        pad_w = max(0, size - W % step if W % step != 0 else 0)
        if pad_h > 0 or pad_w > 0:
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="reflect",
            )

        pH, pW = image.shape[:2]
        prob_acc = np.zeros((pH, pW), dtype=np.float32)
        count_acc = np.zeros((pH, pW), dtype=np.uint16)

        ys = list(range(0, pH - size + 1, step))
        xs = list(range(0, pW - size + 1, step))

        # Ensure the last strip is covered
        if ys[-1] + size < pH:
            ys.append(pH - size)
        if xs[-1] + size < pW:
            xs.append(pW - size)

        for y in ys:
            for x in xs:
                prob_acc[y:y+size, x:x+size]  += self._predict_tile(image[y:y+size, x:x+size])
                count_acc[y:y+size, x:x+size] += 1
 
        del image   # free padded copy before division
 
        np.divide(prob_acc, count_acc, out=prob_acc, where=count_acc > 0)
        del count_acc
 
        return prob_acc[:H, :W]
    

    # ── core predict (pixel-level) ────────────────────────────────────────
 
    def _predict_array(self, image: np.ndarray) -> np.ndarray:
        """
        Run sliding-window inference on a preprocessed HWC uint8 RGB array.
        Returns a float32 probability map (H, W) in [0, 1].
        Thresholding is intentionally deferred to predict() so it happens
        AFTER any upsampling - applying threshold before upsampling would
        produce jagged/rounded edges on building contours.

        TTA is skipped automatically for images > TTA_MAX_PIXELS to avoid
        doubling memory on large tiles.
        """
        H, W = image.shape[:2]
 
        if H <= self.tile_size and W <= self.tile_size:
            pad_h = max(0, self.tile_size - H)
            pad_w = max(0, self.tile_size - W)
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            prob = self._predict_tile(padded)[:H, :W]
        else:
            prob = self._sliding_window_predict(image)

        use_tta = self.use_tta and (H * W) <= TTA_MAX_PIXELS
        if use_tta:
            # Average with TTA predictions on the same (possibly padded) input
            pad_h = max(0, self.tile_size - H)
            pad_w = max(0, self.tile_size - W)
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            tensor = self._to_tensor(padded)
            del padded # free padded copy before TTA
            tta_prob = self.model.predict_tta(tensor).float().squeeze().cpu().numpy()[:H, :W]

            # Combine: if either standard or TTA prob agrees, trust the average
            prob = (prob + tta_prob) / 2.0
            del tta_prob
 
        return prob # float32, NOT thresholded
 
    # ── resolution-aware preprocessing ───────────────────────────────────
 
    def _preprocess(
        self,
        image: np.ndarray,
        input_resolution: Optional[float],
        resample: bool,
    ) -> Tuple[np.ndarray, ResolutionInfo]:
        """
        Apply resolution-aware preprocessing and return:
          - the image to run inference on (may be resampled)
          - a ResolutionInfo describing what was done
        """
        info = ResolutionInfo(input_resolution=input_resolution)
 
        if input_resolution is None:
            return image, info
 
        tgt = self.target_resolution
 
        if abs(input_resolution - tgt) < 1e-4:
            # Already at training resolution — nothing to do
            return image, info
 
        if input_resolution > tgt:
            # Coarser than training — warn, never resample
            info.warning = (
                f"Input resolution {input_resolution:.4f}m/px is coarser than the "
                f"training resolution {tgt}m/px. "
                f"Predictions may be suboptimal."
            )
            return image, info
 
        # Finer than training — resample unless user opted out
        if not resample:
            info.warning = (
                f"Input resolution {input_resolution:.4f}m/px is finer than the "
                f"training resolution {tgt}m/px. "
                f"Resampling was skipped (resample=false). "
                f"Predictions may be suboptimal."
            )
            return image, info
 
        # Resample down to target resolution
        original_h, original_w = image.shape[:2]
        resampled_image = resample_image(image, input_resolution, tgt)
        del image # free original copy before inference
        info.resampled = True
        info.resampled_to = tgt
        return resampled_image, info, original_h, original_w
 
    # ── public API ────────────────────────────────────────────────────────
 
    def predict(
        self,
        image: np.ndarray,
        input_resolution: Optional[float] = None,
        resample: bool = True,
    ) -> Tuple[np.ndarray, ResolutionInfo]:
        """
        Run building segmentation on an RGB image.
 
        Parameters
        ----------
        image            : HWC uint8 numpy array
        input_resolution : metres per pixel, or None if unknown
        resample         : if True (default) and image is finer than training
                           resolution, resample before inference.
                           Has no effect if image is coarser or resolution unknown.
 
        Returns
        -------
        mask : (H, W) bool array aligned to the ORIGINAL input pixel grid
        info : ResolutionInfo with resampling details and any warnings
        """
        original_h, original_w = image.shape[:2]
        result = self._preprocess(image, input_resolution, resample)
 
        # _preprocess returns 4 values when resampling occurred, 2 otherwise
        if len(result) == 4:
            proc_image, info, orig_h, orig_w = result
        else:
            proc_image, info = result
            orig_h, orig_w = original_h, original_w
 
        prob = self._predict_array(proc_image) # float32 probability map
        del proc_image # free preprocessed copy before upsampling
 
        # Upsample probability map BEFORE thresholding (if resampling).
        # This preserves smooth boundary gradients so the final threshold
        # produces clean sharp edges rather than blocky/rounded contours.
        if info.resampled:
            prob = upsample_prob(prob, orig_h, orig_w)

        mask = prob > self.threshold # threshold last, at full resolution
        del prob # free probability map before returning
 
        return mask, info
 

