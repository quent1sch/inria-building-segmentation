"""
api/inference.py

Inference wrapper with sliding-window support for large images.

The model was trained on 512×512 patches; for larger inputs (e.g. a full
5000×5000 Inria tile) we tile the image, run inference on each tile, and
stitch the predictions back with overlap-based averaging to avoid
boundary artefacts.
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml
from PIL import Image

from models.unet import UNetBuilding


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
        self.mean = np.array(self.cfg["data"]["mean"], dtype=np.float32)
        self.std = np.array(self.cfg["data"]["std"], dtype=np.float32)

        self.model = UNetBuilding.load(str(checkpoint_path), device=device)
        print(f"Model loaded on {device}.")

    # ── normalisation ─────────────────────────────────────────────────────

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """HWC uint8 -> HWC float32 normalised with ImageNet stats."""
        img = image.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """HWC float32 -> 1CHW torch.Tensor."""
        img = self._normalize(image)
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    # ── single-tile inference ─────────────────────────────────────────────

    @torch.no_grad()
    def _predict_tile(self, tile: np.ndarray) -> np.ndarray:
        """Predict probability map for a single HWC tile."""
        t = self._to_tensor(tile)
        logit = self.model(t)                   # (1, 1, H, W)
        prob = torch.sigmoid(logit).squeeze().cpu().numpy()
        return prob                              # (H, W) float32 in [0, 1]

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

        # prob_acc = np.zeros((H, W), dtype=np.float32)
        # count_acc = np.zeros((H, W), dtype=np.float32)

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
        count_acc = np.zeros((pH, pW), dtype=np.float32)

        ys = list(range(0, pH - size + 1, step))
        xs = list(range(0, pW - size + 1, step))

        # Ensure the last strip is covered
        if ys[-1] + size < pH:
            ys.append(pH - size)
        if xs[-1] + size < pW:
            xs.append(pW - size)

        for y in ys:
            for x in xs:
                tile = image[y:y + size, x:x + size]
                prob = self._predict_tile(tile)

                prob_acc[y:y + size, x:x + size] += prob
                count_acc[y:y + size, x:x + size] += 1.0

        # Crop back to original size
        with np.errstate(divide="ignore", invalid="ignore"):
            prob_map = np.where(count_acc > 0, prob_acc / count_acc, 0.0)

        return prob_map[:H, :W]

    # ── public API ────────────────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run building segmentation on an RGB image.

        Parameters
        ----------
        image : np.ndarray  shape (H, W, 3)  dtype uint8

        Returns
        -------
        np.ndarray  shape (H, W)  dtype bool
            True where a building is predicted.
        """
        H, W = image.shape[:2]

        if H <= self.tile_size and W <= self.tile_size:
            pad_h = max(0, self.tile_size - H)
            pad_w = max(0, self.tile_size - W)
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            prob = self._predict_tile(padded)[:H, :W]
        else:
            prob = self._sliding_window_predict(image)

        if self.use_tta:
            # Average with TTA predictions on the same (possibly padded) input
            tensor = self._to_tensor(image if H > self.tile_size or W > self.tile_size
                                     else np.pad(image, ((0, max(0, self.tile_size - H)),
                                                         (0, max(0, self.tile_size - W)),
                                                         (0, 0)), mode="reflect"))
            tta_mask = self.model.predict_tta(tensor, threshold=self.threshold)
            tta_prob = tta_mask.float().squeeze().cpu().numpy()[:H, :W]
            # Combine: if either standard or TTA prob agrees, trust the average
            prob = (prob + tta_prob) / 2.0

        return prob > self.threshold

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        """Like predict() but returns float32 probability map instead of binary mask."""
        H, W = image.shape[:2]
        if H <= self.tile_size and W <= self.tile_size:
            pad_h = max(0, self.tile_size - H)
            pad_w = max(0, self.tile_size - W)
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            return self._predict_tile(padded)[:H, :W]
        return self._sliding_window_predict(image)
