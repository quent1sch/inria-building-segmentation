"""
data/transforms.py

Albumentations-based transform pipelines for train / validation.

Both pipelines end with:
  - Normalize  (ImageNet mean/std — matches pretrained ResNet34 encoder)
  - ToTensorV2 (HWC numpy → CHW torch.Tensor)
"""

from typing import List, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize(mean: List[float], std: List[float]) -> A.Normalize:
    return A.Normalize(mean=mean, std=std, max_pixel_value=255.0)


# ── public API ───────────────────────────────────────────────────────────────

def get_train_transforms(
    image_size: int = 512,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Training pipeline with spatial and colour augmentations.

    Spatial: flips, 90° rotations, small affine distortions
    Colour:  brightness/contrast, hue/saturation shifts, slight blur
    """
    return A.Compose([
        # ── Spatial ───────────────────────────────────────────────────────
        A.RandomCrop(height=image_size, width=image_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,   # constant padding
            p=0.4,
        ),
        # ── Colour ────────────────────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),
        # ── Normalise & tensorise ─────────────────────────────────────────
        _normalize(list(mean), list(std)),
        ToTensorV2(),
    ])


def get_val_transforms(
    image_size: int = 512,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Validation / inference pipeline — no stochastic augmentations.
    """
    return A.Compose([
        A.CenterCrop(height=image_size, width=image_size, p=1.0),
        _normalize(list(mean), list(std)),
        ToTensorV2(),
    ])
