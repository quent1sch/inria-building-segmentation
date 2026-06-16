"""
data/dataset.py

PyTorch Dataset for the patched Inria dataset.

Folder layout expected (created by scripts/patch_dataset.py):

    patches_dir/
    ├── austin/
    │   ├── images/   *.png
    │   └── masks/    *.png
    ├── chicago/
    ├── kitsap/
    ├── tyrol-w/
    └── vienna/
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class InriaDataset(Dataset):
    """
    Binary building segmentation dataset built from pre-extracted patches.

    Parameters
    ----------
    patches_dir : str | Path
        Root directory containing one sub-folder per city.
    cities : list[str]
        Which cities to include.  Pass a subset for train/val splits.
    transform : callable, optional
        Albumentations transform applied to both image and mask together.
        Expected signature: transform(image=np.ndarray, mask=np.ndarray)
        Returns a dict with keys "image" (torch.Tensor CHW) and "mask"
        (torch.Tensor HW float32).
    """

    def __init__(
        self,
        patches_dir: str | Path,
        cities: List[str],
        transform: Optional[Callable] = None,
    ):
        self.patches_dir = Path(patches_dir)
        self.transform = transform

        self.image_paths: List[Path] = []
        self.mask_paths: List[Path] = []

        for city in cities:
            city_img_dir = self.patches_dir / city / "images"
            city_msk_dir = self.patches_dir / city / "masks"

            if not city_img_dir.exists():
                raise FileNotFoundError(
                    f"City image directory not found: {city_img_dir}\n"
                    "Run scripts/patch_dataset.py first."
                )

            img_files = sorted(city_img_dir.glob("*.png"))
            for img_path in img_files:
                mask_path = city_msk_dir / img_path.name
                if mask_path.exists():
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No image/mask pairs found for cities {cities} in {patches_dir}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))

        # Binarize mask: 255 → 1
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # torch.Tensor (C, H, W) float32
            mask = augmented["mask"]     # torch.Tensor (H, W)    float32

        return image, mask

    # ── convenience ──────────────────────────────────────────────────────

    @staticmethod
    def available_cities(patches_dir: str | Path) -> List[str]:
        """Return the list of city names found under patches_dir."""
        p = Path(patches_dir)
        return sorted([d.name for d in p.iterdir() if d.is_dir()])
