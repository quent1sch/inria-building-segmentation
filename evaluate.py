"""
evaluate.py

Post-training evaluation on the validation set.
- Per-city metrics (IoU - Intersection over Union, Dice, Precision, Recall)
- Qualitative grid: image | ground-truth | prediction overlay
- Saves results to outputs/evaluation/

Usage
-----
python evaluate.py --checkpoint checkpoints/best_model.pth
python evaluate.py --checkpoint checkpoints/best_model.pth --config configs/config.yaml
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import InriaDataset
from data.transforms import get_val_transforms
from models.unet import UNetBuilding


# ── metrics (same as train.py — kept here to avoid coupling) ─────────────────

def compute_metrics_np(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Binary arrays (0/1)."""
    tp = (pred_mask & gt_mask).sum()
    fp = (pred_mask & ~gt_mask).sum()
    fn = (~pred_mask & gt_mask).sum()

    smooth = 1e-6
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


# ── visualisation helpers ─────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor_chw: torch.Tensor) -> np.ndarray:
    """CHW float tensor → HWC uint8 numpy for display."""
    img = tensor_chw.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def overlay_mask(image: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Returns an RGB overlay:
      Green  = true positive
      Red    = false positive
      Blue   = false negative
    """
    overlay = image.copy()
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    overlay[tp] = [0, 200, 0]
    overlay[fp] = [200, 0, 0]
    overlay[fn] = [0, 0, 200]

    # Blend with original image
    result = (0.5 * image + 0.5 * overlay).astype(np.uint8)
    return result


def save_qualitative_grid(samples: list, out_path: Path, n_cols: int = 4):
    """
    samples: list of (image_np, gt_mask_np, pred_mask_np) tuples
    """
    n = min(len(samples), n_cols * 4)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=(n_cols * 3, n_rows * 9))
    axes = np.array(axes).reshape(n_rows * 3, n_cols)

    for i, (img, gt, pred) in enumerate(samples[:n]):
        row_base = (i // n_cols) * 3
        col = i % n_cols

        axes[row_base, col].imshow(img)
        axes[row_base, col].set_title("Image", fontsize=7)

        axes[row_base + 1, col].imshow(gt, cmap="gray")
        axes[row_base + 1, col].set_title("Ground Truth", fontsize=7)

        axes[row_base + 2, col].imshow(overlay_mask(img, pred, gt))
        axes[row_base + 2, col].set_title("Pred Overlay", fontsize=7)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved qualitative grid → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default="outputs/evaluation")
    parser.add_argument("--n-vis", type=int, default=16,
                        help="Number of samples to visualise")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── model ────────────────────────────────────────────────────────────
    model = UNetBuilding.load(args.checkpoint, device=device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── evaluate per city ────────────────────────────────────────────────
    val_cities = cfg["data"]["val_cities"]
    mean = cfg["data"]["mean"]
    std = cfg["data"]["std"]
    size = cfg["data"]["image_size"]
    threshold = cfg["inference"]["threshold"]

    city_results = {}
    all_samples = []

    for city in val_cities:
        dataset = InriaDataset(
            cfg["data"]["patches_dir"],
            cities=[city],
            transform=get_val_transforms(image_size=size, mean=mean, std=std),
        )
        loader = DataLoader(dataset, 
                            batch_size=8, 
                            shuffle=False, 
                            num_workers=2)

        city_metrics = {"iou": [], "dice": [], "precision": [], "recall": []}

        with torch.no_grad():
            for images, masks in tqdm(loader, desc=f"  {city}"):
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                masks_np = masks.cpu().numpy().astype(bool)

                for i in range(len(images)):
                    pred = probs[i] > threshold
                    gt = masks_np[i]
                    m = compute_metrics_np(pred, gt)
                    for k in city_metrics:
                        city_metrics[k].append(m[k])

                    # Collect samples for visualisation
                    if len(all_samples) < args.n_vis:
                        img_np = denormalize(images[i].cpu())
                        all_samples.append((img_np, gt, pred))

        city_results[city] = {k: np.mean(v) for k, v in city_metrics.items()}

    # ── print results table ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'City':<14} {'IoU':>8} {'Dice':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 60)

    ious, dices = [], []
    for city, m in city_results.items():
        print(
            f"{city:<14} {m['iou']:>8.4f} {m['dice']:>8.4f} "
            f"{m['precision']:>10.4f} {m['recall']:>8.4f}"
        )
        ious.append(m["iou"])
        dices.append(m["dice"])

    print("-" * 60)
    print(
        f"{'Mean':<14} {np.mean(ious):>8.4f} {np.mean(dices):>8.4f}"
    )
    print("=" * 60)

    # ── save results ────────────────────────────────────────────────────
    import json
    results_path = out_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(city_results, f, indent=2)
    print(f"\nMetrics saved → {results_path}")

    # ── qualitative grid ─────────────────────────────────────────────────
    save_qualitative_grid(all_samples, out_dir / "predictions_grid.png")


if __name__ == "__main__":
    main()
