"""
train.py

Training loop for the Inria building segmentation model.

Fine-tuning strategy
--------------------
Phase 1 — Warmup (epochs 1 … warmup_epochs)
  The encoder is frozen. Only the decoder and segmentation head are updated
  at decoder_lr. This lets the randomly-initialised decoder stabilise before
  the encoder starts moving, preventing the pretrained ImageNet features from
  being destroyed by large early gradients (catastrophic forgetting).

Phase 2 — Full fine-tuning (epochs warmup_epochs+1 … total_epochs)
  The encoder is unfrozen. Differential learning rates are applied:
    encoder params → encoder_lr  (e.g. 1e-5, 10× lower)
    decoder + head → decoder_lr  (e.g. 1e-4)
  The encoder adapts slowly to aerial-imagery statistics while the decoder
  continues learning the segmentation-specific mapping.
  A CosineAnnealingLR scheduler runs over phase 2 only.

Test-Time Augmentation (TTA)
-----------------------------
Optionally enabled during validation (tta_val: true in config).
Always recommended for the final evaluate.py run (tta_eval: true).
8 geometric variants (4 rotations × 2 flips) are averaged.

Usage
-----
python train.py                          # uses configs/config.yaml
python train.py --config my_config.yaml
"""

import argparse
import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import InriaDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.unet import UNetBuilding


# ── loss ─────────────────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_w = dice_weight
        self.bce_w = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_4d = targets.unsqueeze(1)   # (B, H, W) → (B, 1, H, W)

        bce_loss = self.bce(logits, targets_4d)

        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets_4d).sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + smooth) / (
            probs.sum(dim=(2, 3)) + targets_4d.sum(dim=(2, 3)) + smooth
        )
        return self.bce_w * bce_loss + self.dice_w * dice_loss.mean()


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    preds = (torch.sigmoid(logits) > threshold).float().squeeze(1)  # (B, H, W)
    targets = targets.float()

    tp = (preds * targets).sum(dim=(1, 2))
    fp = (preds * (1 - targets)).sum(dim=(1, 2))
    fn = ((1 - preds) * targets).sum(dim=(1, 2))

    smooth = 1e-6
    iou       = (tp + smooth) / (tp + fp + fn + smooth)
    dice      = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)

    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


# ── training / validation steps ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    metrics_sum = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

    for images, masks in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            m = compute_metrics(logits.detach(), masks)
            for k in metrics_sum:
                metrics_sum[k] += m[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics_sum.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, use_tta: bool = False):
    model.eval()
    total_loss = 0.0
    metrics_sum = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

    for images, masks in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        # Loss always uses a standard forward pass (TTA returns bool masks,
        # not logits, so we keep loss computation separate)
        logits = model(images)
        loss   = criterion(logits, masks)
        total_loss += loss.item()

        if use_tta:
            # TTA: binary predictions averaged over 8 augmentations
            preds = model.predict_tta(images).float().squeeze(1)   # (B, H, W)
            targets = masks.float()
            tp = (preds * targets).sum(dim=(1, 2))
            fp = (preds * (1 - targets)).sum(dim=(1, 2))
            fn = ((1 - preds) * targets).sum(dim=(1, 2))
            smooth = 1e-6
            m = {
                "iou":       ((tp + smooth) / (tp + fp + fn + smooth)).mean().item(),
                "dice":      ((2*tp + smooth) / (2*tp + fp + fn + smooth)).mean().item(),
                "precision": ((tp + smooth) / (tp + fp + smooth)).mean().item(),
                "recall":    ((tp + smooth) / (tp + fn + smooth)).mean().item(),
            }
        else:
            m = compute_metrics(logits, masks)

        for k in metrics_sum:
            metrics_sum[k] += m[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics_sum.items()}


# ── lr helpers ────────────────────────────────────────────────────────────────

def get_lrs(optimizer) -> dict:
    """Return current lr for each named param group."""
    return {
        pg.get("name", f"group_{i}"): pg["lr"]
        for i, pg in enumerate(optimizer.param_groups)
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── device ───────────────────────────────────────────────────────────
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    print(f"Device: {device}  |  AMP: {use_amp}")

    # ── data ─────────────────────────────────────────────────────────────
    all_cities   = InriaDataset.available_cities(cfg["data"]["patches_dir"])
    val_cities   = cfg["data"]["val_cities"]
    train_cities = [c for c in all_cities if c not in val_cities]

    print(f"Train cities : {train_cities}")
    print(f"Val   cities : {val_cities}")

    mean, std, size = cfg["data"]["mean"], cfg["data"]["std"], cfg["data"]["image_size"]

    train_ds = InriaDataset(
        patches_dir=cfg["data"]["patches_dir"], 
        cities=train_cities,
        transform=get_train_transforms(image_size=size, mean=mean, std=std),
    )
    val_ds = InriaDataset(
        patches_dir=cfg["data"]["patches_dir"], 
        cities=val_cities,
        transform=get_val_transforms(image_size=size, mean=mean, std=std),
    )
    print(f"Train patches: {len(train_ds)}  |  Val patches: {len(val_ds)}")

    train_loader = DataLoader(
        dataset=train_ds, 
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,  
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        dataset=val_ds, 
        batch_size=cfg["training"]["batch_size"],
        shuffle=False, 
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device == "cuda"),
    )

    # ── model ────────────────────────────────────────────────────────────
    model = UNetBuilding(
        encoder_name=cfg["model"]["encoder"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=cfg["model"]["in_channels"],
    ).to(device)

    # ── fine-tuning config ────────────────────────────────────────────────
    t_cfg        = cfg["training"]
    warmup_epochs = t_cfg["warmup_epochs"]
    total_epochs  = t_cfg["epochs"]
    encoder_lr    = t_cfg["encoder_lr"]
    decoder_lr    = t_cfg["decoder_lr"]
    use_tta_val   = t_cfg.get("tta_val", False)

    # ── phase 1 setup: freeze encoder, decoder-only optimizer ────────────
    print(f"\nPhase 1 — warmup ({warmup_epochs} epochs): encoder frozen, decoder lr={decoder_lr}")
    model.freeze_encoder()

    # During warmup the encoder is frozen so we only pass decoder+head params.
    # We still use make_optimizer() for consistency but encoder_lr is unused
    # (frozen params don't receive gradients regardless of lr).
    optimizer = model.make_optimizer(
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        weight_decay=t_cfg["weight_decay"],
    )

    # Phase 2 scheduler — cosine over phase 2 epochs only
    phase2_epochs = total_epochs - warmup_epochs
    if t_cfg["scheduler"] == "cosine" and phase2_epochs > 0:
        # Will be created at the start of phase 2
        scheduler = None
    else:
        scheduler = None

    criterion = DiceBCELoss(
        dice_weight=t_cfg["dice_weight"],
        bce_weight=t_cfg["bce_weight"],
    )

    # ── MLflow ───────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    checkpoint_dir = Path(t_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_iou        = 0.0
    patience_counter = 0
    patience         = t_cfg["early_stopping_patience"]

    with mlflow.start_run():
        mlflow.log_params({
            "encoder":        cfg["model"]["encoder"],
            "encoder_weights": cfg["model"]["encoder_weights"],
            "epochs":         total_epochs,
            "warmup_epochs":  warmup_epochs,
            "encoder_lr":     encoder_lr,
            "decoder_lr":     decoder_lr,
            "weight_decay":   t_cfg["weight_decay"],
            "batch_size":     t_cfg["batch_size"],
            "dice_weight":    t_cfg["dice_weight"],
            "bce_weight":     t_cfg["bce_weight"],
            "image_size":     size,
            "tta_val":        use_tta_val,
            "train_cities":   str(train_cities),
            "val_cities":     str(val_cities),
            "device":         device,
        })

        for epoch in range(1, total_epochs + 1):

            # ── phase transition ──────────────────────────────────────────
            if epoch == warmup_epochs + 1:
                print(
                    f"\nPhase 2 — fine-tuning ({phase2_epochs} epochs): "
                    f"encoder unfrozen  encoder_lr={encoder_lr}  decoder_lr={decoder_lr}"
                )
                model.unfreeze_encoder()
                # Rebuild optimizer — now all three param groups are active
                optimizer = model.make_optimizer(
                    encoder_lr=encoder_lr,
                    decoder_lr=decoder_lr,
                    weight_decay=t_cfg["weight_decay"],
                )
                if t_cfg["scheduler"] == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=phase2_epochs, eta_min=1e-6
                    )

            # ── train & validate ──────────────────────────────────────────
            t0 = time.time()
            train_loss, train_m = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            val_loss, val_m = validate(
                model, val_loader, criterion, device, use_tta=use_tta_val
            )

            if scheduler is not None:
                scheduler.step()

            # ── logging ───────────────────────────────────────────────────
            lrs      = get_lrs(optimizer)
            elapsed  = time.time() - t0
            phase    = 1 if epoch <= warmup_epochs else 2

            print(
                f"Epoch {epoch:03d}/{total_epochs}  [phase {phase}]  [{elapsed:.0f}s]  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_iou={val_m['iou']:.4f}  val_dice={val_m['dice']:.4f}"
            )

            metrics_to_log = {
                "train_loss":    train_loss,
                "val_loss":      val_loss,
                "train_iou":     train_m["iou"],
                "val_iou":       val_m["iou"],
                "train_dice":    train_m["dice"],
                "val_dice":      val_m["dice"],
                "val_precision": val_m["precision"],
                "val_recall":    val_m["recall"],
                "phase":         float(phase),
            }
            # Log each param group lr separately so MLflow shows the split
            for name, lr in lrs.items():
                metrics_to_log[f"lr_{name}"] = lr

            mlflow.log_metrics(metrics_to_log, step=epoch)

            # ── checkpoint ───────────────────────────────────────────────
            if val_m["iou"] > best_iou:
                best_iou = val_m["iou"]
                patience_counter = 0
                ckpt_path = checkpoint_dir / t_cfg["best_model_name"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": best_iou,
                    "model_config": {
                        "encoder": cfg["model"]["encoder"],
                        "in_channels": cfg["model"]["in_channels"],
                    },
                }, ckpt_path)
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
                print(f"New best IoU={best_iou:.4f} — checkpoint saved.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping (no improvement for {patience} epochs).")
                    break

        mlflow.log_metrics({"best_val_iou": best_iou})
        print(f"\nTraining complete.  Best val IoU: {best_iou:.4f}")
        print(f"Checkpoint : {checkpoint_dir / t_cfg['best_model_name']}")
        print(f"MLflow UI  : mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000")


if __name__ == "__main__":
    main()
