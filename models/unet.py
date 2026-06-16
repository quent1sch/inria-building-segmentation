"""
models/unet.py

Thin wrapper around segmentation-models-pytorch U-Net.

Why SMP?
  - Battle-tested U-Net implementation
  - Plug-and-play pretrained encoders (ResNet, EfficientNet, …)
  - Single dependency; easy to swap encoder later

Architecture
  Input  : (B, 3, H, W)  — ImageNet-normalised RGB patches
  Output : (B, 1, H, W)  — raw logits  (apply sigmoid for probabilities)

Fine-tuning strategy
  Two-phase training with differential learning rates:
    Phase 1 (warmup)    : encoder frozen,   decoder lr = decoder_lr
    Phase 2 (fine-tune) : encoder unfrozen, encoder lr = encoder_lr (≪ decoder_lr)
  This prevents the pretrained encoder weights from being destroyed by large
  gradients coming from a randomly-initialised decoder (catastrophic forgetting).
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class UNetBuilding(nn.Module):
    """
    U-Net with a pretrained ResNet34 encoder for binary building segmentation.

    Parameters
    ----------
    encoder_name : str
        SMP encoder key, e.g. "resnet34", "resnet50", "efficientnet-b3".
    encoder_weights : str | None
        Pretrained weights identifier.  Use "imagenet" (default) unless you
        want to train from scratch (pass None).
    in_channels : int
        Number of input channels (3 for RGB).
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
    ):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,          # binary → single logit map
            activation=None,    # return raw logits; we apply sigmoid outside
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, 3, H, W)

        Returns
        -------
        torch.Tensor  shape (B, 1, H, W)  — raw logits
        """
        return self.model(x)

    # ── encoder freeze / unfreeze ─────────────────────────────────────────

    def freeze_encoder(self) -> None:
        """
        Phase 1: freeze all encoder parameters.
        Only the decoder and segmentation head will be updated.
        """
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """
        Phase 2: unfreeze encoder so it can be fine-tuned end-to-end.
        Pair with a lower learning rate for the encoder param group
        (see make_optimizer()) to avoid overwriting pretrained features.
        """
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def make_optimizer(
        self,
        encoder_lr: float,
        decoder_lr: float,
        weight_decay: float = 1e-4,
    ) -> torch.optim.AdamW:
        """
        Build an AdamW optimizer with differential learning rates:
          - encoder params → encoder_lr  (lower, to preserve pretrained features)
          - decoder + head → decoder_lr  (higher, random init needs bigger steps)

        This is the standard fine-tuning recipe from ULMFiT, now widely used
        in vision. Typical ratio: decoder_lr = 10 × encoder_lr.

        Parameters
        ----------
        encoder_lr : float   e.g. 1e-5
        decoder_lr : float   e.g. 1e-4
        weight_decay : float

        Returns
        -------
        torch.optim.AdamW with two param groups
        """
        return torch.optim.AdamW(
            [
                {
                    "params": self.model.encoder.parameters(),
                    "lr": encoder_lr,
                    "name": "encoder",
                },
                {
                    "params": self.model.decoder.parameters(),
                    "lr": decoder_lr,
                    "name": "decoder",
                },
                {
                    "params": self.model.segmentation_head.parameters(),
                    "lr": decoder_lr,
                    "name": "head",
                },
            ],
            weight_decay=weight_decay,
        )

    # ── inference helpers ─────────────────────────────────────────────────

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Forward pass → sigmoid → threshold → binary mask.
        Returns torch.Tensor shape (B, 1, H, W) of dtype torch.bool.
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return probs > threshold

    def predict_tta(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Test-Time Augmentation (TTA): run inference on 8 geometric variants
        (4 rotations × 2 flips), average the probability maps, then threshold.

        Cost: 8× inference time.  Typical gain: +0.5–1.5 IoU points for free.

        Returns torch.Tensor shape (B, 1, H, W) of dtype torch.bool.
        """
        with torch.no_grad():
            prob_sum = torch.zeros_like(x[:, :1])  # (B, 1, H, W)

            for k in range(4):                      # 0°, 90°, 180°, 270°
                x_rot = torch.rot90(x, k=k, dims=[2, 3])
                # Original orientation
                prob = torch.sigmoid(self.forward(x_rot))
                prob_sum += torch.rot90(prob, k=-k, dims=[2, 3])
                # Horizontally flipped
                x_flip = torch.flip(x_rot, dims=[3])
                prob_flip = torch.sigmoid(self.forward(x_flip))
                prob_sum += torch.rot90(torch.flip(prob_flip, dims=[3]), k=-k, dims=[2, 3])

            return (prob_sum / 8.0) > threshold

    # ── alternative constructor ───────────────────────────────────────────

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "UNetBuilding":
        """
        Load a trained model from a checkpoint saved by train.py.

        Uses @classmethod (not @staticmethod) so subclasses return the
        correct type — the standard Python pattern for alternative constructors
        (same idea as dict.fromkeys() or pd.DataFrame.from_dict()).

        encoder_weights=None avoids downloading ImageNet weights that would
        be immediately overwritten by load_state_dict().
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cfg = checkpoint.get("model_config", {})
        model = cls(
            encoder_name=cfg.get("encoder", "resnet34"),
            encoder_weights=None,   # weights come from the checkpoint, not ImageNet
            in_channels=cfg.get("in_channels", 3),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model
