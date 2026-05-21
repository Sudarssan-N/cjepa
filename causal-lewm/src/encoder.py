"""Patch encoders.

- `PatchEncoder`: tiny CNN baseline used for fast smoke tests.
- `DinoV2Encoder`: DINOv2 ViT-S/14 from HuggingFace `facebook/dinov2-small`.
  Returns the patch token sequence (drops the [CLS] token) projected to
  `dim` so downstream slot attention sees a uniform feature width.

`freeze=True` (the default) reproduces C-JEPA's frozen-encoder regime —
useful as the §5.2 ablation baseline. Set `freeze=False` to test the
central hypothesis (joint encoder + slot + predictor training).
"""

from __future__ import annotations

import torch
from torch import nn


class PatchEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 128, patch_size: int = 8):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(z)


class DinoV2Encoder(nn.Module):
    """Wraps `facebook/dinov2-small` (ViT-S/14, ~21M params).

    Image side must be a multiple of 14. With the LeWM default of 224 you
    get 16x16 = 256 patch tokens.
    """

    HF_NAME = "facebook/dinov2-small"

    def __init__(self, dim: int = 128, freeze: bool = True, image_size: int = 224):
        super().__init__()
        from transformers import AutoModel  # local import: heavy dep, only when used

        if image_size % 14 != 0:
            raise ValueError(f"DINOv2 needs image_size % 14 == 0; got {image_size}.")
        self.image_size = image_size
        self.backbone = AutoModel.from_pretrained(self.HF_NAME)
        hidden = self.backbone.config.hidden_size  # 384 for dinov2-small

        self.proj = nn.Linear(hidden, dim) if hidden != dim else nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        self.hidden = hidden

        # ImageNet normalization that DINOv2 was trained with.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def backbone_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) in [0, 1] -> raw frozen patch tokens (B, P, hidden).

        No trainable proj/norm applied — this is what the feature cache stores.
        """
        x = (x - self.mean) / self.std
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            try:
                out = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
            except TypeError:
                # Older transformers versions don't accept the kwarg.
                out = self.backbone(pixel_values=x)
        return out.last_hidden_state[:, 1:]  # drop [CLS]

    def from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply the trainable proj+norm to backbone tokens (B, P, hidden) -> (B, P, dim)."""
        return self.norm(self.proj(tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) in [0, 1] -> tokens: (B, P, dim)."""
        return self.from_tokens(self.backbone_tokens(x))


def build_encoder(name: str, dim: int, image_size: int, freeze: bool, patch_size: int = 8) -> nn.Module:
    name = name.lower()
    if name in ("cnn", "patch", "tiny"):
        return PatchEncoder(in_channels=3, dim=dim, patch_size=patch_size)
    if name in ("dinov2", "dinov2-small", "dino"):
        return DinoV2Encoder(dim=dim, freeze=freeze, image_size=image_size)
    raise ValueError(f"Unknown encoder: {name}")
