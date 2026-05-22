"""Spatial-broadcast slot decoder (DINOSAUR-style feature reconstruction).

Each slot is broadcast across all patch positions, an MLP predicts a feature
vector and an alpha logit per (slot, position), and slots compete via a softmax
over alphas to reconstruct the target patch features. Forcing the slots to
reconstruct the frozen DINOv2 patch tokens grounds them in real spatial/object
content — the counter-pressure missing when only prediction + SIGReg + diversity
are active (which a decorrelated-noise representation satisfies trivially).

Ref: Seitzer et al., "Bridging the Gap to Real-World Object-Centric Learning"
(DINOSAUR), ICLR 2023.
"""

from __future__ import annotations

import torch
from torch import nn


class SpatialBroadcastDecoder(nn.Module):
    def __init__(
        self,
        slot_dim: int,
        feat_dim: int,
        num_patches: int,
        hidden: int = 512,
        depth: int = 3,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.feat_dim = feat_dim
        self.pos = nn.Parameter(torch.randn(1, 1, num_patches, slot_dim) * 0.02)

        layers: list[nn.Module] = [nn.Linear(slot_dim, hidden), nn.ReLU(inplace=True)]
        for _ in range(max(0, depth - 2)):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, feat_dim + 1)  # feature + alpha logit

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """slots: (B, N, D) -> (recon (B, P, feat_dim), masks (B, N, P))."""
        B, N, D = slots.shape
        P = self.num_patches
        x = slots.unsqueeze(2).expand(B, N, P, D) + self.pos  # (B, N, P, D)
        x = self.head(self.mlp(x))                            # (B, N, P, feat_dim + 1)
        feat, alpha = x[..., :-1], x[..., -1:]
        weights = alpha.softmax(dim=1)                        # compete over slots
        recon = (weights * feat).sum(dim=1)                   # (B, P, feat_dim)
        return recon, weights.squeeze(-1)
