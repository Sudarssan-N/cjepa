"""SIGReg — Sketched Isotropic Gaussian Regularizer.

Ported from `le-wm/module.py` (Maes et al., 2026). The base implementation
is unchanged. We add slot-aware wrappers used by Causal-LeWM.
"""

import torch
from torch import nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (single-GPU)."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """proj: (..., B, D) — last two dims are batch and feature."""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class PerSlotSIGReg(nn.Module):
    """§4.3 Option A — independent SIGReg per slot index, averaged.

    Input slots: (B, T, N, D). We treat each slot index `i` as its own
    distribution and apply SIGReg to the (B*T, D) sample for that slot.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.sigreg = SIGReg(knots=knots, num_proj=num_proj)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        B, T, N, D = slots.shape
        flat = slots.reshape(B * T, N, D)
        losses = []
        for i in range(N):
            losses.append(self.sigreg(flat[:, i, :]))
        return torch.stack(losses).mean()
