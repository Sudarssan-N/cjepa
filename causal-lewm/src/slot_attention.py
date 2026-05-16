"""Slot Attention (Locatello et al., 2020) — minimal, trainable.

Iteratively groups patch tokens into N slot vectors via competitive
softmax over slots. Used as the object-centric bottleneck between the
patch encoder and the JEPA predictor.
"""

import torch
from torch import nn


class SlotAttention(nn.Module):
    def __init__(
        self,
        num_slots: int,
        dim: int,
        iters: int = 3,
        hidden_dim: int = 128,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_mlp = nn.LayerNorm(dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: (B, P, D) -> slots: (B, N, D)."""
        B, P, D = inputs.shape
        N = self.num_slots

        mu = self.slots_mu.expand(B, N, D)
        sigma = self.slots_logsigma.exp().expand(B, N, D)
        slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            s = self.norm_slots(slots)
            q = self.to_q(s)

            attn_logits = torch.einsum("bnd,bpd->bnp", q, k) * self.scale
            attn = attn_logits.softmax(dim=1) + self.eps  # competition over slots
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bnp,bpd->bnd", attn, v)

            slots = self.gru(
                updates.reshape(B * N, D), slots_prev.reshape(B * N, D)
            ).reshape(B, N, D)
            slots = slots + self.mlp(self.norm_pre_mlp(slots))

        return slots
