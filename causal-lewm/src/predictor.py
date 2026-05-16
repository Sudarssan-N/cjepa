"""Slot-level transformer predictor.

Adapted from `le-wm/module.py` (ARPredictor + ConditionalBlock). Operates
over (B, T, N, D) slot sequences flattened to (B, T*N, D) so attention
can mix tokens across both time and object slots. AdaLN-zero conditions
each block on the action embedding broadcast over slots.
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, attn_mask=None):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=False
        )
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class SlotPredictor(nn.Module):
    """Predicts next-step slot embeddings from a (B, T, N, D) history.

    A learnable mask token replaces masked-out slot positions. Time and
    slot-index positional embeddings are added before the transformer.
    """

    def __init__(
        self,
        dim: int,
        num_slots: int,
        num_frames: int,
        action_dim: int,
        depth: int = 4,
        heads: int = 4,
        dim_head: int = 32,
        mlp_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_frames = num_frames
        self.time_pos = nn.Parameter(torch.randn(1, num_frames, 1, dim) * 0.02)
        self.slot_pos = nn.Parameter(torch.randn(1, 1, num_slots, dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, dim) * 0.02)
        self.act_proj = nn.Linear(action_dim, dim)
        self.blocks = nn.ModuleList(
            [ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(
        self,
        slots: torch.Tensor,           # (B, T, N, D)
        act_emb: torch.Tensor,         # (B, T, A)
        slot_mask: torch.Tensor | None = None,  # (B, T, N) bool, True = masked
    ) -> torch.Tensor:
        B, T, N, D = slots.shape

        if slot_mask is not None:
            mt = self.mask_token.expand(B, T, N, D)
            slots = torch.where(slot_mask.unsqueeze(-1), mt, slots)

        x = slots + self.time_pos[:, :T] + self.slot_pos[:, :, :N]
        x = rearrange(x, "b t n d -> b (t n) d")

        c_act = self.act_proj(act_emb)                  # (B, T, D)
        c = c_act.unsqueeze(2).expand(B, T, N, D)
        c = rearrange(c, "b t n d -> b (t n) d")

        for blk in self.blocks:
            x = blk(x, c)

        x = self.norm(x)
        x = self.head(x)
        return rearrange(x, "b (t n) d -> b t n d", t=T)
