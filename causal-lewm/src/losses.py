"""Loss helpers for Causal-LeWM.

- prediction_loss: MSE on masked + future slot positions only
- slot_variance_diversity: VICReg-style per-slot variance hinge (§4.2)
"""

import torch
import torch.nn.functional as F


def prediction_loss(
    pred: torch.Tensor,         # (B, T, N, D)
    target: torch.Tensor,       # (B, T, N, D), already detached if needed
    weight_mask: torch.Tensor,  # (B, T, N) float — where to apply the loss
) -> torch.Tensor:
    err = F.mse_loss(pred, target, reduction="none").mean(dim=-1)  # (B, T, N)
    denom = weight_mask.sum().clamp_min(1.0)
    return (err * weight_mask).sum() / denom


def slot_variance_diversity(slots: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Hinge on per-(slot, dim) std across the batch * time axis.

    Encourages each slot index to vary independently — counter-pressure
    against slot collapse where every slot binds to the same content.
    """
    B, T, N, D = slots.shape
    flat = slots.reshape(B * T, N, D)
    std = flat.std(dim=0)  # (N, D)
    return torch.relu(1.0 - std + eps).mean()


def slot_decorrelation(slots: torch.Tensor) -> torch.Tensor:
    """Mean positive off-diagonal cosine similarity between slots *within* a frame.

    `slot_variance_diversity` and SIGReg both leave the within-frame collapse
    (all N slots identical) unpenalized — they only constrain each slot index's
    marginal across the batch. This term directly pushes the N slots of a frame
    apart so they bind to distinct content, the role previously played (by
    accident) by the random per-forward init noise.
    """
    B, T, N, D = slots.shape
    s = F.normalize(slots, dim=-1)
    sim = torch.einsum("btnd,btmd->btnm", s, s)
    eye = torch.eye(N, device=slots.device).view(1, 1, N, N)
    off = (sim * (1.0 - eye)).clamp(min=0.0)
    return off.sum() / (B * T * N * (N - 1))
