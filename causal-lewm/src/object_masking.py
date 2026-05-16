"""Object-level masking — §4.4 of the research plan.

C-JEPA uses object masking on a frozen encoder. With a trainable encoder
the masking pattern shapes gradients flowing back through slot attention
so we ramp the ratio with a curriculum.
"""

import torch


def sample_object_mask(
    batch_size: int,
    num_frames: int,
    num_slots: int,
    ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Per-frame, per-batch boolean mask. Shape (B, T, N), True = masked."""
    if ratio <= 0:
        return torch.zeros(batch_size, num_frames, num_slots, dtype=torch.bool, device=device)
    k = max(1, int(round(num_slots * ratio)))
    rand = torch.rand(batch_size, num_frames, num_slots, device=device)
    _, idx = rand.topk(k, dim=-1)
    mask = torch.zeros(batch_size, num_frames, num_slots, dtype=torch.bool, device=device)
    mask.scatter_(-1, idx, True)
    return mask


def curriculum_ratio(step: int, warmup_steps: int, ramp_steps: int, target: float) -> float:
    if step < warmup_steps:
        return 0.0
    if step >= warmup_steps + ramp_steps:
        return target
    return target * (step - warmup_steps) / ramp_steps
