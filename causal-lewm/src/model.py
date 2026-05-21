"""Causal-LeWM model.

End-to-end stack: Encoder -> SlotAttention -> SlotPredictor.
Loss: ℒ_pred (masked + future slot prediction) + λ_sig·SIGReg + λ_div·diversity.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from einops import rearrange

from .encoder import build_encoder
from .slot_attention import SlotAttention
from .predictor import SlotPredictor
from .sigreg import PerSlotSIGReg
from .losses import prediction_loss, slot_variance_diversity
from .object_masking import sample_object_mask, curriculum_ratio


@dataclass
class CausalLeWMConfig:
    # Encoder
    encoder_name: str = "cnn"          # "cnn" | "dinov2"
    encoder_freeze: bool = True        # only used for dinov2
    image_size: int = 64
    in_channels: int = 3
    patch_size: int = 8                # only used for cnn encoder
    dim: int = 128                     # slot / predictor dim
    # Slot attention
    num_slots: int = 7
    slot_iters: int = 3
    # Sequence
    num_frames: int = 8
    future_frames: int = 1
    action_dim: int = 2
    # Predictor
    pred_depth: int = 4
    pred_heads: int = 4
    pred_dim_head: int = 32
    pred_mlp_dim: int = 256
    # Loss weights
    lambda_sig: float = 1.0
    lambda_div: float = 0.1
    # Object-mask curriculum (steps measured in optimizer steps)
    mask_target: float = 0.4
    warmup_steps: int = 500
    ramp_steps: int = 1500


class CausalLeWM(nn.Module):
    def __init__(self, cfg: CausalLeWMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = build_encoder(
            cfg.encoder_name,
            dim=cfg.dim,
            image_size=cfg.image_size,
            freeze=cfg.encoder_freeze,
            patch_size=cfg.patch_size,
        )
        self.slot_attn = SlotAttention(cfg.num_slots, cfg.dim, iters=cfg.slot_iters)
        self.predictor = SlotPredictor(
            dim=cfg.dim,
            num_slots=cfg.num_slots,
            num_frames=cfg.num_frames,
            action_dim=cfg.action_dim,
            depth=cfg.pred_depth,
            heads=cfg.pred_heads,
            dim_head=cfg.pred_dim_head,
            mlp_dim=cfg.pred_mlp_dim,
        )
        self.sigreg = PerSlotSIGReg()

    def encode_slots(self, x: torch.Tensor, precomputed: bool = False) -> torch.Tensor:
        """x is frames (B,T,C,H,W) or, if precomputed, raw backbone tokens (B,T,P,hidden)."""
        B, T = x.shape[:2]
        if precomputed:
            toks = rearrange(x, "b t p h -> (b t) p h")
            toks = self.encoder.from_tokens(toks)
        else:
            imgs = rearrange(x, "b t c h w -> (b t) c h w")
            toks = self.encoder(imgs)
        slots = self.slot_attn(toks)
        return rearrange(slots, "(b t) n d -> b t n d", b=B, t=T)

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        step: int = 0,
        precomputed: bool = False,
    ) -> dict:
        cfg = self.cfg
        B, T = frames.shape[:2]
        device = frames.device

        slots = self.encode_slots(frames, precomputed=precomputed)
        target = slots.detach()

        ratio = curriculum_ratio(step, cfg.warmup_steps, cfg.ramp_steps, cfg.mask_target)
        history_mask = sample_object_mask(B, T - cfg.future_frames, cfg.num_slots, ratio, device)
        future_mask = torch.ones(B, cfg.future_frames, cfg.num_slots, dtype=torch.bool, device=device)
        slot_mask = torch.cat([history_mask, future_mask], dim=1)

        pred = self.predictor(slots, actions, slot_mask=slot_mask)

        loss_pred = prediction_loss(pred, target, slot_mask.float())
        loss_sig = self.sigreg(slots)
        loss_div = slot_variance_diversity(slots)
        loss = loss_pred + cfg.lambda_sig * loss_sig + cfg.lambda_div * loss_div

        with torch.no_grad():
            s = slots.reshape(B * T, cfg.num_slots, cfg.dim)
            s = torch.nn.functional.normalize(s, dim=-1)
            sim = torch.einsum("bnd,bmd->bnm", s, s)
            eye = torch.eye(cfg.num_slots, device=device).unsqueeze(0)
            slot_uniqueness = (sim * (1 - eye)).sum() / (B * T * cfg.num_slots * (cfg.num_slots - 1))

        return {
            "loss": loss,
            "loss_pred": loss_pred.detach(),
            "loss_sig": loss_sig.detach(),
            "loss_div": loss_div.detach(),
            "mask_ratio": torch.tensor(ratio, device=device),
            "slot_uniqueness": slot_uniqueness,
        }

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def rollout(
        self,
        frames: torch.Tensor,           # (B, H_init, C, H, W) — initial history
        action_sequence: torch.Tensor,  # (B, S, T, action_dim)
        history_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Adapted from le-wm/jepa.py:rollout for slot embeddings.

        Returns slots: (B, S, T, N, D) with the predicted future slots
        appended after the initial history.
        """
        cfg = self.cfg
        was_training = self.training
        self.eval()

        H = frames.size(1)
        HS = history_size or H
        B, S, T, _ = action_sequence.shape

        init_slots = self.encode_slots(frames)        # (B, H, N, D)
        slots = init_slots.unsqueeze(1).expand(B, S, H, cfg.num_slots, cfg.dim)
        slots = slots.reshape(B * S, H, cfg.num_slots, cfg.dim).clone()
        actions = action_sequence.reshape(B * S, T, -1)

        n_steps = T - H
        for t in range(n_steps):
            ctx = slots[:, -HS:]
            act_ctx = actions[:, t : t + HS] if (t + HS) <= T else actions[:, -HS:]
            # Future slot is a single masked step appended to the context.
            mask_token = self.predictor.mask_token.expand(B * S, 1, cfg.num_slots, cfg.dim)
            ctx_in = torch.cat([ctx, mask_token], dim=1)[:, -HS:]  # keep window size
            act_in = act_ctx
            mask = torch.zeros(B * S, ctx_in.size(1), cfg.num_slots, dtype=torch.bool, device=frames.device)
            mask[:, -1] = True
            pred = self.predictor(ctx_in, act_in, slot_mask=mask)
            next_slot = pred[:, -1:]                  # (BS, 1, N, D)
            slots = torch.cat([slots, next_slot], dim=1)

        slots = slots.reshape(B, S, slots.size(1), cfg.num_slots, cfg.dim)
        if was_training:
            self.train()
        return slots

    @torch.no_grad()
    def plan_mpc(
        self,
        history_frames: torch.Tensor,   # (B, H, C, H, W)
        goal_frame: torch.Tensor,       # (B, C, H, W)
        action_dim: int,
        horizon: int = 10,
        n_samples: int = 256,
        n_iters: int = 3,
        elite_frac: float = 0.1,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> torch.Tensor:
        """Cross-Entropy Method MPC at the slot level.

        Cost = MSE between the last predicted slots and the encoded goal
        slots (averaged over slots and dims), mirroring LeWM's criterion
        (`le-wm/jepa.py:criterion`) lifted to (N, D) slot tensors.

        Returns the best action sequence: (B, horizon, action_dim).
        """
        device = history_frames.device
        B, H = history_frames.shape[:2]

        goal_slots = self.encode_slots(goal_frame.unsqueeze(1))[:, 0]    # (B, N, D)

        mu = torch.zeros(B, horizon, action_dim, device=device)
        sigma = torch.full_like(mu, (action_high - action_low) / 2.0)

        elite_k = max(1, int(round(n_samples * elite_frac)))
        for _ in range(n_iters):
            samples = mu.unsqueeze(1) + sigma.unsqueeze(1) * torch.randn(
                B, n_samples, horizon, action_dim, device=device
            )
            samples = samples.clamp(action_low, action_high)

            # Build action_sequence covering the history window + horizon.
            # We pad zeros for the history actions because the predictor
            # only consumes them as conditioning context for the rollout
            # of the *new* steps.
            pad = torch.zeros(B, n_samples, H, action_dim, device=device)
            full = torch.cat([pad, samples], dim=2)                       # (B, S, H+horizon, A)

            slots = self.rollout(history_frames, full, history_size=H)    # (B, S, H+horizon, N, D)
            pred_last = slots[:, :, -1]                                   # (B, S, N, D)
            cost = (pred_last - goal_slots.unsqueeze(1)).pow(2).mean(dim=(-1, -2))  # (B, S)

            # Pick elites per batch element and refit.
            _, elite_idx = cost.topk(elite_k, largest=False, dim=1)
            elite = torch.gather(
                samples, 1, elite_idx[..., None, None].expand(-1, -1, horizon, action_dim)
            )
            mu = elite.mean(dim=1)
            sigma = elite.std(dim=1).clamp_min(1e-3)

        return mu
