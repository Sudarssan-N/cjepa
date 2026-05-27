"""Causal-LeWM training entry point — Hydra-driven.

Examples:
  # Smoke test on synthetic data, CNN encoder
  python train.py data=synthetic encoder=cnn steps=200 batch_size=8

  # Push-T + frozen DINOv2 (C-JEPA-style baseline)
  python train.py data=pusht encoder=dinov2_frozen

  # Push-T + end-to-end DINOv2 (central hypothesis)
  python train.py data=pusht encoder=dinov2_finetune
"""

import json
import os
import random
import time
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.model import CausalLeWM, CausalLeWMConfig


def seed_everything(seed: int) -> None:
    """Make a run reproducible for multi-seed experiments (was previously unset)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cfg: DictConfig):
    if cfg.data.type == "synthetic":
        from train_legacy_synthetic import SyntheticVideo
        return SyntheticVideo(
            num_samples=cfg.batch_size * cfg.steps,
            num_frames=cfg.data.num_steps,
            image_size=cfg.encoder.image_size,
            num_objects=cfg.data.get("num_objects", 3),
        )
    if cfg.data.type == "hdf5":
        from src.data import PushTHDF5
        return PushTHDF5(
            name=cfg.data.name,
            num_steps=cfg.data.num_steps,
            frameskip=cfg.data.frameskip,
            image_size=cfg.encoder.image_size,
            keys_to_load=tuple(cfg.data.keys_to_load),
            path=cfg.data.get("path", None),
            max_episodes=cfg.data.get("max_episodes", None),
        )
    raise ValueError(f"Unknown data.type={cfg.data.type}")


@hydra.main(version_base=None, config_path="configs", config_name="causal_lewm")
def main(cfg: DictConfig):
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable, falling back to CPU.")

    seed_everything(cfg.seed)
    print(f"seed: {cfg.seed}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    model_cfg = CausalLeWMConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    model = CausalLeWM(model_cfg).to(device)

    # Split params so the pretrained DINOv2 backbone trains at a much lower LR
    # than the freshly-initialized heads. With a frozen encoder the backbone
    # group is empty and this collapses to a single-group AdamW (no behavior
    # change vs the frozen baseline).
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("encoder.backbone") else head_params).append(p)
    backbone_lr = cfg.get("backbone_lr", cfg.lr)
    param_groups = [{"params": head_params, "lr": cfg.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
        nb = sum(p.numel() for p in backbone_params) / 1e6
        print(f"finetuning backbone: {nb:.1f}M params @ lr={backbone_lr} (heads @ lr={cfg.lr})")
    opt = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model params: {n_train:.2f}M trainable / {n_total:.2f}M total | device: {device}")

    use_cache = (
        cfg.get("use_dino_cache", False)
        and cfg.data.type == "hdf5"
        and cfg.encoder.name == "dinov2"
        and cfg.encoder.freeze
    )
    if use_cache:
        from src.dino_cache import build_dino_cache, default_cache_path, CachedDinoDataset

        base = build_dataset(cfg)
        cache_path = default_cache_path(base, cfg.encoder.image_size)
        build_dino_cache(base, model.encoder, device, cache_path, cfg.encoder.image_size)
        dataset = CachedDinoDataset(base, cache_path)
    else:
        dataset = build_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    # Tail buffer of per-step scalars so the reported run metric is averaged over
    # the last steps (stable) rather than read off one noisy final step.
    tail_keys = ("pred_nmse", "pred_cos", "loss_pred", "loss_recon", "slot_uniqueness")
    tail = deque(maxlen=500)

    model.train()
    step = 0
    t0 = time.time()
    while step < cfg.steps:
        for frames, actions in loader:
            if step >= cfg.steps:
                break
            frames = frames.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            out = model(frames, actions, step=step, precomputed=use_cache)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
            )
            opt.step()

            tail.append({k: out[k].item() for k in tail_keys})

            if step % cfg.log_every == 0 or step == cfg.steps - 1:
                dt = time.time() - t0
                print(
                    f"step {step:5d} | loss {out['loss'].item():.4f} "
                    f"| pred {out['loss_pred'].item():.4f} "
                    f"| sig {out['loss_sig'].item():.4f} "
                    f"| div {out['loss_div'].item():.4f} "
                    f"| decorr {out['loss_decorr'].item():.4f} "
                    f"| recon {out['loss_recon'].item():.4f} "
                    f"| nmse {out['pred_nmse'].item():.3f} "
                    f"| pcos {out['pred_cos'].item():.3f} "
                    f"| mask {out['mask_ratio'].item():.2f} "
                    f"| slot_sim {out['slot_uniqueness'].item():.3f} "
                    f"| {dt:.1f}s"
                )

            if cfg.save_every and step > 0 and step % cfg.save_every == 0:
                ckpt = out_dir / f"step_{step}.pt"
                torch.save({"model": model.state_dict(), "step": step, "cfg": OmegaConf.to_container(cfg)}, ckpt)

            step += 1

    final = out_dir / "final.pt"
    torch.save({"model": model.state_dict(), "step": step, "cfg": OmegaConf.to_container(cfg)}, final)
    print(f"saved final checkpoint -> {final}")

    # Tail-averaged run summary -> one JSON line appended to results_file, so a
    # multi-seed/ablation sweep collapses to a tiny file instead of many logs.
    n_tail = len(tail)
    summary = {k: float(np.mean([t[k] for t in tail])) for k in tail_keys} if n_tail else {}
    record = {
        "seed": int(cfg.seed),
        "slot_propagate": bool(cfg.model.slot_propagate),
        "max_episodes": cfg.data.get("max_episodes", None),
        "mask_target": float(cfg.model.mask_target),
        "steps": int(cfg.steps),
        "tail_steps": n_tail,
        "nmse": round(summary.get("pred_nmse", float("nan")), 4),
        "pcos": round(summary.get("pred_cos", float("nan")), 4),
        "pred": round(summary.get("loss_pred", float("nan")), 4),
        "recon": round(summary.get("loss_recon", float("nan")), 4),
        "slot_sim": round(summary.get("slot_uniqueness", float("nan")), 4),
        "out_dir": str(out_dir),
    }
    print("SUMMARY " + json.dumps(record))
    results_file = Path(cfg.get("results_file", "results.jsonl"))
    with results_file.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"appended run summary -> {results_file.resolve()}")


if __name__ == "__main__":
    main()
