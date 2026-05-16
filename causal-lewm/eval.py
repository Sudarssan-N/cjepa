"""Slot-level MPC evaluation entry point.

Loads a Causal-LeWM checkpoint and plans an action sequence given a
history window and a goal frame. This is the analogue of `le-wm/eval.py`
adapted to slot embeddings — it does not yet integrate with the
stable-worldmodel environments, so it currently runs CEM on a synthetic
goal pulled from the same loader used for training.

Usage:
  python eval.py --ckpt outputs/.../final.pt --horizon 10 --n-samples 256
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.model import CausalLeWM, CausalLeWMConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--n-iters", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    blob = torch.load(args.ckpt, map_location=args.device)
    cfg = OmegaConf.create(blob["cfg"])
    model_cfg = CausalLeWMConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    model = CausalLeWM(model_cfg).to(args.device)
    model.load_state_dict(blob["model"])
    model.eval()

    if cfg.data.type == "synthetic":
        from train_legacy_synthetic import SyntheticVideo
        ds = SyntheticVideo(
            num_samples=4, num_frames=cfg.data.num_steps + args.horizon,
            image_size=cfg.encoder.image_size, num_objects=cfg.data.get("num_objects", 3),
        )
    else:
        from src.data import PushTHDF5
        ds = PushTHDF5(
            name=cfg.data.name,
            num_steps=cfg.data.num_steps + args.horizon,
            frameskip=cfg.data.frameskip,
            image_size=cfg.encoder.image_size,
            keys_to_load=tuple(cfg.data.keys_to_load),
        )

    loader = DataLoader(ds, batch_size=2, shuffle=False)
    frames, _ = next(iter(loader))                       # (B, T, C, H, W)
    frames = frames.to(args.device)

    H = cfg.data.num_steps - cfg.data.num_preds          # history length
    history = frames[:, :H]
    goal = frames[:, -1]

    plan = model.plan_mpc(
        history_frames=history,
        goal_frame=goal,
        action_dim=cfg.model.action_dim,
        horizon=args.horizon,
        n_samples=args.n_samples,
        n_iters=args.n_iters,
    )
    print(f"plan shape: {tuple(plan.shape)}")
    print(f"first action[0]: {plan[0, 0].tolist()}")
    print(f"plan stats: mean={plan.mean().item():.3f} std={plan.std().item():.3f}")


if __name__ == "__main__":
    main()
