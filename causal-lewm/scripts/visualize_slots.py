"""Visualize what each slot binds to, from a trained checkpoint.

Loads a `final.pt` (or `step_*.pt`), encodes a few real Push-T windows, runs the
DINOSAUR spatial-broadcast decoder, and renders the per-slot alpha masks (the
softmax-over-slots competition weights) overlaid on each frame — plus a hard
argmax segmentation showing which slot owns each patch.

This is the qualitative artifact for the paper: it shows whether slots bind to
the T-block / pusher / background rather than carrying decorrelated noise.

Usage (Colab, after clone + having the .h5):
  python scripts/visualize_slots.py \
      --ckpt /content/final.pt \
      --data /content/pusht_expert_train.h5 \
      --max-episodes 500 --windows 4 --out /content/slots.png

The checkpoint carries its own cfg, so no Hydra is needed. Backbone weights are
re-pulled from HuggingFace (they're frozen, so identical to training).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import PushTHDF5  # noqa: E402
from src.model import CausalLeWM, CausalLeWMConfig  # noqa: E402


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt["cfg"]["model"]
    cfg = CausalLeWMConfig(**model_cfg)
    model = CausalLeWM(cfg).to(device)
    # strict=False: the frozen DINOv2 backbone is reloaded from HF inside the
    # model ctor; HF param names can drift across transformers versions, but the
    # trainable proj/norm/slot/predictor/decoder weights load by exact name.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    trainable_missing = [k for k in missing if not k.startswith("encoder.backbone")]
    if trainable_missing:
        print(f"WARNING: missing non-backbone keys: {trainable_missing[:8]} ...")
    print(f"loaded {ckpt_path.name} @ step {ckpt.get('step', '?')}; "
          f"slot_propagate={cfg.slot_propagate}, num_slots={cfg.num_slots}")
    model.eval()
    return model, cfg


@torch.no_grad()
def slot_masks(model, frames, device):
    """frames (1,T,C,H,W) -> (masks (T,N,h,w), slots (T,N,D))."""
    cfg = model.cfg
    slots, _ = model.encode_slots(frames.to(device), precomputed=False)  # (1,T,N,D)
    T = slots.shape[1]
    _, weights = model.decoder(slots.reshape(T, cfg.num_slots, cfg.dim))  # (T,N,P)
    P = weights.shape[-1]
    h = w = int(round(math.sqrt(P)))
    masks = weights.reshape(T, cfg.num_slots, h, w).cpu().numpy()
    return masks, slots[0].cpu().numpy()


def render(frames_np, masks, out_path, slot_propagate):
    """frames_np (T,C,H,W) in [0,1]; masks (T,N,h,w)."""
    T, N = masks.shape[0], masks.shape[1]
    H = frames_np.shape[2]
    cmap = plt.get_cmap("tab10")

    ncols = N + 2  # original + segmentation + N slots
    fig, axes = plt.subplots(T, ncols, figsize=(2.0 * ncols, 2.0 * T), squeeze=False)
    for t in range(T):
        img = np.transpose(frames_np[t], (1, 2, 0))  # H,W,C
        m = masks[t]  # (N,h,w)
        # upsample masks to image size (nearest is fine for patch grids)
        up = np.repeat(np.repeat(m, H // m.shape[1], 1), H // m.shape[2], 2)
        seg = up.argmax(0)  # (H,W) slot id per pixel

        axes[t][0].imshow(img)
        axes[t][0].set_ylabel(f"t={t}", fontsize=9)
        if t == 0:
            axes[t][0].set_title("frame", fontsize=9)

        axes[t][1].imshow(img)
        axes[t][1].imshow(cmap(seg / max(1, N - 1))[..., :3], alpha=0.55)
        if t == 0:
            axes[t][1].set_title("segmentation", fontsize=9)

        for n in range(N):
            ax = axes[t][n + 2]
            ax.imshow(img)
            ax.imshow(up[n], cmap="inferno", alpha=0.6, vmin=0, vmax=up.max())
            if t == 0:
                ax.set_title(f"slot {n}", fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Slot decoder alpha masks  (slot_propagate={slot_propagate})", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--data", required=True, help="path to pusht_expert_train.h5(.zst)")
    ap.add_argument("--max-episodes", type=int, default=500)
    ap.add_argument("--windows", type=int, default=4, help="how many windows to render")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out", type=Path, default=Path("slots.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)
    if model.decoder is None:
        raise SystemExit("checkpoint has no decoder (lambda_recon was 0) — nothing to draw.")

    ds = PushTHDF5(
        name="pusht_expert_train",
        num_steps=cfg.num_frames,
        image_size=args.image_size,
        path=args.data,
        max_episodes=args.max_episodes,
    )
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=min(args.windows, len(ds)), replace=False)
    print(f"dataset windows: {len(ds)}; rendering {len(idxs)}: {idxs.tolist()}")

    for j, idx in enumerate(idxs):
        frames, _ = ds[int(idx)]               # (T,C,H,W)
        frames = frames.unsqueeze(0)           # (1,T,C,H,W)
        masks, _ = slot_masks(model, frames, device)
        out = args.out if len(idxs) == 1 else args.out.with_name(
            f"{args.out.stem}_{j}{args.out.suffix}"
        )
        render(frames[0].numpy(), masks, out, cfg.slot_propagate)


if __name__ == "__main__":
    main()
