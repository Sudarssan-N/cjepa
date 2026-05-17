"""HDF5 dataset compatible with LeWM's `pusht_expert_train.h5` layout.

LeWM's `swm.data.HDF5Dataset` is a thin wrapper that returns dicts with
`pixels`, `action`, `proprio`, `state` keys. We re-implement just the
pieces Causal-LeWM needs so this repo doesn't require installing
`stable-worldmodel` for a first end-to-end run on Push-T.

File layout (Push-T expert):
  episode_<i>/pixels   (T, H, W, 3) uint8
  episode_<i>/action   (T, action_dim) float32
  episode_<i>/proprio  (T, proprio_dim) float32
  episode_<i>/state    (T, state_dim) float32

The exact key names are read at open time so this works for the
upstream LeWM HDF5 even if minor field names differ.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _stablewm_home() -> Path:
    return Path(os.environ.get("STABLEWM_HOME", str(Path.home() / ".stable-wm")))


def _decompress_zst(src: Path) -> Path:
    """Decompress src (.h5.zst) → src without the .zst suffix, return the .h5 path."""
    dst = src.with_suffix("")  # strips .zst → .h5
    if dst.exists():
        return dst
    print(f"Decompressing {src} → {dst} (one-time, ~2–5 min for 13 GB) …", flush=True)
    try:
        import zstandard as zstd  # type: ignore
        with src.open("rb") as fin, dst.open("wb") as fout:
            dctx = zstd.ZstdDecompressor()
            dctx.copy_stream(fin, fout)
    except ImportError:
        # fall back to the zstd CLI if the Python library isn't installed
        result = subprocess.run(["zstd", "-d", str(src), "-o", str(dst)], check=False)
        if result.returncode != 0:
            dst.unlink(missing_ok=True)
            sys.exit(
                "zstd decompression failed. Install via: pip install zstandard  "
                "or: conda install -c conda-forge zstd"
            )
    print(f"Decompression complete → {dst}", flush=True)
    return dst


class PushTHDF5(Dataset):
    """Yields (frames, actions) windows for next-step JEPA training.

    Args:
      name: dataset name without `.h5` (e.g. `pusht_expert_train`)
      num_steps: window length in frames
      frameskip: stride between sampled frames within a window
      image_size: resize target (square)
      keys_to_load: subset of ['pixels', 'action', 'proprio', 'state']
    """

    def __init__(
        self,
        name: str = "pusht_expert_train",
        num_steps: int = 4,
        frameskip: int = 5,
        image_size: int = 224,
        keys_to_load: Sequence[str] = ("pixels", "action"),
        path: str | None = None,
    ):
        self.path = Path(path) if path else _stablewm_home() / f"{name}.h5"
        if not self.path.exists():
            zst = self.path.with_suffix(self.path.suffix + ".zst")
            if self.path.suffix != ".zst" and zst.exists():
                self.path = _decompress_zst(zst)
            elif self.path.suffix == ".zst":
                self.path = _decompress_zst(self.path)
            else:
                raise FileNotFoundError(
                    f"{self.path} not found. Run scripts/download_pusht.sh, "
                    f"set STABLEWM_HOME to where the .h5 lives, "
                    f"or pass data.path=/path/to/pusht_expert_train.h5.zst"
                )
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.image_size = image_size
        self.keys_to_load = list(keys_to_load)

        # Build (episode_key, start_frame) index of valid windows.
        self._h5: h5py.File | None = None
        self._index: list[tuple[str, int]] = []
        with h5py.File(self.path, "r") as f:
            ep_keys = sorted(k for k in f.keys() if k.startswith("episode"))
            if not ep_keys:
                ep_keys = sorted(f.keys())  # fall back: top-level groups are episodes
            self._episode_keys = ep_keys

            for ek in ep_keys:
                grp = f[ek]
                pix_key = "pixels" if "pixels" in grp else next(iter(grp.keys()))
                T = grp[pix_key].shape[0]
                span = (num_steps - 1) * frameskip + 1
                for s in range(0, T - span + 1, frameskip):
                    self._index.append((ek, s))

    def __len__(self):
        return len(self._index)

    def _open(self) -> h5py.File:
        # h5py file handles are not safe to share across workers; open lazily.
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __getitem__(self, idx: int):
        ek, start = self._index[idx]
        f = self._open()
        grp = f[ek]
        idxs = [start + i * self.frameskip for i in range(self.num_steps)]

        pixels = grp["pixels"][idxs]                    # (T, H, W, 3) uint8 typically
        if pixels.dtype != np.uint8:
            pixels = (pixels * 255).clip(0, 255).astype(np.uint8) if pixels.max() <= 1 else pixels.astype(np.uint8)

        # to (T, 3, H, W) float in [0,1] then resize via simple nearest if needed
        x = torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )

        action = torch.from_numpy(np.asarray(grp["action"][idxs], dtype=np.float32))
        action = torch.nan_to_num(action, 0.0)

        return x, action
