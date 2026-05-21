"""HDF5 dataset compatible with LeWM's `pusht_expert_train.h5` layout.

LeWM's `swm.data.HDF5Dataset` is a thin wrapper that returns dicts with
`pixels`, `action`, `proprio`, `state` keys. We re-implement just the
pieces Causal-LeWM needs so this repo doesn't require installing
`stable-worldmodel` for a first end-to-end run on Push-T.

Two layouts are auto-detected at open time:

  (a) Episode-grouped (one HDF5 Group per episode):
        episode_<i>/pixels   (T, H, W, 3) uint8
        episode_<i>/action   (T, action_dim) float32
        episode_<i>/proprio  (T, proprio_dim) float32
        episode_<i>/state    (T, state_dim) float32

  (b) Flat (stable-worldmodel layout: top-level Datasets, all episodes concatenated):
        pixels   (N_total, H, W, 3) uint8
        action   (N_total, action_dim) float32
        proprio  (N_total, proprio_dim) float32
        state    (N_total, state_dim) float32
        # plus one of: episode_index (per-frame ep id) or
        # episode_starts/episode_ends/episode_lengths (per-episode offsets).
        # If none of these are present, the file is treated as one episode.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import h5py

# Registers Blosc/LZ4/Zstd/etc. HDF5 codec plugins. The LeWM/stable-worldmodel
# Push-T files chunk-compress `pixels` with one of these filters; without the
# plugin, reads fail with "Can't synchronously read data (can't open directory
# .../plugin)". Imported at module level so DataLoader workers register it too.
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

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
        max_episodes: int | None = None,
    ):
        self.path = Path(path) if path else _stablewm_home() / f"{name}.h5"
        if self.path.suffix == ".zst":
            self.path = _decompress_zst(self.path)
        elif not self.path.exists():
            zst = self.path.with_suffix(self.path.suffix + ".zst")
            if zst.exists():
                self.path = _decompress_zst(zst)
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
        self.max_episodes = max_episodes

        # Build the window index. Each entry is (episode_offset, length, start_within_episode).
        # For grouped layout, episode_offset is the group key; for flat, it's the absolute
        # frame offset into the global `pixels` dataset.
        self._h5: h5py.File | None = None
        self._index: list[tuple] = []
        span = (num_steps - 1) * frameskip + 1

        with h5py.File(self.path, "r") as f:
            top_keys = sorted(f.keys())
            if not top_keys:
                raise RuntimeError(f"{self.path} has no top-level keys")
            first = f[top_keys[0]]

            if isinstance(first, h5py.Group):
                # ---- Layout (a): one Group per episode ----
                self._layout = "grouped"
                ep_keys = [k for k in top_keys if k.startswith("episode")] or top_keys
                if max_episodes is not None:
                    ep_keys = ep_keys[:max_episodes]
                self._episode_keys = ep_keys
                for ek in ep_keys:
                    grp = f[ek]
                    pix_key = "pixels" if "pixels" in grp else next(iter(grp.keys()))
                    T = grp[pix_key].shape[0]
                    for s in range(0, T - span + 1, frameskip):
                        self._index.append((ek, s))
            else:
                # ---- Layout (b): flat top-level Datasets ----
                self._layout = "flat"
                if "pixels" not in f:
                    raise KeyError(
                        f"{self.path} is flat but has no 'pixels' dataset; "
                        f"top-level keys: {top_keys}"
                    )
                pixels_ds = f["pixels"]
                if pixels_ds.ndim == 5:
                    # (E, T, H, W, C): per-episode along axis 0
                    E, T = pixels_ds.shape[:2]
                    if max_episodes is not None:
                        E = min(E, max_episodes)
                    self._flat_shape = "5d"
                    self._episode_keys = [f"ep{e}" for e in range(E)]
                    for e in range(E):
                        for s in range(0, T - span + 1, frameskip):
                            # absolute frame index into the flattened view
                            self._index.append((e * T + s, s))
                    self._ep_len = T
                elif pixels_ds.ndim == 4:
                    # (N_total, H, W, C): need episode boundaries
                    N = pixels_ds.shape[0]
                    self._flat_shape = "4d"
                    starts, ends = self._infer_episode_bounds(f, N)
                    bounds = list(zip(starts, ends))
                    if max_episodes is not None:
                        bounds = bounds[:max_episodes]
                    self._episode_keys = [f"ep{i}" for i in range(len(bounds))]
                    self._ep_bounds = bounds
                    for s0, e0 in self._ep_bounds:
                        T_ep = e0 - s0
                        for s in range(0, T_ep - span + 1, frameskip):
                            self._index.append((s0 + s,))
                else:
                    raise ValueError(
                        f"Unexpected pixels ndim={pixels_ds.ndim} (shape {pixels_ds.shape})"
                    )

        if not self._index:
            raise RuntimeError(
                f"No valid windows in {self.path}: num_steps={num_steps}, "
                f"frameskip={frameskip} require at least {span} frames per episode."
            )

    @staticmethod
    def _infer_episode_bounds(f: h5py.File, N: int) -> tuple[list[int], list[int]]:
        """Return (starts, ends) for each episode given a flat HDF5 file with N frames."""
        # Per-episode offset + length (stable-worldmodel layout)
        if "ep_offset" in f and "ep_len" in f:
            offs = list(map(int, f["ep_offset"][:]))
            lens = list(map(int, f["ep_len"][:]))
            return offs, [o + l for o, l in zip(offs, lens)]
        # Explicit start/end arrays
        if "episode_starts" in f and "episode_ends" in f:
            return list(map(int, f["episode_starts"][:])), list(map(int, f["episode_ends"][:]))
        if "episode_starts" in f:
            s = list(map(int, f["episode_starts"][:]))
            return s, s[1:] + [N]
        # Per-episode lengths
        for key in ("episode_lengths", "episode_length", "ep_lengths", "ep_len"):
            if key in f:
                lens = list(map(int, f[key][:]))
                starts = np.cumsum([0] + lens[:-1]).tolist()
                ends = np.cumsum(lens).tolist()
                return starts, ends
        # Per-frame episode id
        for key in ("episode_index", "episode_idx", "episode_ids", "episode_id", "ep_index", "ep_id"):
            if key in f:
                ep_ids = np.asarray(f[key][:])
                if ep_ids.shape[0] != N:
                    continue
                starts = [0]
                for i in range(1, N):
                    if ep_ids[i] != ep_ids[i - 1]:
                        starts.append(i)
                ends = starts[1:] + [N]
                return starts, ends
        # Fallback: one long episode
        print(
            "[PushTHDF5] No episode-boundary dataset found; treating the whole file "
            "as a single episode. Available top-level keys: "
            f"{sorted(f.keys())}",
            flush=True,
        )
        return [0], [N]

    def __len__(self):
        return len(self._index)

    def _open(self) -> h5py.File:
        # h5py file handles are not safe to share across workers; open lazily.
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def _read_window(self, f: h5py.File, idx_entry: tuple) -> tuple[np.ndarray, np.ndarray]:
        """Return (pixels[T,H,W,C], action[T,A]) for the window at idx_entry."""
        if self._layout == "grouped":
            ek, start = idx_entry
            grp = f[ek]
            idxs = [start + i * self.frameskip for i in range(self.num_steps)]
            return grp["pixels"][idxs], np.asarray(grp["action"][idxs], dtype=np.float32)

        # flat
        if self._flat_shape == "5d":
            abs_start, s = idx_entry
            e = abs_start // self._ep_len
            idxs = [s + i * self.frameskip for i in range(self.num_steps)]
            pix = f["pixels"][e, idxs]
            act = np.asarray(f["action"][e, idxs], dtype=np.float32)
            return pix, act

        # flat 4d
        (abs_start,) = idx_entry
        idxs = [abs_start + i * self.frameskip for i in range(self.num_steps)]
        pix = f["pixels"][idxs]
        act = np.asarray(f["action"][idxs], dtype=np.float32)
        return pix, act

    def _window_base(self, idx: int) -> int:
        """Global frame offset of window `idx` (flat layouts only)."""
        if self._layout != "flat":
            raise NotImplementedError("Feature caching supports flat layouts only.")
        return self._index[idx][0]

    def get_frame_ids(self, idx: int) -> list[int]:
        """Global frame indices sampled by window `idx` (flat layouts only).

        For 4d the id indexes the global `pixels` axis; for 5d it is the
        flattened index e * ep_len + frame.
        """
        base = self._window_base(idx)
        return [base + i * self.frameskip for i in range(self.num_steps)]

    def read_frames_by_id(self, ids: Sequence[int]) -> np.ndarray:
        """Read raw pixels for global frame `ids` (must be sorted & unique)."""
        f = self._open()
        if self._layout != "flat":
            raise NotImplementedError("Feature caching supports flat layouts only.")
        if self._flat_shape == "4d":
            return f["pixels"][list(ids)]
        T = self._ep_len
        return np.stack([f["pixels"][i // T, i % T] for i in ids], axis=0)

    def read_action_window(self, idx: int) -> np.ndarray:
        """Actions for window `idx` without decoding pixels."""
        f = self._open()
        entry = self._index[idx]
        if self._layout == "grouped":
            ek, start = entry
            idxs = [start + i * self.frameskip for i in range(self.num_steps)]
            return np.asarray(f[ek]["action"][idxs], dtype=np.float32)
        base = entry[0]
        if self._flat_shape == "5d":
            T = self._ep_len
            e, s = base // T, base % T
            idxs = [s + i * self.frameskip for i in range(self.num_steps)]
            return np.asarray(f["action"][e, idxs], dtype=np.float32)
        idxs = [base + i * self.frameskip for i in range(self.num_steps)]
        return np.asarray(f["action"][idxs], dtype=np.float32)

    def __getitem__(self, idx: int):
        f = self._open()
        pixels, action = self._read_window(f, self._index[idx])

        if pixels.dtype != np.uint8:
            pixels = (pixels * 255).clip(0, 255).astype(np.uint8) if pixels.max() <= 1 else pixels.astype(np.uint8)

        # to (T, 3, H, W) float in [0,1] then resize via simple bilinear if needed
        x = torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )

        action = torch.from_numpy(action)
        action = torch.nan_to_num(action, 0.0)

        return x, action
