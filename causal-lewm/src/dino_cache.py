"""Precompute frozen-DINOv2 patch tokens to disk and serve them.

The DINOv2 backbone is frozen, so its patch tokens for a given frame are
deterministic — re-running the ViT every step is the dominant cost. We encode
each *referenced* frame once into a float16 HDF5 cache, then train the slot
attention + predictor (and the trainable proj/norm) on the cache. Model
behavior is identical; only the backbone forward is amortized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .data import PushTHDF5


def _frames_to_input(pixels: np.ndarray, image_size: int) -> torch.Tensor:
    """uint8/float (B, H, W, C) -> (B, 3, H, W) float in [0,1], resized to image_size."""
    if pixels.dtype != np.uint8:
        pixels = (pixels * 255).clip(0, 255).astype(np.uint8) if pixels.max() <= 1 else pixels.astype(np.uint8)
    x = torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != image_size or x.shape[-2] != image_size:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x


def default_cache_path(base: PushTHDF5, image_size: int) -> Path:
    p = base.path
    return p.with_name(p.name + f".dino-{image_size}.h5")


def build_dino_cache(
    base: PushTHDF5,
    encoder,
    device: str,
    cache_path: str | Path,
    image_size: int,
    batch_size: int = 64,
) -> Path:
    """Encode every frame referenced by `base`'s window index into `cache_path`.

    Stores datasets `tokens` (num_unique, P, hidden) float16 and `frame_ids`
    (num_unique,) int64. Reuses the file if it already exists.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"[dino_cache] reusing existing cache -> {cache_path}", flush=True)
        return cache_path

    ids = set()
    for i in range(len(base)):
        ids.update(base.get_frame_ids(i))
    ids = np.array(sorted(ids), dtype=np.int64)
    n = len(ids)

    hidden = encoder.hidden
    # Probe one frame to learn the patch-token count P.
    probe = _frames_to_input(base.read_frames_by_id(ids[:1].tolist()), image_size).to(device)
    with torch.no_grad():
        P = encoder.backbone_tokens(probe).shape[1]

    est_gb = n * P * hidden * 2 / 1e9
    print(
        f"[dino_cache] encoding {n} unique frames -> {cache_path} "
        f"(tokens shape ({n}, {P}, {hidden}) float16, ~{est_gb:.1f} GB)",
        flush=True,
    )

    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    was_training = encoder.training
    encoder.eval()
    try:
        with h5py.File(tmp, "w") as out:
            tok_ds = out.create_dataset(
                "tokens", shape=(n, P, hidden), dtype="float16", chunks=(1, P, hidden)
            )
            out.create_dataset("frame_ids", data=ids)
            for start in range(0, n, batch_size):
                batch_ids = ids[start : start + batch_size].tolist()
                frames = _frames_to_input(base.read_frames_by_id(batch_ids), image_size).to(device)
                with torch.no_grad():
                    toks = encoder.backbone_tokens(frames)
                tok_ds[start : start + len(batch_ids)] = toks.to(torch.float16).cpu().numpy()
                if start % (batch_size * 20) == 0:
                    print(f"[dino_cache]   {start}/{n}", flush=True)
        tmp.replace(cache_path)
    finally:
        if was_training:
            encoder.train()
        tmp.unlink(missing_ok=True)
    print(f"[dino_cache] done -> {cache_path}", flush=True)
    return cache_path


class CachedDinoDataset(Dataset):
    """Yields (tokens[T, P, hidden] float32, actions[T, A] float32) from a cache.

    Window indexing and actions come from the wrapped `PushTHDF5`; pixels are
    never decoded — token rows are looked up by global frame id.
    """

    def __init__(self, base: PushTHDF5, cache_path: str | Path):
        self.base = base
        self.cache_path = Path(cache_path)
        with h5py.File(self.cache_path, "r") as f:
            self._frame_ids = f["frame_ids"][:]
        self._cache: h5py.File | None = None

    def __len__(self):
        return len(self.base)

    def _open(self) -> h5py.File:
        if self._cache is None:
            self._cache = h5py.File(self.cache_path, "r", swmr=True)
        return self._cache

    def __getitem__(self, idx: int):
        ids = self.base.get_frame_ids(idx)
        rows = np.searchsorted(self._frame_ids, ids)
        if not np.array_equal(self._frame_ids[rows], ids):
            missing = [i for i, r in zip(ids, rows) if r >= len(self._frame_ids) or self._frame_ids[r] != i]
            raise KeyError(f"frame ids {missing} absent from cache {self.cache_path}")
        cache = self._open()
        # h5 fancy indexing needs increasing order; sort then restore.
        order = np.argsort(rows)
        toks_sorted = cache["tokens"][rows[order].tolist()]
        toks = np.empty_like(toks_sorted)
        toks[order] = toks_sorted
        x = torch.from_numpy(toks).float()

        action = torch.from_numpy(self.base.read_action_window(idx))
        action = torch.nan_to_num(action, 0.0)
        return x, action
