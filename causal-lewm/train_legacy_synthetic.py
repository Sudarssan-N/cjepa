"""Synthetic moving-disks dataset, kept around for smoke tests."""

import torch
from torch.utils.data import Dataset


class SyntheticVideo(Dataset):
    def __init__(self, num_samples=4096, num_frames=8, image_size=64, num_objects=3, seed=0):
        self.num_samples = num_samples
        self.T = num_frames
        self.H = image_size
        self.num_objects = num_objects

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T, H, K = self.T, self.H, self.num_objects
        g = torch.Generator().manual_seed(idx)
        pos = torch.rand(K, 2, generator=g) * (H - 16) + 8
        vel = (torch.rand(K, 2, generator=g) - 0.5) * 4.0
        colors = torch.rand(K, 3, generator=g) * 0.7 + 0.3
        radius = 5

        frames = torch.zeros(T, 3, H, H)
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij")
        for t in range(T):
            for k in range(K):
                cx, cy = pos[k]
                disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
                for c in range(3):
                    frames[t, c][disk] = colors[k, c]
            pos = pos + vel
            vel = torch.where((pos < 4) | (pos > H - 4), -vel, vel)

        actions = vel.new_zeros(T, 2)
        actions[:] = vel[0]
        return frames, actions
