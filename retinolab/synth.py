# does GAN data help the classifier? compared against the fair alternative: just re-balance the classes
from __future__ import annotations

import numpy as np
import torch

from .gan import sample


def plan_counts(labels: np.ndarray, n_classes: int = 5) -> list[int]:
    # how many synthetic images each class needs to reach the size of the biggest class
    counts = np.bincount(labels, minlength=n_classes)
    return [int(counts.max() - c) for c in counts]


def make_synthetic(gen, labels: np.ndarray, device, seed: int = 0):
    # returns (uint8 images (N,H,W,3), labels) to fill up the small classes
    need = plan_counts(labels)
    ys = np.concatenate([np.full(k, c, dtype=np.int64) for c, k in enumerate(need)])
    if len(ys) == 0:
        return np.zeros((0, 64, 64, 3), np.uint8), ys
    imgs = sample(gen, torch.from_numpy(ys), device, seed=seed)
    arr = (imgs.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    return arr, ys


def add_synthetic(real: tuple, synth: tuple) -> tuple:
    return np.concatenate([real[0], synth[0]]), np.concatenate([real[1], synth[1]])
