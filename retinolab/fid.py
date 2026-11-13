# FID and a memorisation check. features come from torchvision's InceptionV3 (imagenet weights),
# so the numbers are only good for comparing my own models with each other, not with papers
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import IMAGENET_MEAN, IMAGENET_STD


def fid_from_features(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    # trace of sqrt(s1 @ s2) through eigenvalues of r1 @ s2 @ r1, much faster than scipy sqrtm at 2048 dims
    w1, v1 = np.linalg.eigh(s1)
    r1 = (v1 * np.sqrt(np.clip(w1, 0, None))) @ v1.T
    m = r1 @ s2 @ r1
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2.0 * tr_sqrt)


def nn_distance(queries: np.ndarray, refs: np.ndarray) -> float:
    # mean distance from each query to its closest reference in feature space
    q, r = torch.from_numpy(queries).float(), torch.from_numpy(refs).float()
    return float(torch.cdist(q, r).min(dim=1).values.mean())


def load_inception(device):
    from torchvision import models

    m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    return m.eval().to(device)


@torch.no_grad()
def inception_features(model, imgs01: torch.Tensor, device, batch: int = 100) -> np.ndarray:
    # imgs01: float tensor (N,3,H,W) with values in [0,1]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    feats = []
    for i in range(0, len(imgs01), batch):
        x = imgs01[i:i + batch].to(device).float()
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        feats.append(model((x - mean) / std).float().cpu())
    return torch.cat(feats).numpy()
