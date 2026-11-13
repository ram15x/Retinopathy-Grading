# data loading for RetinaMNIST (fundus photos, 5 DR grades, from the DeepDRiD data)
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import IMAGENET_MEAN, IMAGENET_STD


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_retinamnist(size: int = 224, root: str = "data") -> dict:
    # returns {"train": (imgs, labels), "val": ..., "test": ...}, imgs are uint8 (N,H,W,3)
    from medmnist import RetinaMNIST

    os.makedirs(root, exist_ok=True)
    out = {}
    for split in ("train", "val", "test"):
        ds = RetinaMNIST(split=split, download=True, size=size, root=root)
        out[split] = (np.asarray(ds.imgs), np.asarray(ds.labels).reshape(-1).astype(np.int64))
    return out


def ben_graham(img: np.ndarray) -> np.ndarray:
    # local contrast normalisation often used for fundus photos: subtract a heavy blur
    sigma = max(img.shape[0] / 30.0, 1.0)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def resize_all(imgs: np.ndarray, size: int) -> np.ndarray:
    if imgs.shape[1] == size:
        return imgs
    return np.stack([cv2.resize(i, (size, size), interpolation=cv2.INTER_AREA) for i in imgs])


def prepare_splits(raw: dict, size: int, preproc: str = "none") -> dict:
    # resize and (optionally) preprocess every split once, so training stays fast
    out = {}
    for name, (imgs, labels) in raw.items():
        imgs = resize_all(imgs, size)
        if preproc == "ben":
            imgs = np.stack([ben_graham(i) for i in imgs])
        elif preproc != "none":
            raise ValueError(f"unknown preproc {preproc!r}")
        out[name] = (imgs, labels)
    return out


def train_tf(size: int):
    # fundus photos have no fixed up/down or left/right, so flips and rotations are fine
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_tf(size: int):
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


class ArrayDataset(Dataset):
    def __init__(self, imgs: np.ndarray, labels: np.ndarray, train: bool, size: int):
        self.imgs, self.labels = imgs, labels
        self.tf = train_tf(size) if train else eval_tf(size)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.imgs[i])), int(self.labels[i])


def balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    # draw every class about equally often
    counts = np.bincount(labels, minlength=5).astype(float)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(weights.tolist(), num_samples=len(labels), replacement=True)
