# small class-conditional GAN for 64x64 fundus images, with switches so each trick can be ablated
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

Z_DIM = 128
N_CLASSES = 5


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    # DiffAugment (Zhao et al. 2020): color, translation and cutout, all differentiable.
    # it is applied to real AND fake images before the discriminator sees them
    b, _, h, w = x.shape
    dev = x.device
    # color
    x = x + (torch.rand(b, 1, 1, 1, device=dev) - 0.5)
    m = x.mean(dim=1, keepdim=True)
    x = (x - m) * (torch.rand(b, 1, 1, 1, device=dev) * 2) + m
    m = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - m) * (torch.rand(b, 1, 1, 1, device=dev) + 0.5) + m
    # translation (up to 1/8 of the image, zero padded)
    shift = int(h * 0.125 + 0.5)
    tx = torch.randint(-shift, shift + 1, (b, 1, 1), device=dev)
    ty = torch.randint(-shift, shift + 1, (b, 1, 1), device=dev)
    gb, gx, gy = torch.meshgrid(torch.arange(b, device=dev), torch.arange(h, device=dev),
                                torch.arange(w, device=dev), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    x = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0]).permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)
    # cutout (a square of half the image size)
    cut = int(h * 0.5 + 0.5)
    ox = torch.randint(0, h + (1 - cut % 2), (b, 1, 1), device=dev)
    oy = torch.randint(0, w + (1 - cut % 2), (b, 1, 1), device=dev)
    gb, gx, gy = torch.meshgrid(torch.arange(b, device=dev), torch.arange(cut, device=dev),
                                torch.arange(cut, device=dev), indexing="ij")
    gx = torch.clamp(gx + ox - cut // 2, 0, h - 1)
    gy = torch.clamp(gy + oy - cut // 2, 0, w - 1)
    mask = torch.ones(b, h, w, device=dev)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


class Generator(nn.Module):
    def __init__(self, z_dim: int = Z_DIM, n_classes: int = N_CLASSES, ch: int = 64):
        super().__init__()
        self.embed = nn.Embedding(n_classes, z_dim)
        self.fc = nn.Linear(2 * z_dim, 4 * 4 * ch * 8)
        self.ch = ch
        chans = [ch * 8, ch * 4, ch * 2, ch, ch // 2]  # 4 -> 8 -> 16 -> 32 -> 64
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                       nn.BatchNorm2d(cout), nn.ReLU(True)]
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv2d(chans[-1], 3, 3, padding=1)

    def forward(self, z, y):
        h = self.fc(torch.cat([z, self.embed(y)], dim=1)).view(-1, self.ch * 8, 4, 4)
        return torch.tanh(self.out(self.blocks(h)))


class Discriminator(nn.Module):
    # projection discriminator: the class label is mixed in through an embedding
    def __init__(self, n_classes: int = N_CLASSES, ch: int = 64, use_sn: bool = True, dropout: float = 0.0):
        super().__init__()
        wrap = spectral_norm if use_sn else (lambda m: m)
        chans = [3, ch, ch * 2, ch * 4, ch * 8]  # 64 -> 32 -> 16 -> 8 -> 4
        layers = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            layers += [wrap(nn.Conv2d(cin, cout, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
            if dropout:
                layers.append(nn.Dropout(dropout))
        self.features = nn.Sequential(*layers)
        self.linear = wrap(nn.Linear(ch * 8, 1))
        self.embed = wrap(nn.Embedding(n_classes, ch * 8))

    def forward(self, x, y):
        h = self.features(x).sum(dim=[2, 3])
        return self.linear(h) + (self.embed(y) * h).sum(1, keepdim=True)


@dataclass
class GanConfig:
    name: str = "full"
    loss: str = "hinge"  # "bce" (original) or "hinge"
    spectral_norm: bool = True
    augment: bool = True
    ema: bool = True
    epochs: int = 600
    batch_size: int = 64
    lr: float = 2e-4
    betas: tuple = (0.5, 0.999)
    dropout: float = 0.0
    seed: int = 0


# the four recipes compared in the ablation. A is the original notebook's recipe
# (bce loss, no normalisation tricks, dropout in D, no augmentation, no ema)
RECIPES = {
    "A_original": GanConfig("A_original", "bce", False, False, False, lr=1e-4, betas=(0.9, 0.999), dropout=0.3),
    "B_hinge_sn": GanConfig("B_hinge_sn", "hinge", True, False, False),
    "C_diffaug": GanConfig("C_diffaug", "hinge", True, True, False),
    "D_full": GanConfig("D_full", "hinge", True, True, True),
}


def d_loss(kind, real, fake):
    if kind == "bce":
        return (F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
                + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)))
    return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()


def g_loss(kind, fake):
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))
    return -fake.mean()


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1.0 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def to_tensor_pm1(imgs: np.ndarray, device) -> torch.Tensor:
    # uint8 (N,H,W,3) -> float (N,3,H,W) in [-1,1]
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return x.to(device)


@torch.no_grad()
def sample(gen: nn.Module, labels: torch.Tensor, device, batch: int = 250, seed: int | None = None) -> torch.Tensor:
    # returns float images in [0,1] (N,3,H,W) on cpu
    gen.eval()
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    out = []
    for i in range(0, len(labels), batch):
        y = labels[i:i + batch].to(device)
        z = torch.randn(len(y), Z_DIM, generator=g).to(device)
        out.append(((gen(z, y) + 1) / 2).clamp(0, 1).cpu())
    return torch.cat(out)


def train_gan(cfg: GanConfig, imgs: np.ndarray, labels: np.ndarray, out_dir, device=None,
              on_checkpoint=None, checkpoint_every: int = 0) -> dict:
    # imgs: uint8 (N,64,64,3). saves G (and the ema copy) to out_dir. on_checkpoint(gen, epoch) is optional
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    x_all = to_tensor_pm1(imgs, device)
    y_all = torch.from_numpy(labels).long().to(device)
    n = len(x_all)
    G = Generator().to(device)
    D = Discriminator(use_sn=cfg.spectral_norm, dropout=cfg.dropout).to(device)
    G_ema = copy.deepcopy(G).eval() if cfg.ema else None
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=cfg.betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=cfg.betas)
    aug = diff_augment if cfg.augment else (lambda t: t)

    hist, t0, it = [], time.time(), 0
    steps_per_epoch = max(n // cfg.batch_size, 1)
    for epoch in range(1, cfg.epochs + 1):
        G.train()  # the checkpoint hook may have switched it to eval
        perm = torch.randperm(n, device=device)
        for s in range(steps_per_epoch):
            idx = perm[s * cfg.batch_size:(s + 1) * cfg.batch_size]
            real, yr = x_all[idx], y_all[idx]
            yf = y_all[torch.randint(0, n, (len(idx),), device=device)]  # fake labels follow the data
            z = torch.randn(len(idx), Z_DIM, device=device)
            # discriminator
            fake = G(z, yf).detach()
            ld = d_loss(cfg.loss, D(aug(real), yr), D(aug(fake), yf))
            opt_d.zero_grad(set_to_none=True)
            ld.backward()
            opt_d.step()
            # generator
            z = torch.randn(len(idx), Z_DIM, device=device)
            lg = g_loss(cfg.loss, D(aug(G(z, yf)), yf))
            opt_g.zero_grad(set_to_none=True)
            lg.backward()
            opt_g.step()
            if G_ema is not None:
                ema_update(G_ema, G, 0.999)
            it += 1
        if epoch % 25 == 0 or epoch == cfg.epochs:
            hist.append({"epoch": epoch, "d_loss": float(ld.detach()), "g_loss": float(lg.detach()), "seconds": round(time.time() - t0, 1)})
        if checkpoint_every and epoch % checkpoint_every == 0 and on_checkpoint:
            on_checkpoint(G_ema if G_ema is not None else G, epoch)

    final = G_ema if G_ema is not None else G
    torch.save(G.state_dict(), out_dir / "G.pt")
    torch.save(final.state_dict(), out_dir / "G_final.pt")
    (out_dir / "train_log.json").write_text(json.dumps({"config": asdict(cfg), "history": hist, "iterations": it}, indent=2))
    return {"G": final, "history": hist}


def load_generator(path, device) -> Generator:
    g = Generator().to(device)
    g.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return g.eval()
