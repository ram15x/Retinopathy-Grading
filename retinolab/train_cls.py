# train one classifier run and save everything needed to compare runs later
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import metrics as M
from .config import NUM_CLASSES
from .data import ArrayDataset, balanced_sampler, seed_everything
from .models import build_classifier


@dataclass
class ClsConfig:
    backbone: str = "convnext_tiny"
    head: str = "ce"  # "ce" or "ordinal"
    preproc: str = "none"  # only used for naming, the arrays are prepared before
    img_size: int = 224
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.05
    seed: int = 0
    balance: bool = False  # class-balanced sampling
    pretrained: bool = True
    save_weights: bool = False
    workers: int = 2


@torch.no_grad()
def predict_outputs(model, loader, device, head: str):
    # returns (score, probs). score is the expected grade (ce) or the regression value (ordinal)
    model.eval()
    scores, probs = [], []
    grades = torch.arange(NUM_CLASSES, dtype=torch.float32)
    for x, _ in loader:
        out = model(x.to(device, non_blocking=True)).float().cpu()
        if head == "ce":
            p = torch.softmax(out, 1)
            probs.append(p)
            scores.append(p @ grades)
        else:
            scores.append(out.squeeze(1))
    score = torch.cat(scores).numpy()
    p = torch.cat(probs).numpy() if probs else np.full((len(score), NUM_CLASSES), np.nan)
    return score, p


def _predict_labels(score, probs, head, thresholds=None):
    if head == "ce":
        return probs.argmax(1)
    return M.apply_thresholds(score, thresholds if thresholds is not None else (0.5, 1.5, 2.5, 3.5))


def run_one(cfg: ClsConfig, splits: dict, out_dir, device=None) -> dict:
    out_dir = Path(out_dir)
    done = out_dir / "metrics.json"
    if done.exists():  # resumable: skip runs that already finished
        return json.loads(done.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    (xtr, ytr), (xva, yva), (xte, yte) = splits["train"], splits["val"], splits["test"]
    sampler = balanced_sampler(ytr) if cfg.balance else None
    pin = device.type == "cuda"
    train_loader = DataLoader(ArrayDataset(xtr, ytr, True, cfg.img_size), batch_size=cfg.batch_size,
                              shuffle=sampler is None, sampler=sampler, num_workers=cfg.workers,
                              pin_memory=pin, drop_last=True)
    val_loader = DataLoader(ArrayDataset(xva, yva, False, cfg.img_size), batch_size=64,
                            num_workers=cfg.workers, pin_memory=pin)
    test_loader = DataLoader(ArrayDataset(xte, yte, False, cfg.img_size), batch_size=64,
                             num_workers=cfg.workers, pin_memory=pin)

    model = build_classifier(cfg.backbone, cfg.head, cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05) if cfg.head == "ce" else nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.2)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)

    best = {"val_qwk": -2.0}
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                out = model(x)
                loss = criterion(out, y) if cfg.head == "ce" else criterion(out.squeeze(1).float(), y.float())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        sv, pv = predict_outputs(model, val_loader, device, cfg.head)
        val_qwk = M.qwk(yva, _predict_labels(sv, pv, cfg.head))
        # the test set is only recorded here, the best epoch is picked on validation alone
        if val_qwk > best["val_qwk"]:
            st, pt = predict_outputs(model, test_loader, device, cfg.head)
            best = {"val_qwk": val_qwk, "epoch": epoch, "sv": sv, "pv": pv, "st": st, "pt": pt}
            if cfg.save_weights:
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, out_dir / "best.pt")

    thr = M.fit_thresholds(best["sv"], yva) if cfg.head == "ordinal" else None
    pred_te = _predict_labels(best["st"], best["pt"], cfg.head, thr)
    res = M.evaluate(yte, pred_te, best["st"])
    lo, hi = M.bootstrap_ci(yte, pred_te)
    res.update({"qwk_ci95": [lo, hi], "val_qwk": float(M.qwk(yva, _predict_labels(best["sv"], best["pv"], cfg.head, thr))),
                "best_epoch": best["epoch"], "seconds": round(time.time() - t0, 1),
                "thresholds": thr, "config": asdict(cfg)})
    np.savez_compressed(out_dir / "outputs.npz", y_val=yva, y_test=yte, score_val=best["sv"],
                        score_test=best["st"], probs_val=best["pv"], probs_test=best["pt"],
                        pred_test=pred_te)
    done.write_text(json.dumps(res, indent=2))
    return res
