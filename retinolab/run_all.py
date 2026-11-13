# runs every experiment, resumable (finished runs are skipped).
# python -m retinolab.run_all --stage all --out runs
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import aggregate as A
from . import report
from .data import load_retinamnist, prepare_splits, resize_all
from .explain import gradcam_grid
from .fid import fid_from_features, inception_features, load_inception, nn_distance
from .gan import RECIPES, load_generator, sample, train_gan
from .synth import add_synthetic, make_synthetic
from .train_cls import ClsConfig, run_one

BACKBONES = ["resnet50", "efficientnet_v2_s", "convnext_tiny", "maxvit_t"]
ENSEMBLES = {
    "ensemble: all four": BACKBONES,
    "ensemble: ConvNeXt-T + MaxViT-T (CNN + hybrid)": ["convnext_tiny", "maxvit_t"],
    "ensemble: ResNet50 + EfficientNetV2-S (CNNs)": ["resnet50", "efficientnet_v2_s"],
}


def brief(tag, r):
    print(f"[{tag}] kappa={r['qwk']:.3f} acc={r['accuracy']:.3f} f1={r['macro_f1']:.3f} "
          f"auc={r['referable_auc']:.3f} best_epoch={r['best_epoch']} {r['seconds']:.0f}s", flush=True)


def best_backbone(out: Path, backbones) -> str:
    scores = {}
    for b in backbones:
        runs = report.runs_in(out / "E1_backbones" / b)
        if runs:
            scores[b] = A.mean_std(runs)["qwk"]["mean"]
    return max(scores, key=scores.get)


def stage_cls(a, raw, out):
    splits = prepare_splits(raw, 224, "none")
    for b in a.backbones:
        for s in a.seeds:
            cfg = ClsConfig(backbone=b, seed=s, epochs=a.epochs, save_weights=(s == 0))
            brief(f"E1 {b} seed{s}", run_one(cfg, splits, out / "E1_backbones" / b / f"seed{s}"))


def stage_ens(a, raw, out):
    per = {name: [] for name in ENSEMBLES}
    for s in a.seeds:
        for name, members in ENSEMBLES.items():
            dirs = [out / "E1_backbones" / m / f"seed{s}" for m in members]
            if all((d / "outputs.npz").exists() for d in dirs):
                per[name].append(A.ensemble_from_dirs(dirs))
    summary = {n: A.mean_std(r) for n, r in per.items() if r}
    (out / "E2_ensembles").mkdir(exist_ok=True)
    (out / "E2_ensembles" / "summary.json").write_text(json.dumps(summary, indent=2))
    for n, v in summary.items():
        print(f"[E2 {n}] kappa={v['qwk']['mean']:.3f}", flush=True)


def stage_abl(a, raw, out):
    b = best_backbone(out, a.backbones)
    print(f"ablations on best backbone: {b}", flush=True)
    base = prepare_splits(raw, 224, "none")
    ben = prepare_splits(raw, 224, "ben")
    for s in a.seeds:
        cfg = ClsConfig(backbone=b, head="ordinal", seed=s, epochs=a.epochs)
        brief(f"E3 ordinal seed{s}", run_one(cfg, base, out / "E3_head" / "ordinal" / f"seed{s}"))
        cfg = ClsConfig(backbone=b, preproc="ben", seed=s, epochs=a.epochs)
        brief(f"E4 ben seed{s}", run_one(cfg, ben, out / "E4_preproc" / "ben" / f"seed{s}"))
    # reference rows reuse the E1 runs of the same backbone
    for folder, name in (("E3_head", "class_logits"), ("E4_preproc", "none")):
        dst = out / folder / name
        for s in a.seeds:
            src = out / "E1_backbones" / b / f"seed{s}"
            d = dst / f"seed{s}"
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "metrics.json").exists():
                (d / "metrics.json").write_text((src / "metrics.json").read_text())
                (d / "outputs.npz").write_bytes((src / "outputs.npz").read_bytes())


def grid_png(imgs01: torch.Tensor, labels: torch.Tensor, path, per_row: int = 8):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, per_row, figsize=(per_row * 0.9, 5 * 0.95))
    for g in range(5):
        idx = (labels == g).nonzero().flatten()[:per_row]
        for c in range(per_row):
            ax = axes[g, c]
            ax.axis("off")
            if c < len(idx):
                ax.imshow(imgs01[idx[c]].permute(1, 2, 0).numpy())
        axes[g, 0].set_title(f"grade {g}", fontsize=6, loc="left")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_fid_hook(inc, f_real, tr_y, device, store):
    # called halfway through training to log a FID value (1000 samples)
    rng = np.random.default_rng(0)

    def hook(gen, epoch):
        lab = torch.from_numpy(rng.choice(tr_y, 1000))
        f = inception_features(inc, sample(gen, lab, device, seed=1), device)
        store[str(epoch)] = fid_from_features(f, f_real)

    return hook


def stage_gan(a, raw, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gdir = out / "E5_gan"
    gdir.mkdir(parents=True, exist_ok=True)
    tr_imgs, tr_y = resize_all(raw["train"][0], 64), raw["train"][1]
    held = np.concatenate([resize_all(raw["val"][0], 64), resize_all(raw["test"][0], 64)])
    inc = load_inception(device)
    to01 = lambda arr: torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255  # noqa: E731
    f_real = inception_features(inc, to01(tr_imgs), device)
    nn_held = nn_distance(inception_features(inc, to01(held), device), f_real)
    lab_grid = torch.arange(5).repeat_interleave(8)
    results = {}
    for name, cfg in RECIPES.items():
        cfg.epochs = a.gan_epochs
        d = gdir / name
        if (d / "final.json").exists():
            results[name] = json.loads((d / "final.json").read_text())
            continue
        mid = {}
        hook = make_fid_hook(inc, f_real, tr_y, device, mid)

        train_gan(cfg, tr_imgs, tr_y, d, device, on_checkpoint=hook, checkpoint_every=max(a.gan_epochs // 2, 1))
        gen = load_generator(d / "G_final.pt", device)
        lab = torch.from_numpy(np.random.default_rng(0).choice(tr_y, 2000))
        f_gen = inception_features(inc, sample(gen, lab, device, seed=123), device)
        res = {"fid": fid_from_features(f_gen, f_real), "fid_mid": mid,
               "nn_gen_to_train": nn_distance(f_gen, f_real), "nn_heldout_to_train": nn_held}
        grid_png(sample(gen, lab_grid.repeat(1), device, seed=7), lab_grid, d / "samples.png")
        (d / "final.json").write_text(json.dumps(res, indent=2))
        results[name] = res
        print(f"[E5 {name}] FID={res['fid']:.1f} mid={mid}", flush=True)
    real_grid_idx = torch.cat([(torch.from_numpy(tr_y) == g).nonzero().flatten()[:8] for g in range(5)])
    grid_png(to01(tr_imgs)[real_grid_idx], torch.from_numpy(tr_y)[real_grid_idx], gdir / "real_samples.png")
    (gdir / "summary.json").write_text(json.dumps(results, indent=2))


def stage_synth(a, raw, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = json.loads((out / "E5_gan" / "summary.json").read_text())
    best = min(summary, key=lambda k: summary[k]["fid"])
    print(f"using the {best} generator for synthetic images", flush=True)
    gen = load_generator(out / "E5_gan" / best / "G_final.pt", device)
    splits = prepare_splits(raw, 64, "none")
    synth = make_synthetic(gen, splits["train"][1], device, seed=0)
    print("synthetic images added per grade:", np.bincount(synth[1], minlength=5).tolist(), flush=True)
    arms = {
        "real only": (splits["train"], False),
        "real, class-balanced sampling": (splits["train"], True),
        "real + GAN images": (add_synthetic(splits["train"], synth), False),
    }
    for name, (train, balance) in arms.items():
        for s in a.seeds_synth:
            sp = {"train": train, "val": splits["val"], "test": splits["test"]}
            cfg = ClsConfig(backbone="resnet18", img_size=64, epochs=a.synth_epochs, lr=3e-4, seed=s, balance=balance)
            brief(f"E6 {name} seed{s}", run_one(cfg, sp, out / "E6_synth" / name / f"seed{s}"))


def stage_explain(a, raw, out, docs):
    b = best_backbone(out, a.backbones)
    ck = out / "E1_backbones" / b / "seed0" / "best.pt"
    docs.mkdir(parents=True, exist_ok=True)
    imgs, labels = raw["test"]
    gradcam_grid(ck, imgs, labels, docs / "gradcam.png")
    print("saved gradcam for", b, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "cls", "ens", "abl", "gan", "synth", "explain", "report"])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--seeds-synth", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--gan-epochs", type=int, default=600)
    ap.add_argument("--synth-epochs", type=int, default=30)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = ["cls", "ens", "abl", "gan", "synth", "explain", "report"] if a.stage == "all" else [a.stage]
    raw = None if stages == ["report"] else load_retinamnist(224, a.data)
    if raw is not None:
        print({k: np.bincount(v[1], minlength=5).tolist() for k, v in raw.items()}, flush=True)
    for st in stages:
        print(f"=== stage {st} ===", flush=True)
        if st == "cls":
            stage_cls(a, raw, out)
        elif st == "ens":
            stage_ens(a, raw, out)
        elif st == "abl":
            stage_abl(a, raw, out)
        elif st == "gan":
            stage_gan(a, raw, out)
        elif st == "synth":
            stage_synth(a, raw, out)
        elif st == "explain":
            stage_explain(a, raw, out, Path(a.docs))
        elif st == "report":
            report.build(a.out, a.docs, a.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
