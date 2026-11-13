# redraws the three summary charts from the tables in results/RESULTS.md
# (the full run also draws its own figures on the GPU box, this one only needs the markdown)
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
text = (root / "results" / "RESULTS.md").read_text()


def table(title):
    block = text.split(f"### {title}")[1].split("###")[0]
    rows = [r.strip().strip("|").split("|") for r in block.strip().splitlines()[2:]]
    return [[c.strip() for c in r] for r in rows]


def pm(cell):
    a, b = cell.split("+/-")
    return float(a), float(b)


def bars(ax, labels, vals, errs, title, ylabel, colors=None, ylim=None):
    ax.bar(range(len(labels)), vals, yerr=errs, capsize=4, color=colors or "#3b6ea5")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    for i, v in enumerate(vals):
        ax.text(i, v + (errs[i] if errs else 0) + 0.005, f"{v:.3f}", ha="center", fontsize=7)


# models and ensembles
rows = table("Models and ensembles (test set)")
labels = [r[0].replace("ensemble: ", "ens: ") for r in rows]
k = [pm(r[1]) for r in rows]
cols = ["#8aa4c8"] * 4 + ["#c98b4b"] * 3
fig, ax = plt.subplots(figsize=(8, 4))
bars(ax, labels, [a for a, _ in k], [b for _, b in k], "Quadratic weighted kappa on the 400 test images (mean +/- std, 3 seeds)", "kappa", cols, (0.6, 0.9))
fig.tight_layout()
fig.savefig(root / "docs" / "backbones.png", dpi=140)

# gan fid
rows = table("GAN recipes")
fig, ax = plt.subplots(figsize=(6, 3.6))
bars(ax, [r[0] for r in rows], [float(r[1]) for r in rows], None, "FID after 600 epochs (one seed per recipe, lower is better)", "FID", ylim=(0, 190))
fig.tight_layout()
fig.savefig(root / "docs" / "gan_fid.png", dpi=140)

# synthetic augmentation
rows = table("Does GAN data help (64px, ResNet18)")
k = [pm(r[1]) for r in rows]
fig, ax = plt.subplots(figsize=(6, 3.6))
bars(ax, [r[0] for r in rows], [a for a, _ in k], [b for _, b in k], "ResNet18 at 64px, kappa (mean +/- std, 5 seeds)", "kappa", ylim=(0.45, 0.7))
fig.tight_layout()
fig.savefig(root / "docs" / "synthetic_augmentation.png", dpi=140)
print("ok")
