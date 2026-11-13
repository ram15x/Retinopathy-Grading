# turns the finished runs into tables (markdown + json) and figures
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import aggregate as A
from .config import GRADES

NAMES = {"resnet50": "ResNet50", "efficientnet_v2_s": "EfficientNetV2-S", "convnext_tiny": "ConvNeXt-T",
         "maxvit_t": "MaxViT-T (hybrid)"}


def runs_in(group_dir) -> list[dict]:
    return [A.load_run(d) for d in sorted(Path(group_dir).glob("seed*")) if (d / "metrics.json").exists()]


def fmt(ms: dict, k: str) -> str:
    m, s = ms[k]["mean"], ms[k]["std"]
    return f"{m:.3f} +/- {s:.3f}" if not np.isnan(s) else f"{m:.3f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def collect(out: Path) -> dict:
    res = {"E1": {}, "E2": {}, "E3": {}, "E4": {}, "E5": {}, "E6": {}}
    for d in sorted((out / "E1_backbones").glob("*")):
        r = runs_in(d)
        if r:
            res["E1"][d.name] = {**A.mean_std(r), "seconds": float(np.mean([x["seconds"] for x in r]))}
    ens_file = out / "E2_ensembles" / "summary.json"
    if ens_file.exists():
        res["E2"] = json.loads(ens_file.read_text())
    for tag, folder in (("E3", "E3_head"), ("E4", "E4_preproc")):
        for d in sorted((out / folder).glob("*")):
            r = runs_in(d)
            if r:
                res[tag][d.name] = A.mean_std(r)
    g = out / "E5_gan" / "summary.json"
    if g.exists():
        res["E5"] = json.loads(g.read_text())
    for d in sorted((out / "E6_synth").glob("*")):
        r = runs_in(d)
        if r:
            res["E6"][d.name] = A.mean_std(r)
    return res


def bar_with_err(ax, labels, means, stds, title, color="#3b6ea5"):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[0 if np.isnan(s) else s for s in stds], capsize=3, color=color)
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    lo = min(means) - 0.15
    ax.set_ylim(max(0, lo), min(1.0, max(means) + 0.08))
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.005, f"{m:.3f}", ha="center", fontsize=7)


def make_figures(out: Path, docs: Path, res: dict) -> None:
    docs.mkdir(parents=True, exist_ok=True)
    rows = {NAMES.get(k, k): v for k, v in res["E1"].items()}
    rows.update({k: v for k, v in res["E2"].items()})
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, key, title in zip(axes, ("qwk", "referable_auc"), ("quadratic kappa (test)", "referable DR AUC (test)")):
            labs = list(rows)
            bar_with_err(ax, labs, [rows[k][key]["mean"] for k in labs], [rows[k][key]["std"] for k in labs], title)
        fig.tight_layout()
        fig.savefig(docs / "backbones.png", dpi=150)
        plt.close(fig)
    # confusion matrix of the best single model, summed over its seeds
    if res["E1"]:
        best = max(res["E1"], key=lambda k: res["E1"][k]["qwk"]["mean"])
        cms = [np.array(r["confusion_matrix"]) for r in runs_in(out / "E1_backbones" / best)]
        cm = np.sum(cms, axis=0)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5), [g.replace("Proliferative", "Prolif.") for g in GRADES], rotation=30, fontsize=8)
        ax.set_yticks(range(5), GRADES, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_title(f"{NAMES.get(best, best)}, test, summed over {len(cms)} seeds", fontsize=9)
        fig.tight_layout()
        fig.savefig(docs / "confusion_matrix.png", dpi=150)
        plt.close(fig)
    if res["E5"]:
        names = list(res["E5"])
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        bar_with_err(ax, names, [res["E5"][n]["fid"] for n in names], [0] * len(names), "FID (lower is better)", "#a5543b")
        ax.set_ylim(0, max(res["E5"][n]["fid"] for n in names) * 1.15)
        fig.tight_layout()
        fig.savefig(docs / "gan_fid.png", dpi=150)
        plt.close(fig)
    if res["E6"]:
        arms = list(res["E6"])
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, key, title in zip(axes, ("qwk", "macro_f1"), ("quadratic kappa", "macro F1")):
            bar_with_err(ax, arms, [res["E6"][a][key]["mean"] for a in arms], [res["E6"][a][key]["std"] for a in arms], title, "#5b8c5a")
        fig.tight_layout()
        fig.savefig(docs / "synthetic_augmentation.png", dpi=150)
        plt.close(fig)


def write_markdown(res: dict, path: Path) -> None:
    parts = []
    if res["E1"] or res["E2"]:
        rows = []
        for k, v in res["E1"].items():
            rows.append([NAMES.get(k, k), fmt(v, "qwk"), fmt(v, "accuracy"), fmt(v, "macro_f1"), fmt(v, "referable_auc"),
                         f"{v['seconds']:.0f}s", str(v["n_runs"])])
        for k, v in res["E2"].items():
            rows.append([k, fmt(v, "qwk"), fmt(v, "accuracy"), fmt(v, "macro_f1"), fmt(v, "referable_auc"), "-", str(v["n_runs"])])
        parts.append("### Models and ensembles (test set)\n\n" + md_table(
            ["model", "kappa", "accuracy", "macro F1", "referable AUC", "train time", "seeds"], rows))
    for tag, title in (("E3", "Head: class logits vs ordinal regression"), ("E4", "Preprocessing")):
        if res[tag]:
            rows = [[k, fmt(v, "qwk"), fmt(v, "accuracy"), fmt(v, "macro_f1"), fmt(v, "referable_auc"), str(v["n_runs"])]
                    for k, v in res[tag].items()]
            parts.append(f"### {title}\n\n" + md_table(["variant", "kappa", "accuracy", "macro F1", "referable AUC", "seeds"], rows))
    if res["E5"]:
        rows = [[n, f"{v['fid']:.1f}", ", ".join(f"{k}: {x:.1f}" for k, x in v["fid_mid"].items()),
                 f"{v['nn_gen_to_train']:.2f}", f"{v['nn_heldout_to_train']:.2f}"] for n, v in res["E5"].items()]
        parts.append("### GAN recipes\n\n" + md_table(
            ["recipe", "FID", "FID at midpoint", "nearest-train dist (generated)", "nearest-train dist (unseen real)"], rows))
    if res["E6"]:
        rows = [[k, fmt(v, "qwk"), fmt(v, "accuracy"), fmt(v, "macro_f1"),
                 " / ".join(f"{x:.2f}" for x in v["per_class_recall"]), str(v["n_runs"])] for k, v in res["E6"].items()]
        parts.append("### Does GAN data help (64px, ResNet18)\n\n" + md_table(
            ["training data", "kappa", "accuracy", "macro F1", "recall per grade 0-4", "seeds"], rows))
    path.write_text("\n\n".join(parts) + "\n")


def build(out: str, docs: str, results_dir: str) -> dict:
    out_p, docs_p, res_p = Path(out), Path(docs), Path(results_dir)
    res_p.mkdir(parents=True, exist_ok=True)
    res = collect(out_p)
    (res_p / "summary.json").write_text(json.dumps(res, indent=2))
    write_markdown(res, res_p / "RESULTS.md")
    make_figures(out_p, docs_p, res)
    return res
