# combine finished runs: mean/std over seeds, and ensembles built from saved outputs
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import metrics as M
from .config import NUM_CLASSES

KEYS = ("qwk", "accuracy", "macro_f1", "referable_auc")


def load_run(run_dir) -> dict:
    return json.loads((Path(run_dir) / "metrics.json").read_text())


def mean_std(results: list[dict]) -> dict:
    # std is the sample std over seeds (ddof=1), nan when there is only one run
    out = {"n_runs": len(results)}
    for k in KEYS:
        v = np.array([r[k] for r in results], dtype=float)
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan")}
    recall = np.array([r["per_class_recall"] for r in results], dtype=float)
    out["per_class_recall"] = np.nanmean(recall, axis=0).tolist()
    return out


def ensemble_from_dirs(run_dirs) -> dict:
    # average class probabilities of several runs (same test set), then score as usual
    outs = [np.load(Path(d) / "outputs.npz") for d in run_dirs]
    y = outs[0]["y_test"]
    for o in outs[1:]:
        if not np.array_equal(o["y_test"], y):
            raise ValueError("runs were evaluated on different test sets")
    probs = np.mean([o["probs_test"] for o in outs], axis=0)
    if np.isnan(probs).any():
        raise ValueError("ensembles need probability outputs (head 'ce')")
    pred = probs.argmax(1)
    score = probs @ np.arange(NUM_CLASSES)
    res = M.evaluate(y, pred, score)
    res["qwk_ci95"] = list(M.bootstrap_ci(y, pred))
    return res
