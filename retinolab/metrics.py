# metrics for DR grading. quadratic weighted kappa is the usual headline number for DR
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score

from .config import NUM_CLASSES, REFERABLE_FROM


def qwk(y, pred) -> float:
    return float(cohen_kappa_score(y, pred, weights="quadratic", labels=list(range(NUM_CLASSES))))


def referable_auc(y, score) -> float:
    # can the model separate grade >= 2 from grade < 2 using its severity score
    return float(roc_auc_score(np.asarray(y) >= REFERABLE_FROM, score))


def evaluate(y, pred, score) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y, pred, labels=labels)
    recall = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan") for i in labels]
    ref_true, ref_pred = y >= REFERABLE_FROM, pred >= REFERABLE_FROM
    tp, fn = int((ref_true & ref_pred).sum()), int((ref_true & ~ref_pred).sum())
    tn, fp = int((~ref_true & ~ref_pred).sum()), int((~ref_true & ref_pred).sum())
    return {
        "n": int(len(y)),
        "accuracy": float((y == pred).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "qwk": qwk(y, pred),
        "referable_auc": referable_auc(y, score),
        "referable_sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "referable_specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "per_class_recall": recall,
        "confusion_matrix": cm.tolist(),
    }


def bootstrap_ci(y, pred, fn=qwk, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    # percentile bootstrap over test images
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        vals.append(fn(y[idx], pred[idx]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def apply_thresholds(score, thresholds) -> np.ndarray:
    return np.digitize(np.asarray(score), thresholds)


def fit_thresholds(score, y, init=(0.5, 1.5, 2.5, 3.5), passes: int = 3) -> list[float]:
    # for a regression-style model: pick the 4 cut points that maximise kappa on validation
    t = list(init)
    for _ in range(passes):
        for k in range(4):
            lo = t[k - 1] + 0.01 if k > 0 else -0.5
            hi = t[k + 1] - 0.01 if k < 3 else 4.5
            best, best_v = t[k], qwk(y, apply_thresholds(score, t))
            for c in np.arange(lo, hi, 0.02):
                trial = t.copy()
                trial[k] = float(c)
                v = qwk(y, apply_thresholds(score, trial))
                if v > best_v + 1e-12:
                    best, best_v = float(c), v
            t[k] = best
    return t
