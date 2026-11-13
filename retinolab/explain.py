# grad-cam grid: two random test images per grade, scan on top and heatmap below
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import GRADES
from .data import eval_tf
from .gradcam import GradCAM, denormalize, overlay
from .models import build_classifier, gradcam_layer


def gradcam_grid(ckpt_path, test_imgs, test_labels, out_path, per_grade: int = 2, seed: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ck["cfg"]
    model = build_classifier(cfg["backbone"], cfg["head"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    engine = GradCAM(model, gradcam_layer(model, cfg["backbone"]))
    from PIL import Image

    tf = eval_tf(cfg["img_size"])
    rng = np.random.default_rng(seed)
    picks = []
    for g in range(len(GRADES)):
        idx = np.where(test_labels == g)[0]
        if len(idx):
            picks += list(rng.choice(idx, size=min(per_grade, len(idx)), replace=False))
    fig, axes = plt.subplots(2, len(picks), figsize=(1.9 * len(picks), 4.2))
    for k, i in enumerate(picks):
        x = tf(Image.fromarray(test_imgs[i])).unsqueeze(0).to(device)
        cam, probs, pred = engine(x) if cfg["head"] == "ce" else (*engine(x)[:1], None, None)
        img = denormalize(x[0])
        axes[0, k].imshow(img)
        axes[0, k].set_title(f"true: {GRADES[test_labels[i]]}", fontsize=7)
        axes[1, k].imshow(overlay(img, cam))
        if pred is not None:
            axes[1, k].set_title(f"pred: {GRADES[pred]}", fontsize=7,
                                 color="green" if pred == test_labels[i] else "red")
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    engine.close()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
