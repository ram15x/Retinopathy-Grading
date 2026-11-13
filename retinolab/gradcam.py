# grad-cam, done with a forward hook plus a hook on the tensor itself

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from .config import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, _module, _inp, out):
        self._acts = out
        out.register_hook(lambda g: setattr(self, "_grads", g))

    def close(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        # x is one image (1,3,H,W). returns heatmap in 0..1, class probabilities, class index
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            probs = torch.softmax(logits.detach(), dim=1)[0]
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1))
            logits[0, class_idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._acts).sum(dim=1))[0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
        return cam, probs.cpu().numpy(), class_idx


def denormalize(x: torch.Tensor) -> np.ndarray:
    # undo the normalisation so the image can be shown
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(image01: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[..., ::-1] / 255.0
    return np.clip((1 - alpha) * image01 + alpha * heat, 0, 1)
