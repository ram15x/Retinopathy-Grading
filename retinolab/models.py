# classifier backbones, all start from imagenet weights
from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .config import NUM_CLASSES

BACKBONES = ("resnet50", "efficientnet_v2_s", "convnext_tiny", "maxvit_t", "resnet18")


def build_classifier(name: str, head: str = "ce", pretrained: bool = True) -> nn.Module:
    # head "ce": 5 logits. head "ordinal": one number that predicts the grade (regression)
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}, pick from {BACKBONES}")
    if head not in ("ce", "ordinal"):
        raise ValueError(f"unknown head {head!r}")
    out = NUM_CLASSES if head == "ce" else 1
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, out)
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, out)
    elif name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        )
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, out)
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, out)
    else:  # maxvit_t, convolution + attention hybrid
        m = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT if pretrained else None)
        m.classifier[5] = nn.Linear(m.classifier[5].in_features, out, bias=False)
    return m


def gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    # last conv stage, that is where the heatmap has the most spatial detail
    if name in ("convnext_tiny", "efficientnet_v2_s"):
        return model.features[-1]
    if name == "maxvit_t":
        return model.blocks[-1]
    return model.layer4[-1]
