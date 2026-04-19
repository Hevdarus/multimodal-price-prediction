from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ImageRegressionModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet18(weights=None)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, images):
        outputs = self.backbone(images)
        return outputs.squeeze(1)