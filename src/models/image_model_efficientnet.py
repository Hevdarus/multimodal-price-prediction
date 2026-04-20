from __future__ import annotations

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetRegressionModel(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            self.backbone = efficientnet_b0(weights=weights)
        else:
            self.backbone = efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, images):
        outputs = self.backbone(images)
        return outputs.squeeze(1)