from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet18, ResNet18_Weights


class MultimodalRegressionModel(nn.Module):
    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        image_pretrained: bool = True,
        text_dropout: float = 0.1,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.2,
    ):
        super().__init__()

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_hidden_size = self.text_encoder.config.hidden_size

        # Image encoder
        if image_pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.image_encoder = resnet18(weights=weights)
        else:
            self.image_encoder = resnet18(weights=None)

        image_hidden_size = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()

        # Optional projection layers
        self.text_proj = nn.Sequential(
            nn.Dropout(text_dropout),
            nn.Linear(text_hidden_size, 256),
            nn.ReLU(),
        )

        self.image_proj = nn.Sequential(
            nn.Linear(image_hidden_size, 256),
            nn.ReLU(),
        )

        # Fusion head
        fusion_input_dim = 256 + 256

        self.regressor = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask, images):
        # Text branch
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_cls = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_proj(text_cls)

        # Image branch
        image_features_raw = self.image_encoder(images)
        image_features = self.image_proj(image_features_raw)

        # Fusion
        fused = torch.cat([text_features, image_features], dim=1)

        output = self.regressor(fused)
        return output.squeeze(1)