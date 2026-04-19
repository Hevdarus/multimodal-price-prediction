import torch
import torch.nn as nn
from transformers import AutoModel


class TextRegressionModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token embedding
        cls_output = outputs.last_hidden_state[:, 0, :]

        out = self.regressor(cls_output)
        return out.squeeze()