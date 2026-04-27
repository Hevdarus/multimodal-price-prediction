import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, df, max_length=128):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row["catalog_content"])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        sample_id = row["sample_id"] if "sample_id" in self.df.columns else idx

        item = {
            "sample_id": str(sample_id),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if "log_price" in self.df.columns:
            item["target"] = torch.tensor(
                row["log_price"],
                dtype=torch.float32
            )

        return item