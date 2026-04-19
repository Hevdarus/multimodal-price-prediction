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
        text = self.df.iloc[idx]["catalog_content"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }

        if "log_price" in self.df.columns:
            item["target"] = torch.tensor(
                self.df.iloc[idx]["log_price"],
                dtype=torch.float
            )

        return item