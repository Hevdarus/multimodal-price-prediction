from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class ImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path = "data/images",
        transform=None,
        fallback_image_size: int = 224,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.fallback_image_size = fallback_image_size

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, sample_id: str) -> Image.Image:
        image_path = self.image_dir / f"{sample_id}.jpg"

        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception:
            return Image.new("RGB", (self.fallback_image_size, self.fallback_image_size), color=(0, 0, 0))

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        sample_id = row["sample_id"]
        image = self._load_image(sample_id)

        if self.transform is not None:
            image = self.transform(image)

        item = {
            "sample_id": sample_id,
            "image": image,
        }

        if "log_price" in self.df.columns:
            item["target"] = torch.tensor(row["log_price"], dtype=torch.float32)

        return item