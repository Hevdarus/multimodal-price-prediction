import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from src.models.text_model import TextRegressionModel
from src.models.text_dataset import TextDataset

import pandas as pd


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = torch.nn.functional.mse_loss(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = torch.nn.functional.mse_loss(outputs, targets)

            total_loss += loss.item()

    return total_loss / len(loader)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")

    train_df = pd.read_csv("data/processed/train_split.csv")
    val_df = pd.read_csv("data/processed/val_split.csv")

    use_cuda = torch.cuda.is_available()

    train_dataset = TextDataset(train_df)
    val_dataset = TextDataset(val_df)

    train_loader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=use_cuda)
    val_loader = DataLoader(val_dataset,
                            batch_size=64,
                            num_workers=4,
                            pin_memory=use_cuda)

    model = TextRegressionModel().to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)

        print(f"Epoch {epoch+1}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")