from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.text_dataset import TextDataset
from src.models.text_model import TextRegressionModel
from src.utils.experiment_config import get_experiment_config


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = F.mse_loss(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0

    all_preds_log = []
    all_targets_log = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            outputs = model(input_ids, attention_mask)
            loss = F.mse_loss(outputs, targets)

            total_loss += loss.item()

            all_preds_log.extend(outputs.detach().cpu().numpy())
            all_targets_log.extend(targets.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)

    all_preds_log = np.array(all_preds_log)
    all_targets_log = np.array(all_targets_log)

    # Visszaalakítás eredeti árskálára
    all_preds_price = np.expm1(all_preds_log)
    all_targets_price = np.expm1(all_targets_log)

    # Negatív predikciók levágása 0-ra
    all_preds_price = np.clip(all_preds_price, a_min=0.0, a_max=None)

    mae = mean_absolute_error(all_targets_price, all_preds_price)
    rmse = np.sqrt(mean_squared_error(all_targets_price, all_preds_price))

    metrics = {
        "val_loss_log_mse": avg_loss,
        "val_mae_price": mae,
        "val_rmse_price": rmse,
    }

    predictions_df = pd.DataFrame(
        {
            "target_log_price": all_targets_log,
            "pred_log_price": all_preds_log,
            "target_price": all_targets_price,
            "pred_price": all_preds_price,
        }
    )

    return metrics, predictions_df


def save_checkpoint(model, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="Experiments.txt",
        help="Path to the experiment config txt file.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Experiment name to run from the config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = get_experiment_config(
        config_path=args.config_path,
        experiment_name=args.experiment_name,
    )

    experiment_name = config["experiment_name"]
    lr = config["lr"]
    max_length = config["max_length"]
    num_epochs = config["epochs"]

    print("\nLoaded experiment config:")
    print(f"experiment_name = {experiment_name}")
    print(f"lr              = {lr}")
    print(f"max_length      = {max_length}")
    print(f"epochs          = {num_epochs}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"\nDevice: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")

    train_df = pd.read_csv("data/processed/train_split.csv")
    val_df = pd.read_csv("data/processed/val_split.csv")

    train_dataset = TextDataset(train_df, max_length=max_length)
    val_dataset = TextDataset(val_df, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=use_cuda,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=use_cuda,
    )

    model = TextRegressionModel().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    history = []

    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = output_dir / f"{experiment_name}.pt"
    val_pred_path = output_dir / f"{experiment_name}_val_predictions.csv"
    history_path = output_dir / f"{experiment_name}_history.csv"

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics, val_predictions = evaluate_epoch(model, val_loader, device)

        current_val_loss = val_metrics["val_loss_log_mse"]

        print(f"Train loss (log MSE): {train_loss:.4f}")
        print(f"Val loss (log MSE):   {val_metrics['val_loss_log_mse']:.4f}")
        print(f"Val MAE (price):      {val_metrics['val_mae_price']:.4f}")
        print(f"Val RMSE (price):     {val_metrics['val_rmse_price']:.4f}")

        history.append(
            {
                "experiment_name": experiment_name,
                "epoch": epoch + 1,
                "lr": lr,
                "max_length": max_length,
                "train_loss_log_mse": train_loss,
                "val_loss_log_mse": val_metrics["val_loss_log_mse"],
                "val_mae_price": val_metrics["val_mae_price"],
                "val_rmse_price": val_metrics["val_rmse_price"],
            }
        )

        # Legjobb modell mentése
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            save_checkpoint(model, best_model_path)
            val_predictions.to_csv(val_pred_path, index=False)
            print("Best model updated and saved.")

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)

    print("\nTraining finished.")
    print("Saved files:")
    print(f"- {best_model_path}")
    print(f"- {val_pred_path}")
    print(f"- {history_path}")