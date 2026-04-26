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

from src.models.image_dataset import ImageDataset, get_eval_transform, get_train_transform
from src.models.image_model_efficientnet import EfficientNetRegressionModel


def train_epoch(model, loader, optimizer, device, use_huber: bool = True):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        if use_huber:
            loss = F.huber_loss(outputs, targets, delta=1.0)
        else:
            loss = F.mse_loss(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_epoch(model, loader, device, use_huber: bool = True):
    model.eval()
    total_loss = 0.0

    all_preds_log = []
    all_targets_log = []
    all_sample_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            outputs = model(images)

            if use_huber:
                loss = F.huber_loss(outputs, targets, delta=1.0)
            else:
                loss = F.mse_loss(outputs, targets)

            total_loss += loss.item()

            all_preds_log.extend(outputs.detach().cpu().numpy())
            all_targets_log.extend(targets.detach().cpu().numpy())
            all_sample_ids.extend(batch["sample_id"])

    avg_loss = total_loss / len(loader)

    all_preds_log = np.array(all_preds_log)
    all_targets_log = np.array(all_targets_log)

    all_preds_price = np.expm1(all_preds_log)
    all_targets_price = np.expm1(all_targets_log)

    all_preds_price = np.clip(all_preds_price, a_min=0.0, a_max=None)

    mae = mean_absolute_error(all_targets_price, all_preds_price)
    rmse = np.sqrt(mean_squared_error(all_targets_price, all_preds_price))

    metrics = {
        "val_loss": avg_loss,
        "val_mae_price": mae,
        "val_rmse_price": rmse,
    }

    predictions_df = pd.DataFrame(
        {
            "sample_id": all_sample_ids,
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
    parser.add_argument("--train_csv", type=str, default="data/processed/train_split_with_images.csv")
    parser.add_argument("--val_csv", type=str, default="data/processed/val_split_with_images.csv")
    parser.add_argument("--image_dir", type=str, default="data/images")
    parser.add_argument("--experiment_name", type=str, default="image_efficientnet_b0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_huber", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    train_dataset = ImageDataset(
        train_df,
        image_dir=args.image_dir,
        transform=get_train_transform(),
    )
    val_dataset = ImageDataset(
        val_df,
        image_dir=args.image_dir,
        transform=get_eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_cuda,
    )

    model = EfficientNetRegressionModel(
        pretrained=True,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    output_dir = Path("outputs/models/image")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = output_dir / f"{args.experiment_name}.pt"
    val_pred_path = output_dir / f"{args.experiment_name}_val_predictions.csv"
    history_path = output_dir / f"{args.experiment_name}_history.csv"

    best_val_mae = float("inf")
    patience = 3
    patience_counter = 0

    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            use_huber=args.use_huber,
        )
        val_metrics, val_predictions = evaluate_epoch(
            model=model,
            loader=val_loader,
            device=device,
            use_huber=args.use_huber,
        )

        print(f"Train loss:        {train_loss:.4f}")
        print(f"Val loss:          {val_metrics['val_loss']:.4f}")
        print(f"Val MAE (price):   {val_metrics['val_mae_price']:.4f}")
        print(f"Val RMSE (price):  {val_metrics['val_rmse_price']:.4f}")

        history.append(
            {
                "experiment_name": args.experiment_name,
                "epoch": epoch + 1,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "dropout": args.dropout,
                "loss_type": "huber" if args.use_huber else "mse",
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "val_mae_price": val_metrics["val_mae_price"],
                "val_rmse_price": val_metrics["val_rmse_price"],
            }
        )

        current_val_mae = val_metrics["val_mae_price"]

        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            patience_counter = 0

            save_checkpoint(model, best_model_path)
            val_predictions.to_csv(val_pred_path, index=False)

            print("Best model updated and saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    pd.DataFrame(history).to_csv(history_path, index=False)

    print("\nTraining finished.")
    print(f"Saved model: {best_model_path}")
    print(f"Saved predictions: {val_pred_path}")
    print(f"Saved history: {history_path}")