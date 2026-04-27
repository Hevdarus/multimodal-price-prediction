from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from src.models.text_model import TextRegressionModel
from src.models.image_model import ImageRegressionModel
from src.models.multimodal_model import MultimodalRegressionModel

from src.models.text_dataset import TextDataset
from src.models.image_dataset import ImageDataset, get_eval_transform
from src.models.multimodal_dataset import (
    MultimodalDataset,
    get_multimodal_eval_transform,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["text", "image", "multimodal"],
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="data/images")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--image_encoder", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"])

    return parser.parse_args()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(args, device):
    if args.model_type == "text":
        model = TextRegressionModel()

    elif args.model_type == "image":
        model = ImageRegressionModel()


    elif args.model_type == "multimodal":
        model = MultimodalRegressionModel(image_encoder_name=args.image_encoder)

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def build_dataset(args):
    df = pd.read_csv(args.data_csv)

    if args.model_type == "text":
        dataset = TextDataset(
            df,
            max_length=args.max_length,
        )

    elif args.model_type == "image":
        dataset = ImageDataset(
            df,
            image_dir=args.image_dir,
            transform=get_eval_transform(),
        )

    elif args.model_type == "multimodal":
        dataset = MultimodalDataset(
            df=df,
            image_dir=args.image_dir,
            max_length=args.max_length,
            transform=get_multimodal_eval_transform(),
        )

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    return dataset


def evaluate(model, loader, device):
    preds_log = []
    targets_log = []
    sample_ids = []

    with torch.no_grad():
        for batch in loader:
            if "sample_id" in batch:
                sample_ids.extend(batch["sample_id"])

            if "target" in batch:
                targets_log.extend(batch["target"].cpu().numpy())

            if isinstance(model, TextRegressionModel):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                outputs = model(input_ids, attention_mask)

            elif isinstance(model, ImageRegressionModel):
                images = batch["image"].to(device, non_blocking=True)

                outputs = model(images)

            elif isinstance(model, MultimodalRegressionModel):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                images = batch["image"].to(device, non_blocking=True)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                )

            else:
                raise ValueError("Unsupported model class.")

            preds_log.extend(outputs.detach().cpu().numpy())

    preds_log = np.array(preds_log)
    preds_price = np.expm1(preds_log)
    preds_price = np.clip(preds_price, a_min=0.0, a_max=None)

    if len(targets_log) > 0:
        targets_log = np.array(targets_log)
        targets_price = np.expm1(targets_log)

        mae = mean_absolute_error(targets_price, preds_price)
        rmse = np.sqrt(mean_squared_error(targets_price, preds_price))
    else:
        targets_log = None
        targets_price = None
        mae = None
        rmse = None

    return {
        "sample_ids": sample_ids,
        "preds_log": preds_log,
        "preds_price": preds_price,
        "targets_log": targets_log,
        "targets_price": targets_price,
        "mae": mae,
        "rmse": rmse,
    }


def make_output_paths(args):
    model_name = Path(args.model_path).stem
    data_name = Path(args.data_csv).stem

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_path is not None:
        prediction_path = Path(args.output_path)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        prediction_path = output_dir / f"{model_name}__on__{data_name}_predictions.csv"

    metrics_path = output_dir / f"{model_name}__on__{data_name}_metrics.csv"

    return prediction_path, metrics_path


def save_outputs(args, results):
    prediction_path, metrics_path = make_output_paths(args)

    out_df = pd.DataFrame(
        {
            "sample_id": results["sample_ids"],
            "pred_log_price": results["preds_log"],
            "pred_price": results["preds_price"],
        }
    )

    if results["targets_log"] is not None:
        out_df["target_log_price"] = results["targets_log"]
        out_df["target_price"] = results["targets_price"]
        out_df["absolute_error"] = (out_df["target_price"] - out_df["pred_price"]).abs()
        out_df["squared_error"] = (out_df["target_price"] - out_df["pred_price"]) ** 2

    out_df.to_csv(prediction_path, index=False)

    metrics_df = pd.DataFrame(
        [
            {
                "model_type": args.model_type,
                "model_path": args.model_path,
                "data_csv": args.data_csv,
                "image_dir": args.image_dir if args.model_type in ["image", "multimodal"] else None,
                "batch_size": args.batch_size,
                "max_length": args.max_length if args.model_type in ["text", "multimodal"] else None,
                "n_samples": len(out_df),
                "mae": results["mae"],
                "rmse": results["rmse"],
                "prediction_file": str(prediction_path),
            }
        ]
    )

    metrics_df.to_csv(metrics_path, index=False)

    return prediction_path, metrics_path


if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    print(f"Device: {device}")

    model = build_model(args, device)
    dataset = build_dataset(args)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    results = evaluate(model, loader, device)

    print("\n===== EVALUATION RESULTS =====")
    print(f"Model type: {args.model_type}")
    print(f"Samples:    {len(results['preds_price'])}")

    if results["mae"] is not None:
        print(f"MAE:        {results['mae']:.4f}")
        print(f"RMSE:       {results['rmse']:.4f}")
    else:
        print("No target column found. Saved predictions only.")

    prediction_path, metrics_path = save_outputs(args, results)

    print("\nSaved files:")
    print(f"- Predictions: {prediction_path}")
    print(f"- Metrics:     {metrics_path}")