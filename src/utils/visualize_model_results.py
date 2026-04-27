from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_COLUMNS = {
    "sample_id",
    "target_price",
    "pred_price",
}


def load_prediction_file(csv_path: str | Path, model_name: str) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {csv_path}")

    df = pd.read_csv(csv_path).copy()

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Prediction file {csv_path.name} is missing required columns: {sorted(missing)}"
        )

    df["model_name"] = model_name
    df["abs_error"] = (df["target_price"] - df["pred_price"]).abs()
    df["squared_error"] = (df["target_price"] - df["pred_price"]) ** 2
    df["residual"] = df["pred_price"] - df["target_price"]

    return df


def create_metrics_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model_name, model_df in df.groupby("model_name"):
        mae = model_df["abs_error"].mean()
        rmse = np.sqrt(model_df["squared_error"].mean())
        median_ae = model_df["abs_error"].median()
        mean_pred = model_df["pred_price"].mean()
        mean_target = model_df["target_price"].mean()
        n_samples = len(model_df)

        rows.append(
            {
                "model_name": model_name,
                "n_samples": n_samples,
                "mae": mae,
                "rmse": rmse,
                "median_absolute_error": median_ae,
                "mean_target_price": mean_target,
                "mean_pred_price": mean_pred,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(by="mae", ascending=True).reset_index(drop=True)
    return summary_df


def save_metrics_summary(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)


def save_metrics_markdown(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison\n\n")
        f.write(summary_df.to_markdown(index=False))


def plot_true_vs_pred(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for model_name, model_df in df.groupby("model_name"):
        plt.figure(figsize=(7, 7))
        plt.scatter(
            model_df["target_price"],
            model_df["pred_price"],
            alpha=0.4,
            s=12,
        )

        min_val = min(model_df["target_price"].min(), model_df["pred_price"].min())
        max_val = max(model_df["target_price"].max(), model_df["pred_price"].max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

        plt.xlabel("True price")
        plt.ylabel("Predicted price")
        plt.title(f"True vs predicted price - {model_name}")
        plt.tight_layout()

        out_path = output_dir / f"{model_name}_true_vs_pred.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        saved_paths.append(out_path)

    return saved_paths


def plot_residuals(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for model_name, model_df in df.groupby("model_name"):
        plt.figure(figsize=(8, 6))
        plt.scatter(
            model_df["target_price"],
            model_df["residual"],
            alpha=0.4,
            s=12,
        )
        plt.axhline(0.0, linestyle="--")

        plt.xlabel("True price")
        plt.ylabel("Residual (pred - true)")
        plt.title(f"Residual plot - {model_name}")
        plt.tight_layout()

        out_path = output_dir / f"{model_name}_residuals.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        saved_paths.append(out_path)

    return saved_paths


def plot_absolute_error_distribution(
    df: pd.DataFrame,
    output_path: str | Path,
    bins: int = 50,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for model_name, model_df in df.groupby("model_name"):
        plt.hist(
            model_df["abs_error"],
            bins=bins,
            alpha=0.45,
            density=True,
            label=model_name,
        )

    plt.xlabel("Absolute error")
    plt.ylabel("Density")
    plt.title("Absolute error distribution by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_model_mae_bar(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["model_name"], summary_df["mae"])
    plt.ylabel("MAE")
    plt.title("Model comparison by MAE")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def merge_prediction_files(
    files: Iterable[tuple[str, str | Path]],
) -> pd.DataFrame:
    dfs = []
    for model_name, path in files:
        dfs.append(load_prediction_file(path, model_name))
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    prediction_files = [
        ("text", "outputs/models/text_best_test_predictions.csv"),
        ("image_resnet18", "outputs/models/image_resnet18_best_test_predictions.csv"),
        ("multimodal", "outputs/models/multimodal_best_test_predictions.csv"),
    ]

    combined_df = merge_prediction_files(prediction_files)

    output_dir = Path("outputs/figures/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = create_metrics_summary(combined_df)
    save_metrics_summary(summary_df, output_dir / "model_comparison_summary.csv")
    save_metrics_markdown(summary_df, output_dir / "model_comparison_summary.md")

    mae_bar_path = output_dir / "model_mae_bar.png"
    abs_err_path = output_dir / "absolute_error_distribution.png"

    plot_model_mae_bar(summary_df, mae_bar_path)
    plot_absolute_error_distribution(combined_df, abs_err_path)
    true_vs_pred_paths = plot_true_vs_pred(combined_df, output_dir)
    residual_paths = plot_residuals(combined_df, output_dir)

    print("\nMODEL COMPARISON SUMMARY")
    print("-" * 80)
    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print(f"- {output_dir / 'model_comparison_summary.csv'}")
    print(f"- {output_dir / 'model_comparison_summary.md'}")
    print(f"- {mae_bar_path}")
    print(f"- {abs_err_path}")
    for p in true_vs_pred_paths + residual_paths:
        print(f"- {p}")