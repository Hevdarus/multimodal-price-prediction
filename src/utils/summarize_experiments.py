from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.experiment_config import load_experiments_config


def collect_history_files(history_dir: str | Path) -> list[Path]:
    history_dir = Path(history_dir)
    history_files = sorted(history_dir.glob("*_history.csv"))
    return history_files


def summarize_history_files(
    history_dir: str | Path = "outputs/models",
) -> pd.DataFrame:
    """
    Read all *_history.csv files and create a summary table with
    the best epoch and best validation metrics for each experiment.
    """
    history_files = collect_history_files(history_dir)

    if not history_files:
        raise FileNotFoundError(f"No history files found in: {Path(history_dir).resolve()}")

    summary_rows = []

    for file_path in history_files:
        df = pd.read_csv(file_path)

        required_cols = {
            "experiment_name",
            "epoch",
            "lr",
            "max_length",
            "train_loss_log_mse",
            "val_loss_log_mse",
            "val_mae_price",
            "val_rmse_price",
        }
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {file_path.name}: {sorted(missing_cols)}"
            )

        best_idx = df["val_loss_log_mse"].idxmin()
        best_row = df.loc[best_idx]

        summary_rows.append(
            {
                "history_file": file_path.name,
                "experiment_name": best_row["experiment_name"],
                "status": "completed",
                "best_epoch": int(best_row["epoch"]),
                "lr": float(best_row["lr"]),
                "max_length": int(best_row["max_length"]),
                "configured_epochs": int(df["epoch"].max()),
                "best_train_loss_log_mse": float(best_row["train_loss_log_mse"]),
                "best_val_loss_log_mse": float(best_row["val_loss_log_mse"]),
                "best_val_mae_price": float(best_row["val_mae_price"]),
                "best_val_rmse_price": float(best_row["val_rmse_price"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by=["best_val_mae_price", "best_val_rmse_price"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return summary_df


def merge_with_config(
    summary_df: pd.DataFrame,
    config_path: str | Path = "Experiments.txt",
) -> pd.DataFrame:
    """
    Merge executed experiment summaries with configured experiments from Experiments.txt.
    Also shows which experiments have not been run yet.
    """
    config_dict = load_experiments_config(config_path)
    config_df = pd.DataFrame(config_dict.values()).copy()

    merged_df = config_df.merge(
        summary_df,
        on=["experiment_name", "lr", "max_length"],
        how="left",
        suffixes=("_config", "_run"),
    )

    merged_df["status"] = merged_df["status"].fillna("not_run")

    # Use configured epochs from txt if run data missing
    if "epochs" in merged_df.columns:
        merged_df["configured_epochs"] = merged_df["configured_epochs"].fillna(merged_df["epochs"])

    sort_cols = ["status", "best_val_mae_price"]
    available_sort_cols = [col for col in sort_cols if col in merged_df.columns]

    merged_df = merged_df.sort_values(
        by=available_sort_cols,
        ascending=[True] * len(available_sort_cols),
        na_position="last",
    ).reset_index(drop=True)

    return merged_df


def save_summary_outputs(
    merged_df: pd.DataFrame,
    output_dir: str | Path = "outputs/models",
) -> tuple[Path, Path]:
    """
    Save the merged summary both as CSV and markdown.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "experiment_summary.csv"
    md_path = output_dir / "experiment_summary.md"

    merged_df.to_csv(csv_path, index=False)

    markdown_df = merged_df.copy()

    preferred_cols = [
        "experiment_name",
        "status",
        "lr",
        "max_length",
        "epochs",
        "configured_epochs",
        "best_epoch",
        "best_val_loss_log_mse",
        "best_val_mae_price",
        "best_val_rmse_price",
    ]
    markdown_cols = [col for col in preferred_cols if col in markdown_df.columns]
    markdown_df = markdown_df[markdown_cols]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Summary\n\n")
        f.write(markdown_df.to_markdown(index=False))

    return csv_path, md_path


def plot_histories(
    history_dir: str | Path = "outputs/models",
    output_dir: str | Path = "outputs/figures/experiments",
) -> list[Path]:
    """
    Create a separate plot for each history CSV:
    - train_loss_log_mse
    - val_loss_log_mse
    """
    history_files = collect_history_files(history_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for file_path in history_files:
        df = pd.read_csv(file_path)

        if "experiment_name" not in df.columns:
            continue

        experiment_name = str(df["experiment_name"].iloc[0])

        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["train_loss_log_mse"], marker="o", label="Train loss")
        plt.plot(df["epoch"], df["val_loss_log_mse"], marker="o", label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Log MSE loss")
        plt.title(f"Loss curves - {experiment_name}")
        plt.legend()
        plt.tight_layout()

        plot_path = output_dir / f"{experiment_name}_loss.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        saved_paths.append(plot_path)

    return saved_paths


def print_summary(merged_df: pd.DataFrame) -> None:
    print("\nEXPERIMENT SUMMARY")
    print("-" * 120)
    print(merged_df.to_string(index=False))

    completed_df = merged_df[merged_df["status"] == "completed"].copy()

    if not completed_df.empty:
        completed_df = completed_df.sort_values(
            by=["best_val_mae_price", "best_val_rmse_price"],
            ascending=[True, True],
        )
        best = completed_df.iloc[0]

        print("\nBEST COMPLETED EXPERIMENT")
        print("-" * 120)
        print(f"Experiment name:     {best['experiment_name']}")
        print(f"Learning rate:       {best['lr']}")
        print(f"Max length:          {best['max_length']}")
        if "best_epoch" in best.index and pd.notna(best["best_epoch"]):
            print(f"Best epoch:          {int(best['best_epoch'])}")
        if "best_val_mae_price" in best.index and pd.notna(best["best_val_mae_price"]):
            print(f"Best val MAE price:  {best['best_val_mae_price']:.4f}")
        if "best_val_rmse_price" in best.index and pd.notna(best["best_val_rmse_price"]):
            print(f"Best val RMSE price: {best['best_val_rmse_price']:.4f}")

    not_run_df = merged_df[merged_df["status"] == "not_run"].copy()
    if not_run_df.empty:
        print("\nAll configured experiments have been run.")
    else:
        print("\nEXPERIMENTS NOT RUN YET")
        print("-" * 120)
        cols = [c for c in ["experiment_name", "lr", "max_length", "epochs"] if c in not_run_df.columns]
        print(not_run_df[cols].to_string(index=False))


if __name__ == "__main__":
    summary_df = summarize_history_files(history_dir="outputs/models")
    merged_df = merge_with_config(summary_df, config_path="Experiments.txt")

    csv_path, md_path = save_summary_outputs(
        merged_df,
        output_dir="outputs/models",
    )

    plot_paths = plot_histories(
        history_dir="outputs/models",
        output_dir="outputs/figures/experiments",
    )

    print_summary(merged_df)

    print("\nSaved summary files:")
    print(f"- {csv_path}")
    print(f"- {md_path}")

    if plot_paths:
        print("\nSaved plots:")
        for path in plot_paths:
            print(f"- {path}")