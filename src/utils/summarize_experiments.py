from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


def collect_history_files(root_dir: str | Path = "outputs/models") -> list[Path]:
    root_dir = Path(root_dir)
    return sorted(root_dir.rglob("*_history.csv"))


def infer_model_type(experiment_name: str) -> str:
    name = experiment_name.lower()

    if "multimodal" in name or "multi" in name:
        return "multimodal"
    if "image" in name or "resnet" in name or "efficientnet" in name or "efficinet" in name:
        return "image"
    if "text" in name or "distilbert" in name:
        return "text"

    return "unknown"


def extract_param_from_name(name: str, pattern: str):
    match = re.search(pattern, name.lower())
    return match.group(1) if match else None


def normalize_history_df(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    df = df.copy()

    if "experiment_name" not in df.columns:
        experiment_name = file_path.name.replace("_history.csv", "")
        df["experiment_name"] = experiment_name

    # Egységes loss oszlopnevek
    if "train_loss_log_mse" not in df.columns and "train_loss" in df.columns:
        df["train_loss_log_mse"] = df["train_loss"]

    if "val_loss_log_mse" not in df.columns and "val_loss" in df.columns:
        df["val_loss_log_mse"] = df["val_loss"]

    if "val_loss_log_mse" not in df.columns and "val_loss_mse" in df.columns:
        df["val_loss_log_mse"] = df["val_loss_mse"]

    if "train_loss_log_mse" not in df.columns and "train_loss_mse" in df.columns:
        df["train_loss_log_mse"] = df["train_loss_mse"]

    if "epoch" not in df.columns:
        raise ValueError(f"Missing epoch column in {file_path}")

    if "val_mae_price" not in df.columns:
        raise ValueError(f"Missing val_mae_price column in {file_path}")

    if "val_rmse_price" not in df.columns:
        raise ValueError(f"Missing val_rmse_price column in {file_path}")

    if "lr" not in df.columns:
        df["lr"] = pd.NA

    if "batch_size" not in df.columns:
        df["batch_size"] = pd.NA

    if "max_length" not in df.columns:
        experiment_name = str(df["experiment_name"].iloc[0])
        extracted_len = extract_param_from_name(experiment_name, r"len(\d+)")
        df["max_length"] = int(extracted_len) if extracted_len is not None else pd.NA

    if "dropout" not in df.columns:
        df["dropout"] = pd.NA

    if "loss_type" not in df.columns:
        df["loss_type"] = "mse"

    return df


def summarize_single_history(file_path: Path) -> dict:
    raw_df = pd.read_csv(file_path)
    df = normalize_history_df(raw_df, file_path)

    experiment_name = str(df["experiment_name"].iloc[0])
    model_type = infer_model_type(experiment_name)

    # Fő modellválasztási metrika: MAE price térben
    best_idx = df["val_mae_price"].idxmin()
    best_row = df.loc[best_idx]

    final_row = df.sort_values("epoch").iloc[-1]

    best_val_loss = (
        df["val_loss_log_mse"].min()
        if "val_loss_log_mse" in df.columns
        else pd.NA
    )

    best_loss_epoch = (
        int(df.loc[df["val_loss_log_mse"].idxmin(), "epoch"])
        if "val_loss_log_mse" in df.columns
        else pd.NA
    )

    return {
        "history_file": str(file_path),
        "experiment_name": experiment_name,
        "model_type": model_type,
        "status": "completed",
        "configured_epochs": int(df["epoch"].max()),
        "best_epoch_by_mae": int(best_row["epoch"]),
        "best_epoch_by_loss": best_loss_epoch,
        "lr": best_row.get("lr", pd.NA),
        "batch_size": best_row.get("batch_size", pd.NA),
        "max_length": best_row.get("max_length", pd.NA),
        "dropout": best_row.get("dropout", pd.NA),
        "loss_type": best_row.get("loss_type", "mse"),
        "best_train_loss_log_mse": best_row.get("train_loss_log_mse", pd.NA),
        "best_val_loss_log_mse_at_best_mae": best_row.get("val_loss_log_mse", pd.NA),
        "best_val_loss_log_mse": best_val_loss,
        "best_val_mae_price": best_row["val_mae_price"],
        "best_val_rmse_price": best_row["val_rmse_price"],
        "final_epoch": int(final_row["epoch"]),
        "final_val_mae_price": final_row["val_mae_price"],
        "final_val_rmse_price": final_row["val_rmse_price"],
    }


def summarize_all_histories(root_dir: str | Path = "outputs/models") -> pd.DataFrame:
    history_files = collect_history_files(root_dir)

    if not history_files:
        raise FileNotFoundError(f"No *_history.csv files found under: {Path(root_dir).resolve()}")

    rows = []
    failed = []

    for file_path in history_files:
        try:
            rows.append(summarize_single_history(file_path))
        except Exception as exc:
            failed.append({"history_file": str(file_path), "error": str(exc)})

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["model_type", "best_val_mae_price", "best_val_rmse_price"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    failed_df = pd.DataFrame(failed)

    return summary_df, failed_df


def create_best_by_model_type(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    best_rows = []
    for model_type, group in summary_df.groupby("model_type"):
        best_rows.append(group.sort_values("best_val_mae_price").iloc[0])

    return pd.DataFrame(best_rows).sort_values("best_val_mae_price").reset_index(drop=True)


def save_tables(
    summary_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    output_dir: str | Path = "outputs/summary",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "experiment_summary_all.csv"
    summary_md_path = output_dir / "experiment_summary_all.md"
    best_path = output_dir / "best_by_model_type.csv"
    failed_path = output_dir / "experiment_summary_failed.csv"

    summary_df.to_csv(summary_path, index=False)

    md_cols = [
        "model_type",
        "experiment_name",
        "best_epoch_by_mae",
        "lr",
        "batch_size",
        "max_length",
        "loss_type",
        "best_val_mae_price",
        "best_val_rmse_price",
        "best_val_loss_log_mse_at_best_mae",
    ]
    md_cols = [c for c in md_cols if c in summary_df.columns]

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Summary\n\n")
        f.write(summary_df[md_cols].to_markdown(index=False))

    best_df = create_best_by_model_type(summary_df)
    best_df.to_csv(best_path, index=False)

    if not failed_df.empty:
        failed_df.to_csv(failed_path, index=False)

    print("\nSaved tables:")
    print(f"- {summary_path}")
    print(f"- {summary_md_path}")
    print(f"- {best_path}")
    if not failed_df.empty:
        print(f"- {failed_path}")


def plot_history_file(file_path: Path, output_dir: Path) -> Path | None:
    raw_df = pd.read_csv(file_path)
    df = normalize_history_df(raw_df, file_path)

    if "train_loss_log_mse" not in df.columns or "val_loss_log_mse" not in df.columns:
        return None

    experiment_name = str(df["experiment_name"].iloc[0])

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss_log_mse"], marker="o", label="Train loss")
    plt.plot(df["epoch"], df["val_loss_log_mse"], marker="o", label="Validation loss")

    best_epoch = int(df.loc[df["val_mae_price"].idxmin(), "epoch"])
    plt.axvline(best_epoch, linestyle="--", label=f"Best MAE epoch: {best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss in log-price space")
    plt.title(f"Loss curves - {experiment_name}")
    plt.legend()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{experiment_name}_loss.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


def plot_all_histories(
    root_dir: str | Path = "outputs/models",
    output_dir: str | Path = "outputs/figures/experiments",
) -> list[Path]:
    output_dir = Path(output_dir)
    plot_paths = []

    for file_path in collect_history_files(root_dir):
        try:
            plot_path = plot_history_file(file_path, output_dir)
            if plot_path is not None:
                plot_paths.append(plot_path)
        except Exception as exc:
            print(f"Could not plot {file_path}: {exc}")

    return plot_paths


def plot_model_type_comparison(
    summary_df: pd.DataFrame,
    output_dir: str | Path = "outputs/figures/experiments",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_df = create_best_by_model_type(summary_df)

    plt.figure(figsize=(8, 5))
    plt.bar(best_df["model_type"], best_df["best_val_mae_price"])
    plt.ylabel("Best validation MAE")
    plt.xlabel("Model type")
    plt.title("Best validation MAE by model type")
    plt.tight_layout()

    out_path = output_dir / "best_val_mae_by_model_type.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def print_summary(summary_df: pd.DataFrame, failed_df: pd.DataFrame) -> None:
    print("\nEXPERIMENT SUMMARY")
    print("-" * 120)

    display_cols = [
        "model_type",
        "experiment_name",
        "best_epoch_by_mae",
        "lr",
        "batch_size",
        "max_length",
        "loss_type",
        "best_val_mae_price",
        "best_val_rmse_price",
    ]
    display_cols = [c for c in display_cols if c in summary_df.columns]

    print(summary_df[display_cols].to_string(index=False))

    print("\nBEST BY MODEL TYPE")
    print("-" * 120)
    best_df = create_best_by_model_type(summary_df)
    print(best_df[display_cols].to_string(index=False))

    if not failed_df.empty:
        print("\nFAILED FILES")
        print("-" * 120)
        print(failed_df.to_string(index=False))


if __name__ == "__main__":
    root_dir = Path("outputs/models")

    summary_df, failed_df = summarize_all_histories(root_dir=root_dir)

    save_tables(
        summary_df=summary_df,
        failed_df=failed_df,
        output_dir="outputs/summary",
    )

    plot_paths = plot_all_histories(
        root_dir=root_dir,
        output_dir="outputs/figures/experiments",
    )

    comparison_plot = plot_model_type_comparison(
        summary_df=summary_df,
        output_dir="outputs/figures/experiments",
    )

    print_summary(summary_df, failed_df)

    print("\nSaved plots:")
    for p in plot_paths:
        print(f"- {p}")
    print(f"- {comparison_plot}")