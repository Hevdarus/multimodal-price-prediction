from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_split(csv_path: str | Path, split_name: str) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path).copy()
    df["split"] = split_name
    return df


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("split")
        .agg(
            n_samples=("sample_id", "count"),
            price_mean=("price", "mean"),
            price_std=("price", "std"),
            price_min=("price", "min"),
            price_median=("price", "median"),
            price_max=("price", "max"),
            log_price_mean=("log_price", "mean"),
            log_price_std=("log_price", "std"),
        )
        .reset_index()
    )

    return summary


def save_summary_table(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)


def plot_price_distribution(
    df: pd.DataFrame,
    output_path: str | Path,
    bins: int = 60,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        plt.hist(
            split_df["price"],
            bins=bins,
            alpha=0.45,
            label=split_name,
            density=True,
        )

    plt.xlabel("Price")
    plt.ylabel("Density")
    plt.title("Price distribution across train / val / test splits")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_log_price_distribution(
    df: pd.DataFrame,
    output_path: str | Path,
    bins: int = 60,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        plt.hist(
            split_df["log_price"],
            bins=bins,
            alpha=0.45,
            label=split_name,
            density=True,
        )

    plt.xlabel("log_price")
    plt.ylabel("Density")
    plt.title("Log-price distribution across train / val / test splits")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_price_bin_distribution(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = (
        df.groupby(["split", "price_bin"])
        .size()
        .reset_index(name="count")
    )

    pivot_df = counts.pivot(index="price_bin", columns="split", values="count").fillna(0)

    ax = pivot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_xlabel("price_bin")
    ax.set_ylabel("Count")
    ax.set_title("Price bin distribution across splits")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_boxplot_price(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    data = [
        df[df["split"] == "train"]["price"],
        df[df["split"] == "val"]["price"],
        df[df["split"] == "test"]["price"],
    ]
    plt.boxplot(data, labels=["train", "val", "test"], showfliers=False)
    plt.ylabel("Price")
    plt.title("Price boxplot across splits")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    train_df = load_split("data/processed/train_split.csv", "train")
    val_df = load_split("data/processed/val_split.csv", "val")
    test_df = load_split("data/processed/test_split.csv", "test")

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    output_dir = Path("outputs/figures/splits")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = create_summary_table(combined_df)
    save_summary_table(summary_df, output_dir / "split_summary.csv")

    plot_price_distribution(combined_df, output_dir / "price_distribution.png")
    plot_log_price_distribution(combined_df, output_dir / "log_price_distribution.png")
    plot_price_bin_distribution(combined_df, output_dir / "price_bin_distribution.png")
    plot_boxplot_price(combined_df, output_dir / "price_boxplot.png")

    print("\nSplit summary:")
    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print("- outputs/figures/splits/split_summary.csv")
    print("- outputs/figures/splits/price_distribution.png")
    print("- outputs/figures/splits/log_price_distribution.png")
    print("- outputs/figures/splits/price_bin_distribution.png")
    print("- outputs/figures/splits/price_boxplot.png")