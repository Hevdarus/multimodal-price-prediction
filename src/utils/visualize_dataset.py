from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(path: str):
    return pd.read_csv(path)


def plot_price_distribution(df, title, output_path):
    plt.figure(figsize=(8, 5))
    plt.hist(df["price"], bins=50)
    plt.title(title)
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_log_price_distribution(df, title, output_path):
    plt.figure(figsize=(8, 5))
    plt.hist(df["log_price"], bins=50)
    plt.title(title)
    plt.xlabel("Log Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_split_comparison(train_df, val_df, test_df, output_path):
    plt.figure(figsize=(8, 5))

    plt.hist(train_df["log_price"], bins=50, alpha=0.5, label="Train")
    plt.hist(val_df["log_price"], bins=50, alpha=0.5, label="Validation")
    plt.hist(test_df["log_price"], bins=50, alpha=0.5, label="Test")

    plt.legend()
    plt.xlabel("Log Price")
    plt.ylabel("Frequency")
    plt.title("Price Distribution Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_text_length(df, output_path):
    lengths = df["catalog_content"].astype(str).apply(len)

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=50)
    plt.xlabel("Text Length (characters)")
    plt.ylabel("Frequency")
    plt.title("Text Length Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_missing_images(df, output_path):
    if "image_path" not in df.columns:
        return

    missing = df["image_path"].isna().sum()
    present = len(df) - missing

    plt.figure(figsize=(5, 5))
    plt.bar(["Has Image", "Missing Image"], [present, missing])
    plt.title("Image Availability")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_basic_stats(df, name):
    print(f"\n--- {name} ---")
    print(f"Samples: {len(df)}")
    print(f"Price mean: {df['price'].mean():.2f}")
    print(f"Price std: {df['price'].std():.2f}")
    print(f"Min price: {df['price'].min():.2f}")
    print(f"Max price: {df['price'].max():.2f}")


def main():
    output_dir = Path("outputs/figures/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_data("data/processed/train_split_with_images.csv")
    val = load_data("data/processed/val_split_with_images.csv")
    test = load_data("data/processed/test_split_with_images.csv")

    # ===== STATISZTIKA =====
    print_basic_stats(train, "TRAIN")
    print_basic_stats(val, "VALIDATION")
    print_basic_stats(test, "TEST")

    # ===== ÁBRÁK =====
    plot_price_distribution(train, "Train Price Distribution",
                            output_dir / "train_price.png")

    plot_log_price_distribution(train, "Train Log Price Distribution",
                                output_dir / "train_log_price.png")

    plot_split_comparison(train, val, test,
                          output_dir / "split_comparison.png")

    plot_text_length(train,
                     output_dir / "text_length.png")

    plot_missing_images(train,
                        output_dir / "image_availability.png")

    print("\nFigures saved to:", output_dir)


if __name__ == "__main__":
    main()