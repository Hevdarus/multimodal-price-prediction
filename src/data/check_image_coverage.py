from __future__ import annotations

from pathlib import Path

import pandas as pd


def add_image_availability(
    df: pd.DataFrame,
    image_dir: str | Path = "data/images",
    image_extension: str = ".jpg",
) -> pd.DataFrame:
    """
    Add a boolean column indicating whether the image file exists locally.
    """
    image_dir = Path(image_dir)
    df = df.copy()

    df["image_exists"] = df["sample_id"].apply(
        lambda sample_id: (image_dir / f"{sample_id}{image_extension}").exists()
    )
    return df


def summarize_coverage(df: pd.DataFrame, name: str) -> None:
    total = len(df)
    available = int(df["image_exists"].sum())
    missing = total - available
    coverage = available / total if total > 0 else 0.0

    print(f"\n{name.upper()} IMAGE COVERAGE")
    print("-" * 50)
    print(f"Total rows:          {total}")
    print(f"Images available:    {available}")
    print(f"Images missing:      {missing}")
    print(f"Coverage:            {coverage:.2%}")


def save_missing_ids(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save sample_ids for rows where image is missing.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing_df = df.loc[~df["image_exists"], ["sample_id", "image_link"]].copy()
    missing_df.to_csv(output_path, index=False)


def save_available_subset(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save only rows where image is available.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    available_df = df.loc[df["image_exists"]].copy()
    available_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    train_path = Path("data/processed/train_split.csv")
    val_path = Path("data/processed/val_split.csv")
    image_dir = Path("data/images")

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing file: {val_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    train_df = add_image_availability(train_df, image_dir=image_dir)
    val_df = add_image_availability(val_df, image_dir=image_dir)

    summarize_coverage(train_df, "train")
    summarize_coverage(val_df, "validation")

    print("\nSample missing train images:")
    print(train_df.loc[~train_df["image_exists"], ["sample_id", "image_link"]].head(10))

    print("\nSample missing validation images:")
    print(val_df.loc[~val_df["image_exists"], ["sample_id", "image_link"]].head(10))

    # Save reports
    save_missing_ids(train_df, "data/processed/train_missing_images.csv")
    save_missing_ids(val_df, "data/processed/val_missing_images.csv")

    # Save filtered subsets that can be used directly for image training
    save_available_subset(train_df, "data/processed/train_split_with_images.csv")
    save_available_subset(val_df, "data/processed/val_split_with_images.csv")

    print("\nSaved files:")
    print("- data/processed/train_missing_images.csv")
    print("- data/processed/val_missing_images.csv")
    print("- data/processed/train_split_with_images.csv")
    print("- data/processed/val_split_with_images.csv")