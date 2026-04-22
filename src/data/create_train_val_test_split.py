from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def ensure_price_bins(df: pd.DataFrame, target_col: str = "price", n_bins: int = 10) -> pd.DataFrame:
    """
    Create quantile-based price bins for stratified splitting if not already present.
    """
    df = df.copy()

    if "price_bin" not in df.columns:
        df["price_bin"] = pd.qcut(
            df[target_col],
            q=n_bins,
            labels=False,
            duplicates="drop",
        )

    return df


def add_image_availability(
    df: pd.DataFrame,
    image_dir: str | Path = "data/images",
    image_extension: str = ".jpg",
) -> pd.DataFrame:
    """
    Add a boolean column showing whether the image exists locally.
    """
    image_dir = Path(image_dir)
    df = df.copy()

    df["image_exists"] = df["sample_id"].apply(
        lambda sample_id: (image_dir / f"{sample_id}{image_extension}").exists()
    )
    return df


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_split_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name.upper()} SUMMARY")
    print("-" * 60)
    print(f"Rows: {len(df)}")

    if "price" in df.columns:
        print("Price statistics:")
        print(df["price"].describe())

    if "price_bin" in df.columns:
        print("\nPrice bin distribution:")
        print(df["price_bin"].value_counts().sort_index())

    if "image_exists" in df.columns:
        image_count = int(df["image_exists"].sum())
        print(f"\nImages available: {image_count}")
        print(f"Image coverage: {image_count / len(df):.2%}" if len(df) > 0 else "Image coverage: 0.00%")


if __name__ == "__main__":
    input_path = Path("data/processed/train_preprocessed.csv")
    image_dir = Path("data/images")
    output_dir = Path("data/processed")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing file: {input_path}\n"
            f"Run preprocessing first to create train_preprocessed.csv"
        )

    df = pd.read_csv(input_path)

    required_cols = {"sample_id", "catalog_content", "price", "log_price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {sorted(missing)}")

    df = ensure_price_bins(df, target_col="price", n_bins=10)

    # 1) First split: train vs temp (70 / 30)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["price_bin"],
    )

    # 2) Second split: temp -> val/test (15 / 15 overall)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["price_bin"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save full splits
    save_dataframe(train_df, output_dir / "train_split.csv")
    save_dataframe(val_df, output_dir / "val_split.csv")
    save_dataframe(test_df, output_dir / "test_split.csv")

    print("\nFULL SPLITS CREATED")
    print("=" * 60)
    print_split_summary("train", train_df)
    print_split_summary("validation", val_df)
    print_split_summary("test", test_df)

    # If images exist locally, also save image-aware splits
    if image_dir.exists():
        train_img_df = add_image_availability(train_df, image_dir=image_dir)
        val_img_df = add_image_availability(val_df, image_dir=image_dir)
        test_img_df = add_image_availability(test_df, image_dir=image_dir)

        save_dataframe(train_img_df, output_dir / "train_split_image_status.csv")
        save_dataframe(val_img_df, output_dir / "val_split_image_status.csv")
        save_dataframe(test_img_df, output_dir / "test_split_image_status.csv")

        train_with_images = train_img_df[train_img_df["image_exists"]].reset_index(drop=True)
        val_with_images = val_img_df[val_img_df["image_exists"]].reset_index(drop=True)
        test_with_images = test_img_df[test_img_df["image_exists"]].reset_index(drop=True)

        save_dataframe(train_with_images, output_dir / "train_split_with_images.csv")
        save_dataframe(val_with_images, output_dir / "val_split_with_images.csv")
        save_dataframe(test_with_images, output_dir / "test_split_with_images.csv")

        print("\nIMAGE-AWARE SPLITS CREATED")
        print("=" * 60)
        print_split_summary("train_with_images", train_img_df)
        print_split_summary("val_with_images", val_img_df)
        print_split_summary("test_with_images", test_img_df)

    print("\nSaved files:")
    print("- data/processed/train_split.csv")
    print("- data/processed/val_split.csv")
    print("- data/processed/test_split.csv")

    if image_dir.exists():
        print("- data/processed/train_split_image_status.csv")
        print("- data/processed/val_split_image_status.csv")
        print("- data/processed/test_split_image_status.csv")
        print("- data/processed/train_split_with_images.csv")
        print("- data/processed/val_split_with_images.csv")
        print("- data/processed/test_split_with_images.csv")