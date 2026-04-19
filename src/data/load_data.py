from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


TRAIN_REQUIRED_COLUMNS = ["sample_id", "catalog_content", "image_link", "price"]
TEST_REQUIRED_COLUMNS = ["sample_id", "catalog_content", "image_link"]


def validate_columns(df: pd.DataFrame, required_columns: list[str], df_name: str = "DataFrame") -> None:
    """
    Check whether the required columns exist in the given DataFrame.

    Args:
        df: Input pandas DataFrame.
        required_columns: List of required column names.
        df_name: Friendly name used in the error message.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{df_name} is missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def clean_text(text: object) -> str:
    """
    Basic text cleaning for catalog content.

    Args:
        text: Raw input value.

    Returns:
        Cleaned string.
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()


def add_basic_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple text-based features to the DataFrame.

    Features:
        - text_length_chars
        - text_length_words

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with added features.
    """
    df = df.copy()

    df["text_length_chars"] = df["catalog_content"].apply(len)
    df["text_length_words"] = df["catalog_content"].apply(lambda x: len(x.split()))

    return df


def basic_cleaning(df: pd.DataFrame, has_target: bool = True) -> pd.DataFrame:
    """
    Perform basic cleaning steps:
        - remove duplicate sample_id rows
        - clean catalog_content
        - standardize image_link
        - convert price to numeric if target exists
        - drop rows with missing required values where necessary

    Args:
        df: Raw input DataFrame.
        has_target: Whether the DataFrame contains the target column (`price`).

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Remove duplicated rows based on sample_id
    if "sample_id" in df.columns:
        df = df.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)

    # Clean text column
    df["catalog_content"] = df["catalog_content"].apply(clean_text)

    # Standardize image_link
    df["image_link"] = df["image_link"].fillna("").astype(str).str.strip()

    # Drop rows with missing essential fields
    df = df[df["sample_id"].notna()].copy()
    df = df[df["catalog_content"] != ""].copy()

    if has_target:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df[df["price"].notna()].copy()
        df = df[df["price"] >= 0].copy()

    df = df.reset_index(drop=True)
    return df


def load_train_data(csv_path: str | Path, add_features: bool = True) -> pd.DataFrame:
    """
    Load and preprocess training data.

    Args:
        csv_path: Path to the training CSV file.
        add_features: Whether to add simple text features.

    Returns:
        Cleaned training DataFrame.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Train file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_columns(df, TRAIN_REQUIRED_COLUMNS, df_name="Train DataFrame")
    df = basic_cleaning(df, has_target=True)

    if add_features:
        df = add_basic_text_features(df)

    return df


def load_test_data(csv_path: str | Path, add_features: bool = True) -> pd.DataFrame:
    """
    Load and preprocess test data.

    Args:
        csv_path: Path to the test CSV file.
        add_features: Whether to add simple text features.

    Returns:
        Cleaned test DataFrame.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Test file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_columns(df, TEST_REQUIRED_COLUMNS, df_name="Test DataFrame")
    df = basic_cleaning(df, has_target=False)

    if add_features:
        df = add_basic_text_features(df)

    return df


def save_dataframe(df: pd.DataFrame, output_path: str | Path, index: bool = False) -> None:
    """
    Save a DataFrame to CSV. Creates parent directories if needed.

    Args:
        df: DataFrame to save.
        output_path: Destination CSV path.
        index: Whether to save row indices.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)


def print_data_summary(df: pd.DataFrame, name: str = "dataset") -> None:
    """
    Print a compact summary about the dataset.

    Args:
        df: Input DataFrame.
        name: Dataset name for printing.
    """
    print(f"\n{name.upper()} SUMMARY")
    print("-" * 50)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nMissing values:")
    print(df.isna().sum())

    if "price" in df.columns:
        print("\nPrice statistics:")
        print(df["price"].describe())

    if "text_length_chars" in df.columns:
        print("\nText length (chars) statistics:")
        print(df["text_length_chars"].describe())

    if "text_length_words" in df.columns:
        print("\nText length (words) statistics:")
        print(df["text_length_words"].describe())


if __name__ == "__main__":
    train_path = Path("data/raw/train.csv")
    test_path = Path("data/raw/test.csv")

    if train_path.exists():
        train_df = load_train_data(train_path)
        print_data_summary(train_df, name="train")
        save_dataframe(train_df, "data/processed/train_clean.csv")

    if test_path.exists():
        test_df = load_test_data(test_path)
        print_data_summary(test_df, name="test")
        save_dataframe(test_df, "data/processed/test_clean.csv")