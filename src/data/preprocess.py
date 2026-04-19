from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


IPQ_PATTERNS = [
    r"(\d+)\s?(pack|pk|pcs|pieces|count|ct)\b",
    r"pack of\s?(\d+)",
    r"set of\s?(\d+)",
    r"(\d+)\s?x\s?\d+",
]

UNIT_ALIASES = {
    "fl oz": "fl_oz",
    "fluid ounce": "fl_oz",
    "fluid ounces": "fl_oz",
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "ml": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "millilitre": "ml",
    "millilitres": "ml",
    "l": "l",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "litres": "l",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "mg": "mg",
    "mcg": "mcg",
    "count": "count",
    "ct": "count",
    "pcs": "pcs",
    "pieces": "pcs",
    "piece": "pcs",
    "pack": "pack",
    "packs": "pack",
    "none" : "none",
    "nan" : "none"
}


def normalize_text(text: object) -> str:
    """
    Basic normalization for product text.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()


def normalize_unit(unit: str) -> str:
    """
    Normalize extracted unit names into a smaller canonical set.
    """
    unit = normalize_text(unit)
    unit = unit.replace(".", "").strip()

    if unit in UNIT_ALIASES:
        return UNIT_ALIASES[unit]

    return unit if unit else "unknown"


def extract_ipq(text: str) -> int:
    """
    Extract Item Pack Quantity (IPQ) or quantity-like value from product text.

    Fallback value is 1 if no pattern is found.
    """
    text = normalize_text(text)

    for pattern in IPQ_PATTERNS:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            for group in groups:
                if group and str(group).isdigit():
                    value = int(group)
                    if value > 0:
                        return value

    return 1


def extract_numbers(text: str) -> list[float]:
    """
    Extract all numeric values from text.
    """
    text = normalize_text(text)
    matches = re.findall(r"\d+(?:[\.,]\d+)?", text)

    numbers = []
    for match in matches:
        try:
            numbers.append(float(match.replace(",", ".")))
        except ValueError:
            continue
    return numbers


def has_quantity_pattern(text: str) -> int:
    """
    Binary feature indicating whether quantity-related patterns are present.
    """
    text = normalize_text(text)
    quantity_keywords = [
        "pack", "pcs", "pieces", "count", "ct", "set", "x",
        "ml", "l", "kg", "g", "oz", "fl oz", "lb", "value", "unit"
    ]
    return int(any(keyword in text for keyword in quantity_keywords))


def extract_value_unit(text: str) -> tuple[float, str]:
    """
    Extract 'Value' and 'Unit' fields from catalog content.

    Example:
        'Item Name: Organic Vinegar; Apple Cider Value: 102.0 Unit: Fl Oz'
        -> (102.0, 'fl_oz')

    If not found, returns (0.0, 'unknown').
    """
    text = normalize_text(text)

    # Primary pattern: explicit "Value: ... Unit: ..."
    pattern_1 = r"value:\s*([\d]+(?:[\.,]\d+)?)\s*unit:\s*([a-zA-Z ]+?)(?=$|[;,\|])"
    match_1 = re.search(pattern_1, text)
    if match_1:
        value_str = match_1.group(1)
        unit_str = match_1.group(2)

        try:
            value = float(value_str.replace(",", "."))
        except ValueError:
            value = 0.0

        unit = normalize_unit(unit_str)
        return value, unit

    # Fallback pattern: "<number> <unit>"
    pattern_2 = r"([\d]+(?:[\.,]\d+)?)\s*(fl oz|oz|ml|l|kg|g|lb|mg|mcg|pcs|pieces|count|ct)\b"
    match_2 = re.search(pattern_2, text)
    if match_2:
        value_str = match_2.group(1)
        unit_str = match_2.group(2)

        try:
            value = float(value_str.replace(",", "."))
        except ValueError:
            value = 0.0

        unit = normalize_unit(unit_str)
        return value, unit

    return 0.0, "unknown"


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple engineered text features.
    """
    df = df.copy()

    df["catalog_content"] = df["catalog_content"].apply(normalize_text)

    df["text_length_chars"] = df["catalog_content"].apply(len)
    df["text_length_words"] = df["catalog_content"].apply(lambda x: len(x.split()))
    df["digit_count"] = df["catalog_content"].apply(lambda x: sum(char.isdigit() for char in x))
    df["special_char_count"] = df["catalog_content"].apply(
        lambda x: sum(not char.isalnum() and not char.isspace() for char in x)
    )

    df["ipq"] = df["catalog_content"].apply(extract_ipq)
    df["has_quantity_pattern"] = df["catalog_content"].apply(has_quantity_pattern)
    df["num_numeric_tokens"] = df["catalog_content"].apply(lambda x: len(extract_numbers(x)))
    df["max_numeric_value"] = df["catalog_content"].apply(
        lambda x: max(extract_numbers(x)) if extract_numbers(x) else 0.0
    )

    value_unit_pairs = df["catalog_content"].apply(extract_value_unit)
    df["value_extracted"] = value_unit_pairs.apply(lambda x: x[0])
    df["unit_extracted"] = value_unit_pairs.apply(lambda x: x[1])

    df["has_explicit_value_unit"] = (
        (df["value_extracted"] > 0) & (df["unit_extracted"] != "unknown")
    ).astype(int)

    return df


def add_price_features(df: pd.DataFrame, log_target: bool = True) -> pd.DataFrame:
    """
    Add target-related helper columns for training data.
    """
    df = df.copy()

    if "price" not in df.columns:
        return df

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].notna()].copy()
    df = df[df["price"] >= 0].copy()

    if log_target:
        df["log_price"] = np.log1p(df["price"])

    return df


def create_price_bins(
    df: pd.DataFrame,
    target_col: str = "price",
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Create quantile-based bins for stratified split.
    """
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")

    df["price_bin"] = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates="drop")

    return df


def preprocess_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for train data.
    """
    df = df.copy()
    df = add_text_features(df)
    df = add_price_features(df, log_target=True)
    df = create_price_bins(df, target_col="price", n_bins=10)
    df = df.reset_index(drop=True)
    return df


def preprocess_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for test data.
    """
    df = df.copy()
    df = add_text_features(df)
    df = df.reset_index(drop=True)
    return df


def train_val_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split preprocessed training data into train and validation sets.

    If stratify=True, price_bin is used for stratification.
    """
    df = df.copy()

    if stratify:
        if "price_bin" not in df.columns:
            raise ValueError("price_bin column is missing. Run preprocessing before splitting.")

        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=random_state,
            stratify=df["price_bin"],
        )
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=random_state,
            shuffle=True,
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save DataFrame to CSV, creating parent directories if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_preprocess_summary(df: pd.DataFrame, name: str = "dataset") -> None:
    """
    Print summary of engineered features.
    """
    print(f"\n{name.upper()} PREPROCESS SUMMARY")
    print("-" * 60)
    print(f"Shape: {df.shape}")

    feature_cols = [
        "text_length_chars",
        "text_length_words",
        "digit_count",
        "special_char_count",
        "ipq",
        "has_quantity_pattern",
        "num_numeric_tokens",
        "max_numeric_value",
        "value_extracted",
        "unit_extracted",
        "has_explicit_value_unit",
    ]

    available_cols = [col for col in feature_cols if col in df.columns]

    if available_cols:
        numeric_cols = [col for col in available_cols if df[col].dtype != "object"]
        object_cols = [col for col in available_cols if df[col].dtype == "object"]

        if numeric_cols:
            print("\nNumeric engineered features:")
            print(df[numeric_cols].describe())

        if object_cols:
            for col in object_cols:
                print(f"\nTop values for {col}:")
                print(df[col].value_counts(dropna=False).head(10))

    if "price" in df.columns:
        print("\nPrice statistics:")
        print(df["price"].describe())

    if "log_price" in df.columns:
        print("\nLog-price statistics:")
        print(df["log_price"].describe())

    if "price_bin" in df.columns:
        print("\nPrice bin distribution:")
        print(df["price_bin"].value_counts().sort_index())


if __name__ == "__main__":
    from src.data.load_data import load_test_data, load_train_data

    train_path = Path("data/raw/train.csv")
    test_path = Path("data/raw/test.csv")

    if train_path.exists():
        train_df = load_train_data(train_path, add_features=False)
        train_df = preprocess_train_data(train_df)
        train_split_df, val_split_df = train_val_split(train_df, val_size=0.2, stratify=True)

        print_preprocess_summary(train_df, "train_full")
        print(f"\nTrain split shape: {train_split_df.shape}")
        print(f"Validation split shape: {val_split_df.shape}")

        save_dataframe(train_df, "data/processed/train_preprocessed.csv")
        save_dataframe(train_split_df, "data/processed/train_split.csv")
        save_dataframe(val_split_df, "data/processed/val_split.csv")

    if test_path.exists():
        test_df = load_test_data(test_path, add_features=False)
        test_df = preprocess_test_data(test_df)

        print_preprocess_summary(test_df, "test")
        save_dataframe(test_df, "data/processed/test_preprocessed.csv")