from pathlib import Path

import pandas as pd


def check_value_unit_features(df: pd.DataFrame):
    print("\n=== VALUE / UNIT CHECK ===")

    total = len(df)
    has_value_unit = df["has_explicit_value_unit"].sum()

    print(f"Total rows: {total}")
    print(f"Rows with extracted value+unit: {has_value_unit}")
    print(f"Coverage: {has_value_unit / total:.2%}")

    print("\nTop units:")
    print(df["unit_extracted"].value_counts().head(10))

    print("\nSample rows with extracted values:")
    print(
        df[df["has_explicit_value_unit"] == 1][
            ["catalog_content", "value_extracted", "unit_extracted"]
        ].head(5)
    )

    print("\nSample rows WITHOUT extracted values:")
    print(
        df[df["has_explicit_value_unit"] == 0][
            ["catalog_content"]
        ].head(5)
    )


def check_ipq(df: pd.DataFrame):
    print("\n=== IPQ CHECK ===")

    print("IPQ stats:")
    print(df["ipq"].describe())

    print("\nSample IPQ values:")
    print(
        df[["catalog_content", "ipq"]]
        .sample(5, random_state=42)
    )


if __name__ == "__main__":
    path = Path("data/processed/train_preprocessed.csv")

    if not path.exists():
        raise FileNotFoundError("Run preprocess first!")

    df = pd.read_csv(path)

    check_value_unit_features(df)
    check_ipq(df)