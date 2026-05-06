from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_error_distribution(
    predictions_csv: str,
    output_path: str = "outputs/figures/fig_06_error_distribution.png",
    clip_outliers: bool = True,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
):
    df = pd.read_csv(predictions_csv)

    required_columns = {"target_price", "pred_price"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Hiányzó oszlop(ok): {missing_columns}. "
            f"A CSV-ben szükséges: target_price, pred_price"
        )

    df = df.copy()

    # Hiba: pozitív érték = túlbecslés, negatív érték = alulbecslés
    df["error"] = df["pred_price"] - df["target_price"]
    df["absolute_error"] = df["error"].abs()

    plot_df = df.copy()

    if clip_outliers:
        lower = plot_df["error"].quantile(lower_quantile)
        upper = plot_df["error"].quantile(upper_quantile)
        plot_df = plot_df[
            (plot_df["error"] >= lower) &
            (plot_df["error"] <= upper)
        ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))

    plt.hist(
        plot_df["error"],
        bins=70,
        alpha=0.85,
    )

    plt.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=2.5,
        label="Nincs hiba (0)"
    )

    mean_error = df["error"].mean()
    median_error = df["error"].median()

    plt.axvline(
        mean_error,
        linestyle="-",
        linewidth=2,
        label=f"Átlagos hiba: {mean_error:.2f}"
    )

    plt.axvline(
        median_error,
        linestyle=":",
        linewidth=2,
        label=f"Medián hiba: {median_error:.2f}"
    )

    plt.xlabel("Predikciós hiba (prediktált ár - valós ár)")
    plt.ylabel("Gyakoriság")
    plt.title("Predikciós hibák eloszlása")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Ábra mentve ide: {output_path}")

    print("\nHiba statisztikák:")
    print(f"Minták száma: {len(df)}")
    print(f"Átlagos hiba: {mean_error:.4f}")
    print(f"Medián hiba: {median_error:.4f}")
    print(f"Átlagos abszolút hiba (MAE): {df['absolute_error'].mean():.4f}")
    print(f"Hiba szórása: {df['error'].std():.4f}")

    if clip_outliers:
        print(
            f"\nMegjelenítéshez levágott tartomány: "
            f"{lower_quantile:.0%}–{upper_quantile:.0%} kvantilis"
        )


if __name__ == "__main__":
    plot_error_distribution(
        predictions_csv="outputs/evaluation/text_distilbert_lr2e5_len64_ep5__on__test_split_with_images_predictions.csv",
        output_path="outputs/figures/fig_06_error_distribution_text.png",
        clip_outliers=True,
    )