from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_pred_vs_true_better(predictions_csv: str):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(predictions_csv)

    y_true = df["target_price"]
    y_pred = df["pred_price"]

    # ===== 1. LOG SCALE =====
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.25, s=10)

    plt.xscale("log")
    plt.yscale("log")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Valós ár (log)")
    plt.ylabel("Prediktált ár (log)")
    plt.title("Predikció vs valós érték (log skála)")
    plt.tight_layout()
    plt.savefig("outputs/figures/fig_05a_log.png", dpi=300)
    plt.close()

    # ===== 2. ZOOM (outlier nélkül) =====
    upper = y_true.quantile(0.99)
    mask = y_true <= upper

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true[mask], y_pred[mask], alpha=0.25, s=10)

    plt.plot([0, upper], [0, upper], linestyle="--")

    plt.xlabel("Valós ár")
    plt.ylabel("Prediktált ár")
    plt.title("Predikció vs valós érték (zoomolt)")
    plt.tight_layout()
    plt.savefig("outputs/figures/fig_05b_zoom.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_pred_vs_true_better(
        predictions_csv="outputs/evaluation/text_distilbert_lr2e5_len64_ep5__on__test_split_with_images_predictions.csv"
    )