import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("C:\\temp\diplomaCode\multimodal-price-detection\outputs\evaluation\\text_distilbert_lr2e5_len64_ep5__on__test_split_with_images_predictions.csv")

    plt.figure(figsize=(6,6))
    plt.scatter(df["true"], df["pred"], alpha=0.3)
    plt.xlabel("Valós ár")
    plt.ylabel("Prediktált ár")
    plt.title("Predikció vs valós érték")

# ideális egyenes
    plt.plot([df["true"].min(), df["true"].max()],
         [df["true"].min(), df["true"].max()],
         color="red")

    plt.savefig("outputs/figures/pred_vs_true.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()