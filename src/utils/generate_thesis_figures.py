from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.models.multimodal_dataset import get_multimodal_train_transform


FIG_DIR = Path("outputs/figures/thesis")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


# ÁBRA 7 – Augmentáció példák
def fig_07_augmentation_examples(
    image_path: str = "data/images/70.jpg",
    output_path: Path = FIG_DIR / "fig_07_augmentation_examples.png",
):
    image = Image.open(image_path).convert("RGB")
    transform = get_multimodal_train_transform()

    images = [image]
    titles = ["Eredeti kép"]

    for i in range(3):
        aug_img = transform(image)
        aug_img = aug_img.permute(1, 2, 0).numpy()

        # ImageNet normalization visszafordítása közelítőleg
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        aug_img = aug_img * std + mean
        aug_img = aug_img.clip(0, 1)

        images.append(aug_img)
        titles.append(f"Augmentált {i + 1}")

    plt.figure(figsize=(10, 3))

    for i, img in enumerate(images):
        plt.subplot(1, 4, i + 1)
        plt.imshow(img)
        plt.title(titles[i])
        plt.axis("off")

    save_fig(output_path)


def load_metrics_files(metrics_dir: str = "outputs/evaluation") -> pd.DataFrame:
    rows = []

    for path in Path(metrics_dir).glob("*_metrics.csv"):
        df = pd.read_csv(path)
        row = df.iloc[0].to_dict()

        model_file = Path(row["model_path"]).stem
        row["model_name"] = model_file

        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No *_metrics.csv files found in {metrics_dir}")

    return pd.DataFrame(rows)


def prettify_model_name(name: str) -> str:
    name_lower = name.lower()

    if "text" in name_lower:
        return "Text\nDistilBERT"
    if "resnet18" in name_lower and "multimodal" not in name_lower:
        return "Image\nResNet18"
    if "efficientnet" in name_lower and "multimodal" not in name_lower:
        return "Image\nEfficientNet-B0"
    if "multimodal" in name_lower and "resnet18" in name_lower:
        return "Multimodal\nDistilBERT+ResNet18"
    if "multimodal" in name_lower and "efficientnet" in name_lower:
        return "Multimodal\nDistilBERT+EfficientNet"

    return name


# ÁBRA 10 – Modellek összehasonlítása MAE alapján
def fig_10_model_mae_comparison(
    metrics_dir: str = "outputs/evaluation",
    output_path: Path = FIG_DIR / "fig_10_model_mae_comparison.png",
):
    df = load_metrics_files(metrics_dir)
    df["display_name"] = df["model_name"].apply(prettify_model_name)
    df = df.sort_values("mae")

    plt.figure(figsize=(10, 5))
    plt.bar(df["display_name"], df["mae"])

    plt.ylabel("MAE")
    plt.title("Modellek összehasonlítása MAE alapján")
    plt.xticks(rotation=20, ha="right")

    save_fig(output_path)


# ÁBRA 11 – Multimodális architektúra
def fig_11_multimodal_architecture(
    output_path: Path = FIG_DIR / "fig_11_multimodal_architecture.png",
):
    plt.figure(figsize=(11, 5))
    ax = plt.gca()
    ax.axis("off")

    boxes = {
        "Szöveg\n(catalog_content)": (0.05, 0.65),
        "DistilBERT\nszöveges encoder": (0.28, 0.65),
        "Szöveges\nreprezentáció": (0.52, 0.65),
        "Kép\n(termékfotó)": (0.05, 0.25),
        "CNN képi encoder\nResNet/EfficientNet": (0.28, 0.25),
        "Képi\nreprezentáció": (0.52, 0.25),
        "Konkatenáció\n(feature fusion)": (0.72, 0.45),
        "Regressziós fej": (0.88, 0.45),
        "Prediktált ár": (1.04, 0.45),
    }

    for text, (x, y) in boxes.items():
        ax.text(
            x, y, text,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white"),
            fontsize=10,
        )

    arrows = [
        ((0.14, 0.65), (0.22, 0.65)),
        ((0.39, 0.65), (0.47, 0.65)),
        ((0.61, 0.65), (0.67, 0.50)),
        ((0.14, 0.25), (0.22, 0.25)),
        ((0.39, 0.25), (0.47, 0.25)),
        ((0.61, 0.25), (0.67, 0.40)),
        ((0.78, 0.45), (0.83, 0.45)),
        ((0.94, 0.45), (0.99, 0.45)),
    ]

    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.8))

    plt.xlim(0, 1.15)
    plt.ylim(0, 1)
    plt.title("Multimodális modell architektúrája")

    save_fig(output_path)


# ÁBRA 12 – Feature fusion szemléltetés
def fig_12_feature_fusion(
    output_path: Path = FIG_DIR / "fig_12_feature_fusion.png",
):
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.axis("off")

    items = {
        "h_text\nszöveges embedding": (0.12, 0.65),
        "Lineáris projekció\nW_t h_text": (0.35, 0.65),
        "h_image\nképi embedding": (0.12, 0.25),
        "Lineáris projekció\nW_i h_image": (0.35, 0.25),
        "Összefűzés\n[h_text ; h_image]": (0.62, 0.45),
        "Regressziós fej": (0.83, 0.45),
        "log_price": (1.02, 0.45),
    }

    for text, (x, y) in items.items():
        ax.text(
            x, y, text,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white"),
            fontsize=10,
        )

    arrows = [
        ((0.22, 0.65), (0.28, 0.65)),
        ((0.45, 0.65), (0.54, 0.50)),
        ((0.22, 0.25), (0.28, 0.25)),
        ((0.45, 0.25), (0.54, 0.40)),
        ((0.70, 0.45), (0.77, 0.45)),
        ((0.90, 0.45), (0.97, 0.45)),
    ]

    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.8))

    plt.xlim(0, 1.1)
    plt.ylim(0, 1)
    plt.title("Feature-szintű fúzió szemléltetése")

    save_fig(output_path)


# ÁBRA 14 – Modalitások hatása
def fig_14_modality_effect(
    metrics_dir: str = "outputs/evaluation",
    output_path: Path = FIG_DIR / "fig_14_modality_effect.png",
):
    df = load_metrics_files(metrics_dir)

    def modality(row):
        name = row["model_name"].lower()
        if "multimodal" in name:
            return "Multimodális"
        if "text" in name or "distilbert" in name:
            return "Szöveg"
        return "Kép"

    df["modality"] = df.apply(modality, axis=1)

    best_df = (
        df.sort_values("mae")
        .groupby("modality", as_index=False)
        .first()
        .sort_values("mae")
    )

    plt.figure(figsize=(7, 5))
    plt.bar(best_df["modality"], best_df["mae"])
    plt.ylabel("Legjobb MAE")
    plt.title("Modalitások hatása a predikciós hibára")

    save_fig(output_path)


# ÁBRA 15 – Összehasonlító diagram MAE + RMSE
def fig_15_mae_rmse_comparison(
    metrics_dir: str = "outputs/evaluation",
    output_path: Path = FIG_DIR / "fig_15_mae_rmse_comparison.png",
):
    df = load_metrics_files(metrics_dir)
    df["display_name"] = df["model_name"].apply(prettify_model_name)
    df = df.sort_values("mae")

    x = range(len(df))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar([i - width / 2 for i in x], df["mae"], width=width, label="MAE")
    plt.bar([i + width / 2 for i in x], df["rmse"], width=width, label="RMSE")

    plt.xticks(list(x), df["display_name"], rotation=20, ha="right")
    plt.ylabel("Hiba")
    plt.title("Modellek összehasonlítása MAE és RMSE alapján")
    plt.legend()

    save_fig(output_path)


def extract_lr_from_name(name: str):
    match = re.search(r"lr(\d+)e(\d+)", name.lower())
    if not match:
        return None

    base = int(match.group(1))
    exp = int(match.group(2))
    return base * (10 ** (-exp))


# ÁBRA 16 – Learning rate hatása
def fig_16_learning_rate_effect(
    summary_csv: str = "outputs/summary/experiment_summary_all.csv",
    output_path: Path = FIG_DIR / "fig_16_learning_rate_effect.png",
):
    df = pd.read_csv(summary_csv)

    if "lr" not in df.columns or df["lr"].isna().all():
        df["lr"] = df["experiment_name"].apply(extract_lr_from_name)

    df = df.dropna(subset=["lr", "best_val_mae_price"]).copy()

    grouped = (
        df.groupby("lr", as_index=False)["best_val_mae_price"]
        .min()
        .sort_values("lr")
    )

    plt.figure(figsize=(7, 5))
    plt.plot(grouped["lr"].astype(str), grouped["best_val_mae_price"], marker="o")

    plt.xlabel("Learning rate")
    plt.ylabel("Legjobb validációs MAE")
    plt.title("Learning rate hatása a validációs hibára")

    save_fig(output_path)


# ÁBRA 17 – Összegző diagram
def fig_17_summary_diagram(
    metrics_dir: str = "outputs/evaluation",
    output_path: Path = FIG_DIR / "fig_17_summary_diagram.png",
):
    df = load_metrics_files(metrics_dir)

    def group_name(name: str):
        name = name.lower()
        if "multimodal" in name:
            return "Multimodális modell"
        if "text" in name or "distilbert" in name:
            return "Szöveges modell"
        return "Képi modell"

    df["group"] = df["model_name"].apply(group_name)

    best_df = (
        df.sort_values("mae")
        .groupby("group", as_index=False)
        .first()
        .sort_values("mae")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(best_df["group"], best_df["mae"], marker="o", linewidth=2)

    plt.ylabel("Legjobb teszt MAE")
    plt.title("A fő modellcsoportok összegző összehasonlítása")
    plt.xticks(rotation=15, ha="right")

    save_fig(output_path)


if __name__ == "__main__":
    fig_07_augmentation_examples(
        image_path="data/images/70.jpg",
    )

    fig_10_model_mae_comparison()
    fig_11_multimodal_architecture()
    fig_12_feature_fusion()
    fig_14_modality_effect()
    fig_15_mae_rmse_comparison()
    fig_16_learning_rate_effect()
    fig_17_summary_diagram()