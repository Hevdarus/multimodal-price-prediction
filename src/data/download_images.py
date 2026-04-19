from __future__ import annotations

from pathlib import Path
from io import BytesIO

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


def download_image(url: str, save_path: Path, timeout: int = 10) -> bool:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path, format="JPEG", quality=95)
        return True
    except Exception:
        return False


def download_images_from_csv(
    csv_path: str | Path,
    image_dir: str | Path = "data/images",
    limit: int | None = None,
) -> None:
    csv_path = Path(csv_path)
    image_dir = Path(image_dir)

    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit).copy()

    success_count = 0
    fail_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        sample_id = row["sample_id"]
        image_link = str(row["image_link"]).strip()

        save_path = image_dir / f"{sample_id}.jpg"

        if save_path.exists():
            success_count += 1
            continue

        ok = download_image(image_link, save_path)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print(f"\nDownloaded/already exists: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    download_images_from_csv("data/raw/train.csv", image_dir="data/images", limit=10000)