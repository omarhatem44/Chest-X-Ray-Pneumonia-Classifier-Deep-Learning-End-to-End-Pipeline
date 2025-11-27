import os
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
from PIL import Image


DATA_DIR = Path(
    r"D:\projects for my CV\Chest X-Ray Images (Pneumonia) DL\Data\chest_xray\chest_xray"
)

def count_images_in_split(split_name: str):
    split_dir = DATA_DIR / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    counts = {}
    for class_name in sorted(os.listdir(split_dir)):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        n_images = sum(
            1
            for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        counts[class_name] = n_images

    return counts


def print_all_counts():
    print("===== IMAGE COUNTS PER SPLIT & CLASS =====")
    for split in ["train", "val", "test"]:
        counts = count_images_in_split(split)
        total = sum(counts.values())
        print(f"\n[{split.upper()}]  total images: {total}")
        for cls, n in counts.items():
            print(f" - {cls:<10}: {n}")


def get_sample_image_paths(split: str, class_name: str, max_images: int = 6):
    class_dir = DATA_DIR / split / class_name
    img_files = [
        class_dir / f
        for f in os.listdir(class_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return img_files[:max_images]


def show_sample_images():
    print("\n[INFO] Plotting sample images from train/NORMAL and train/PNEUMONIA ...")

    normal_paths = get_sample_image_paths("train", "NORMAL", max_images=6)
    pneu_paths = get_sample_image_paths("train", "PNEUMONIA", max_images=6)

    n_normal = len(normal_paths)
    n_pneu = len(pneu_paths)
    n_cols = 6
    n_rows = 2

    plt.figure(figsize=(18, 6))

    for i, img_path in enumerate(normal_paths):
        img = Image.open(img_path)
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title("NORMAL")
        plt.axis("off")

    for i, img_path in enumerate(pneu_paths):
        img = Image.open(img_path)
        plt.subplot(n_rows, n_cols, n_cols + i + 1)
        plt.imshow(img, cmap="gray")
        plt.title("PNEUMONIA")
        plt.axis("off")

    plt.suptitle("Sample Chest X-Ray Images (train set)", fontsize=16)
    plt.tight_layout()
    plt.show()


def analyze_image_sizes(split: str = "train"):
    print(f"\n[INFO] Analyzing image sizes in '{split}' split...")
    split_dir = DATA_DIR / split
    size_counter = Counter()

    for class_name in os.listdir(split_dir):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue

        for f in os.listdir(class_dir):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = class_dir / f
            try:
                with Image.open(img_path) as img:
                    size_counter[img.size] += 1  # (width, height)
            except Exception:
                continue

    print("Most common image sizes (width x height):")
    for (w, h), cnt in size_counter.most_common(10):
        print(f" - {w} x {h}: {cnt} images")


def main():
    print(f"[INFO] Using data directory: {DATA_DIR}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    print_all_counts()

    analyze_image_sizes(split="train")

    show_sample_images()

    print("\n[INFO] EDA finished.")


if __name__ == "__main__":
    main()
