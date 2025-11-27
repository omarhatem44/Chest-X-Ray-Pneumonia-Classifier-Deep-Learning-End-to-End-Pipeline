import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATA_DIR = Path(
    r"D:\projects for my CV\Chest X-Ray Images (Pneumonia) DL\Data\chest_xray\chest_xray\train"
)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_normal_aug_gen():
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode="nearest"
    ).flow_from_directory(
        DATA_DIR,
        classes=["NORMAL"],
        target_size=IMG_SIZE,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
    )


def build_pneumonia_gen():
    return ImageDataGenerator(
        rescale=1./255
    ).flow_from_directory(
        DATA_DIR,
        classes=["PNEUMONIA"],
        target_size=IMG_SIZE,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
    )


def combine_generators(gen_normal, gen_pneu):
    while True:
        batch_norm_imgs, batch_norm_labels = next(gen_normal)
        batch_pneu_imgs, batch_pneu_labels = next(gen_pneu)

        
        images = np.concatenate([batch_norm_imgs, batch_pneu_imgs])
        labels = np.concatenate([batch_norm_labels, batch_pneu_labels])

        idx = np.random.permutation(len(images))
        yield images[idx], labels[idx]


def preview_aug(gen_normal):
    imgs, labels = next(gen_normal)

    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(imgs[i])
        plt.title("NORMAL (augmented)")
        plt.axis("off")
    plt.suptitle("Augmented NORMAL Samples", fontsize=16)
    plt.show()


def main():
    print("[INFO] Building generators...")

    normal_aug_gen = build_normal_aug_gen()
    pneu_gen = build_pneumonia_gen()

    print("[INFO] Previewing NORMAL augmentation...")
    preview_aug(normal_aug_gen)

    print("[INFO] Combined generator is ready for training!")
    combined_gen = combine_generators(normal_aug_gen, pneu_gen)

    imgs, labels = next(combined_gen)
    print(f"[INFO] Combined batch → images: {imgs.shape}, labels: {labels.shape}")


if __name__ == "__main__":
    main()
