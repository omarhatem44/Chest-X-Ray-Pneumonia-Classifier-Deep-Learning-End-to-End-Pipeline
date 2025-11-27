import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ===================== PATHS & CONFIG =====================

BASE_DIR = Path(
    r"D:\projects for my CV\Chest X-Ray Images (Pneumonia) DL\Data\chest_xray\chest_xray"
)
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
TEST_DIR = BASE_DIR / "test"

IMG_SIZE = (224, 224)
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

print("[INFO] BASE_DIR:", BASE_DIR)


# ===================== GENERATORS =====================

def build_train_gen():
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode="nearest",
    )
    gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode="rgb",
        batch_size=TRAIN_BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )
    return gen


def build_val_gen():
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    gen = datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        color_mode="rgb",
        batch_size=VAL_BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    return gen


def build_test_gen():
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    gen = datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode="rgb",
        batch_size=TEST_BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    return gen


# ===================== MODEL (CNN BASELINE) =====================

def build_cnn_model(input_shape=(224, 224, 3)):
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ===================== TRAINING =====================

def main():
    print("[INFO] Building generators...")

    train_gen = build_train_gen()
    val_gen = build_val_gen()
    test_gen = build_test_gen()

    print("[INFO] Class indices (train):", train_gen.class_indices)
    print("[INFO] Class indices (val):", val_gen.class_indices)

    steps_per_epoch = math.ceil(train_gen.n / TRAIN_BATCH_SIZE)
    val_steps = math.ceil(val_gen.n / VAL_BATCH_SIZE)
    test_steps = math.ceil(test_gen.n / TEST_BATCH_SIZE)

    print(f"[INFO] train_count = {train_gen.n}")
    print(f"[INFO] val_count   = {val_gen.n}")
    print(f"[INFO] test_count  = {test_gen.n}")
    print(f"[INFO] steps_per_epoch = {steps_per_epoch}")
    print(f"[INFO] val_steps = {val_steps}, test_steps = {test_steps}")

    labels = train_gen.classes  
    normal_count = np.sum(labels == 0)
    pneu_count = np.sum(labels == 1)
    total = normal_count + pneu_count

    w_normal = total / (2.0 * normal_count)
    w_pneu = total / (2.0 * pneu_count)
    class_weight = {0: w_normal, 1: w_pneu}

    print(f"[INFO] normal_count = {normal_count}, pneumonia_count = {pneu_count}")
    print(f"[INFO] class_weight = {class_weight}")

    model = build_cnn_model(input_shape=(224, 224, 3))
    model.summary()

    # Callbacks
    checkpoint_path = (
        r"D:\projects for my CV\Chest X-Ray Images (Pneumonia) DL\cnn_pneumonia_best.h5"
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # ====== TRAIN ======
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=20,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # ====== EVALUATE on test ======
    print("\n[INFO] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen, steps=test_steps)
    print(f"[RESULT] Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")

    final_path = (
        r"D:\projects for my CV\Chest X-Ray Images (Pneumonia) DL\cnn_pneumonia_final_v2.h5"
    )
    model.save(final_path)
    print(f"[INFO] Saved final model as {final_path}")


if __name__ == "__main__":
    main()
