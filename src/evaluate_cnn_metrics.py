import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ===================== PATHS & CONFIG =====================

BASE_DIR = Path(
    r"D:\projects for my CV\Chest X-Ray Images (Pneumonia) DL\Data\chest_xray\chest_xray"
)
TEST_DIR = BASE_DIR / "test"

IMG_SIZE = (224, 224)
TEST_BATCH_SIZE = 32

MODEL_PATH = Path(
    r"D:\projects for my CV\Chest X-Ray Images (Pneumonia) DL\my model\cnn_pneumonia_final_v2.h5"
)

print("[INFO] BASE_DIR:", BASE_DIR)
print("[INFO] MODEL_PATH:", MODEL_PATH)


# ===================== BUILD TEST GENERATOR =====================

def build_test_gen():
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    gen = datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode="rgb",
        batch_size=TEST_BATCH_SIZE,
        class_mode="binary",
        shuffle=False  
    )
    return gen


# ===================== PLOT CONFUSION MATRIX =====================

def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (CNN)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Confusion matrix saved to: {save_path}")
    plt.show()


# ===================== MAIN =====================

def main():
    # 1) Load model
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded.")

    # 2) Build test generator
    print("[INFO] Building test generator...")
    test_gen = build_test_gen()
    class_indices = test_gen.class_indices
    print("[INFO] class_indices (test):", class_indices)

    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[0], idx_to_class[1]]

    # 3) Get predictions
    steps = math.ceil(test_gen.n / TEST_BATCH_SIZE)
    print(f"[INFO] test samples = {test_gen.n}, steps = {steps}")

    print("[INFO] Predicting on test set...")
    y_prob = model.predict(test_gen, steps=steps)
    y_pred = (y_prob >= 0.5).astype(int).flatten()
    y_true = test_gen.classes  # shape (n_samples,)

    print("[INFO] Shapes → y_true:", y_true.shape, ", y_pred:", y_pred.shape)

    # 4) Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n===== Confusion Matrix =====")
    print(cm)

    # 5) Classification report (Precision / Recall / F1)
    print("\n===== Classification Report =====")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4
        )
    )

    # 6) Plot confusion matrix
    save_path = (
        r"D:\projects for my CV\Chest X-Ray Images (Pneumonia) DL\my model\confusion_matrix_cnn.png"
    )
    plot_confusion_matrix(cm, class_names, save_path=save_path)

    print("\n[INFO] Evaluation finished.")


if __name__ == "__main__":
    main()
