"""Model evaluation script on test dataset with confusion matrix generation."""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Ensure project root is in sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows PowerShell UTF-8 console support
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.data import load_test_dataset, IMAGE_SIZE, CLASS_EMOJIS

MODELS_DIR = PROJECT_ROOT / "models"


def evaluate(
    model_path: Path = MODELS_DIR / "rps_cnn.keras",
    classes_path: Path = MODELS_DIR / "class_indices.json"
):
    """Evaluates the trained model on test dataset."""
    model_path = Path(model_path)
    classes_path = Path(classes_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    if classes_path.exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
            class_names = list(class_indices.keys())
    else:
        class_names = ["paper", "rock", "scissors"]

    print("Loading test dataset...")
    test_ds, _ = load_test_dataset(image_size=IMAGE_SIZE)

    print("Evaluating model performance...")
    results = model.evaluate(test_ds, verbose=1)
    print(f"\n📊 Test Loss: {results[0]:.4f}")
    print(f"📊 Test Accuracy: {results[1] * 100:.2f}%\n")

    # Collect predictions and ground truth labels
    all_preds = []
    all_labels = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    y_pred_probs = np.vstack(all_preds)
    y_true_onehot = np.vstack(all_labels)

    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_true_onehot, axis=1)

    # Classification Report
    display_names = [f"{CLASS_EMOJIS.get(name, '')} {name.capitalize()}" for name in class_names]
    plot_labels = [name.capitalize() for name in class_names]
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=display_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_plot_path = MODELS_DIR / "confusion_matrix.png"

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Rock-Paper-Scissors Confusion Matrix", fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, plot_labels, fontsize=11)
    plt.yticks(tick_marks, plot_labels, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
                fontweight="bold"
            )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {cm_plot_path}")


if __name__ == "__main__":
    evaluate()
