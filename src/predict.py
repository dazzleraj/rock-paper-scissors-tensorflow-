"""Single image inference script for Rock-Paper-Scissors model."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf

# Ensure project root is in sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows PowerShell UTF-8 console support for emojis
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.data import IMAGE_SIZE, CLASS_EMOJIS

MODELS_DIR = PROJECT_ROOT / "models"


def load_classifier(
    model_path: Path = MODELS_DIR / "rps_cnn.keras",
    classes_path: Path = MODELS_DIR / "class_indices.json"
) -> Tuple[tf.keras.Model, list]:
    """Loads model and class names."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first via train.py.")

    model = tf.keras.models.load_model(model_path)

    if classes_path.exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
            class_names = list(class_indices.keys())
    else:
        class_names = ["paper", "rock", "scissors"]

    return model, class_names


def predict_image(
    image_path: str,
    model: tf.keras.Model = None,
    class_names: list = None
) -> Dict:
    """Runs prediction on a single image file."""
    if model is None or class_names is None:
        model, class_names = load_classifier()

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    img_tensor = np.expand_dims(img_array, axis=0)  # Shape (1, 128, 128, 3)

    predictions = model.predict(img_tensor, verbose=0)[0]
    best_idx = int(np.argmax(predictions))
    best_class = class_names[best_idx]
    best_confidence = float(predictions[best_idx])
    emoji = CLASS_EMOJIS.get(best_class, "")

    all_scores = {
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    }

    return {
        "class": best_class,
        "emoji": emoji,
        "confidence": best_confidence,
        "scores": all_scores
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Rock-Paper-Scissors gesture on an image")
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    args = parser.parse_args()

    result = predict_image(args.image_path)
    print("\n" + "=" * 40)
    print(f"  Prediction: {result['emoji']}  {result['class'].upper()}")
    print(f"  Confidence: {result['confidence'] * 100:.2f}%")
    print("=" * 40)
    print("Class Probabilities:")
    for cls_name, score in result["scores"].items():
        emoji = CLASS_EMOJIS.get(cls_name, "")
        bar = "█" * int(score * 25) + "░" * (25 - int(score * 25))
        print(f"  {emoji} {cls_name:<10}: {bar} {score * 100:6.2f}%")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
