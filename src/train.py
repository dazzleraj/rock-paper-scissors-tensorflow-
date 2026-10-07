import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# Ensure project root is in sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import download_and_extract_dataset, load_datasets, IMAGE_SIZE, BATCH_SIZE
from src.model import build_rps_model

MODELS_DIR = PROJECT_ROOT / "models"


def plot_history(history, save_path: Path):
    """Plots and saves the training history curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, "b-o", label="Training Accuracy")
    plt.plot(epochs_range, val_acc, "r--s", label="Validation Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, "b-o", label="Training Loss")
    plt.plot(epochs_range, val_loss, "r--s", label="Validation Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training history saved to: {save_path}")


def train(
    epochs: int = 25,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = 1e-3,
    models_dir: Path = MODELS_DIR
):
    """Trains the CNN model and saves the best checkpoint."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = models_dir / "rps_cnn.keras"
    classes_save_path = models_dir / "class_indices.json"
    history_save_path = models_dir / "training_history.png"

    print("Step 1: Preparing datasets...")
    train_path, test_path = download_and_extract_dataset()
    train_ds, val_ds, class_names = load_datasets(train_path, batch_size=batch_size, image_size=IMAGE_SIZE)

    print(f"Classes found ({len(class_names)}): {class_names}")
    # Save class indices mapping
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    with open(classes_save_path, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Class indices saved to: {classes_save_path}")

    print("\nStep 2: Building model architecture...")
    model = build_rps_model(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=len(class_names),
        include_augmentation=True
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    print("\nStep 3: Setting up training callbacks...")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\nStep 4: Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    print("\nStep 5: Saving history plot...")
    plot_history(history, history_save_path)

    # Save final model if checkpoint didn't trigger
    if not model_save_path.exists():
        model.save(model_save_path)
    print(f"\n✅ Training complete! Best model saved to: {model_save_path}")
    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Rock-Paper-Scissors CNN Model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
