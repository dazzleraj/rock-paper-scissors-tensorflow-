"""CNN Model Architecture for Rock-Paper-Scissors classification."""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_rps_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 3,
    include_augmentation: bool = True
) -> tf.keras.Model:
    """
    Builds the Rock-Paper-Scissors CNN model.

    Architecture:
      - [Optional] Data Augmentation (RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomBrightness)
      - Rescaling (1./255)
      - Conv2D (32, 3x3, ReLU) + MaxPooling2D (2x2)
      - Conv2D (64, 3x3, ReLU) + MaxPooling2D (2x2)
      - Conv2D (128, 3x3, ReLU) + MaxPooling2D (2x2)
      - Conv2D (128, 3x3, ReLU) + MaxPooling2D (2x2)
      - Flatten
      - Dropout (0.5)
      - Dense (512, ReLU)
      - Dense (num_classes, Softmax)
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = inputs

    # Data augmentation for robust training
    if include_augmentation:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.15)(x)
        x = layers.RandomZoom(0.15)(x)
        x = layers.RandomTranslation(height_factor=0.1, width_factor=0.1)(x)
        x = layers.RandomBrightness(0.1)(x)

    # Normalize inputs to [0, 1]
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

    # Block 1
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification Head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    x = layers.Dense(512, activation="relu", name="dense1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="rps_cnn_model")
    return model


if __name__ == "__main__":
    model = build_rps_model()
    model.summary()

