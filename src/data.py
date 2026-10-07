"""Data loading, downloading, and preprocessing utilities for Rock-Paper-Scissors."""

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple, List, Any

# Standard URLs for the Laurence Moroney / TensorFlow RPS dataset
RPS_TRAIN_URL = "https://storage.googleapis.com/learning-datasets/rps.zip"
RPS_TEST_URL = "https://storage.googleapis.com/learning-datasets/rps-test-set.zip"

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
DEFAULT_TEST_DIR = Path(__file__).resolve().parent.parent / "dataset_test"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
CLASS_NAMES = ["paper", "rock", "scissors"]
CLASS_EMOJIS = {"paper": "✋", "rock": "✊", "scissors": "✌️"}


def download_and_extract_dataset(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    test_dir: Path = DEFAULT_TEST_DIR,
    force_download: bool = False
) -> Tuple[Path, Path]:
    """Downloads and extracts the RPS training and test datasets if not already present."""
    dataset_dir = Path(dataset_dir)
    test_dir = Path(test_dir)
    
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if train dataset exists
    train_rock = dataset_dir / "rock"
    train_paper = dataset_dir / "paper"
    train_scissors = dataset_dir / "scissors"
    
    if force_download or not (train_rock.exists() and train_paper.exists() and train_scissors.exists()):
        print("Downloading Rock-Paper-Scissors training dataset...")
        zip_path = dataset_dir.parent / "rps_train.zip"
        urllib.request.urlretrieve(RPS_TRAIN_URL, zip_path)
        
        print("Extracting training dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir.parent / "temp_train")
            
        temp_extracted = dataset_dir.parent / "temp_train" / "rps"
        if temp_extracted.exists():
            if dataset_dir.exists():
                import shutil
                shutil.rmtree(dataset_dir)
            temp_extracted.rename(dataset_dir)
            import shutil
            shutil.rmtree(dataset_dir.parent / "temp_train", ignore_errors=True)
        
        if zip_path.exists():
            zip_path.unlink()
        print(f"Training dataset ready at: {dataset_dir}")
    else:
        print(f"Training dataset already present at: {dataset_dir}")

    # Check if test dataset exists
    test_rock = test_dir / "rock"
    test_paper = test_dir / "paper"
    test_scissors = test_dir / "scissors"
    
    if force_download or not (test_rock.exists() and test_paper.exists() and test_scissors.exists()):
        print("Downloading Rock-Paper-Scissors test dataset...")
        test_zip_path = test_dir.parent / "rps_test.zip"
        urllib.request.urlretrieve(RPS_TEST_URL, test_zip_path)
        
        print("Extracting test dataset...")
        with zipfile.ZipFile(test_zip_path, "r") as zip_ref:
            zip_ref.extractall(test_dir.parent / "temp_test")
            
        temp_extracted_test = test_dir.parent / "temp_test" / "rps-test-set"
        if temp_extracted_test.exists():
            if test_dir.exists():
                import shutil
                shutil.rmtree(test_dir)
            temp_extracted_test.rename(test_dir)
            import shutil
            shutil.rmtree(test_dir.parent / "temp_test", ignore_errors=True)
            
        if test_zip_path.exists():
            test_zip_path.unlink()
        print(f"Test dataset ready at: {test_dir}")
    else:
        print(f"Test dataset already present at: {test_dir}")

    return dataset_dir, test_dir


def create_data_augmentation() -> Any:
    """Returns a data augmentation sequential layer."""
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
    ], name="data_augmentation")


def load_datasets(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    val_split: float = 0.2,
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    seed: int = 42
) -> Tuple[Any, Any, List[str]]:
    """Loads training and validation datasets from directory."""
    import tensorflow as tf
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        download_and_extract_dataset(dataset_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def load_test_dataset(
    test_dir: Path = DEFAULT_TEST_DIR,
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE
) -> Tuple[Any, List[str]]:
    """Loads test dataset from directory."""
    import tensorflow as tf
    test_dir = Path(test_dir)
    if not test_dir.exists():
        download_and_extract_dataset(test_dir=test_dir)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )
    
    return test_ds.prefetch(buffer_size=tf.data.AUTOTUNE), test_ds.class_names


if __name__ == "__main__":
    print("Testing dataset downloader...")
    train_p, test_p = download_and_extract_dataset()
    print(f"Train directory: {train_p}")
    print(f"Test directory: {test_p}")
