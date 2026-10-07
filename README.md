# ✊✋✌️ Rock-Paper-Scissors CNN Classifier & Real-Time Webcam Detector

A Computer Vision deep learning project using **TensorFlow / Keras**, **OpenCV**, and **Streamlit** to classify hand gestures (Rock, Paper, Scissors) in real-time.

---

## 🌟 Features

* **3-Stage Convolutional Neural Network (CNN)** with data augmentation (random flips, rotations, zooming, translations) and batch normalization for robust generalization.
* **Automated Dataset Pipeline**: One-click download, extraction, and validation split of the Laurence Moroney / TensorFlow Rock-Paper-Scissors dataset (2,892+ images).
* **Real-Time OpenCV Webcam Feed**:
  * Interactive Region-of-Interest (ROI) bounding box.
  * Real-time HUD overlays with confidence meters, gesture labels, and FPS counter.
  * Keyboard controls to pause, resume, and save snapshots.
* **Interactive Streamlit Web Application**:
  * Live camera input directly in the browser (`st.camera_input`).
  * File uploader to test custom photos.
  * Interactive **Rock-Paper-Scissors Game vs AI** with live score tracking.
  * Interactive architecture diagram, training history curves, and confusion matrix.
* **Evaluation Suite**: Generates test accuracy, classification reports (Precision, Recall, F1-score), and confusion matrix heatmaps.

---

## 🏗️ Architecture

```text
Input (128 × 128 × 3 RGB)
       ↓
Data Augmentation (Flip, Rotation, Zoom, Translation)
       ↓
Rescaling (1./255)
       ↓
Conv2D(32, 3x3) + BatchNorm + ReLU → MaxPooling2D(2x2)
       ↓
Conv2D(64, 3x3) + BatchNorm + ReLU → MaxPooling2D(2x2)
       ↓
Conv2D(128, 3x3) + BatchNorm + ReLU → MaxPooling2D(2x2)
       ↓
Flatten → Dense(128, ReLU) → Dropout(0.5)
       ↓
Dense(3, Softmax) → Output: [Paper, Rock, Scissors]
```

---

## 🚀 Quickstart

### 1. Environment Setup

Ensure virtual environment dependencies are installed:

```bash
# Using pip
pip install -r requirements.txt
```

### 2. Train the CNN Model

Run the training script (this will automatically download the dataset if not already present, train the CNN with early stopping, and save the best model checkpoint):

```bash
python src/train.py --epochs 20 --batch-size 32
```

Outputs generated:
* Model: `models/rps_cnn.keras`
* Label mapping: `models/class_indices.json`
* Training curves: `models/training_history.png`

### 3. Evaluate the Model

Evaluate the model against the independent test dataset:

```bash
python src/evaluate.py
```

Outputs generated:
* Test accuracy & loss metrics
* Classification Report (Precision, Recall, F1)
* Confusion Matrix: `models/confusion_matrix.png`

### 4. Single Image Prediction (CLI)

Test inference on an individual image:

```bash
python src/predict.py dataset/rock/rock01-001.png
```

### 5. Real-Time Webcam Detection (OpenCV)

Launch the live webcam detector:

```bash
python src/webcam.py
```

**Controls**:
* `[Q]`: Quit webcam window
* `[C]`: Freeze / Resume live prediction
* `[S]`: Save current frame snapshot to disk

### 6. Interactive Streamlit Web App

Launch the web app with in-browser webcam support and the interactive AI game:

```bash
streamlit run src/app.py
```

---

## 📁 Project Structure

```text
rps-web cnn/
│
├── dataset/                     # Training dataset (downloaded automatically)
│   ├── paper/
│   ├── rock/
│   └── scissors/
│
├── dataset_test/                # Independent test dataset
│   ├── paper/
│   ├── rock/
│   └── scissors/
│
├── models/
│   ├── rps_cnn.keras           # Trained native Keras model
│   ├── class_indices.json      # Class label mapping
│   ├── training_history.png    # Accuracy & Loss curves
│   └── confusion_matrix.png    # Evaluation confusion matrix
│
├── src/
│   ├── __init__.py
│   ├── data.py                 # Dataset downloader, loader & augmentation
│   ├── model.py                # CNN model architecture
│   ├── train.py                # Model training runner
│   ├── evaluate.py             # Evaluation & confusion matrix generator
│   ├── predict.py              # Single image inference
│   ├── webcam.py               # Real-time OpenCV webcam application
│   └── app.py                  # Streamlit web dashboard & AI mini-game
│
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata
└── README.md                   # Documentation
```

---

## 🎯 Tech Stack

* **Deep Learning**: TensorFlow 2.x, Keras
* **Computer Vision**: OpenCV (`opencv-python`), Pillow
* **Web UI & Dashboard**: Streamlit
* **Scientific Computing**: NumPy, Scikit-learn, Matplotlib
