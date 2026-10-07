

### 🚀 How to Run

#### 1. Train the Model
```bash
python src/train.py --epochs 20 --batch-size 32
```
* Automatically trains the model, applies early stopping, and saves the best model to `models/rps_cnn.keras` and training history curves to `models/training_history.png`.

#### 2. Evaluate on Test Dataset
```bash
python src/evaluate.py
```
* Computes test accuracy, loss, classification report (Precision, Recall, F1), and generates `models/confusion_matrix.png`.

#### 3. Single-Image Prediction (CLI)
```bash
python src/predict.py dataset_test/rock/testrock01-01.png
```

#### 4. Real-Time OpenCV Webcam Feed
```bash
python src/webcam.py
```
* **Interactive HUD Controls**:
  * `[Q]` — Quit stream
  * `[C]` — Pause / Resume prediction
  * `[S]` — Save current frame snapshot to disk

#### 5. Interactive Streamlit Web App & AI Game
```bash
streamlit run src/app.py
```
* Access the web dashboard in your browser to use your webcam (`st.camera_input`), upload custom images, and play an interactive Rock-Paper-Scissors game against the AI with live score tracking.

> [!NOTE]
> If your Windows system has **Smart App Control (SAC)** enabled in strict enforcement mode, ensure it allows Python C-extensions or run via WSL2/native terminal.