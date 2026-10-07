"""Real-time OpenCV webcam inference for Rock-Paper-Scissors classifier."""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

# Ensure project root is in sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import IMAGE_SIZE, CLASS_EMOJIS
from src.predict import load_classifier

MODELS_DIR = PROJECT_ROOT / "models"


def draw_hud(
    frame: np.ndarray,
    prediction: str,
    confidence: float,
    scores: dict,
    fps: float,
    roi_coords: tuple,
    paused: bool = False
):
    """Draws sleek HUD overlays, probability meters, and bounding boxes."""
    h, w, _ = frame.shape
    x1, y1, x2, y2 = roi_coords

    # Color based on confidence
    if confidence > 0.80:
        box_color = (0, 255, 128)  # Bright Green
    elif confidence > 0.50:
        box_color = (0, 215, 255)  # Amber / Yellow
    else:
        box_color = (0, 100, 255)  # Orange

    # 1. Draw ROI Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(
        frame,
        "Place hand inside box",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        box_color,
        1,
        cv2.LINE_AA
    )

    # 2. Top Info Header Card (Semi-transparent background)
    header_overlay = frame.copy()
    cv2.rectangle(header_overlay, (15, 15), (320, 155), (20, 20, 20), -1)
    cv2.addWeighted(header_overlay, 0.65, frame, 0.35, 0, frame)

    # 3. Text & Metrics
    status_text = "PAUSED" if paused else f"{prediction.upper()}"
    cv2.putText(
        frame,
        f"Gesture: {status_text}",
        (25, 45),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        box_color,
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # Probability Bars
    y_offset = 100
    for cls_name, score in scores.items():
        bar_w = int(score * 100)
        cv2.putText(
            frame,
            f"{cls_name[:4].capitalize()}:",
            (25, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        # Background bar
        cv2.rectangle(frame, (90, y_offset - 10), (190, y_offset - 2), (60, 60, 60), -1)
        # Filled bar
        cv2.rectangle(frame, (90, y_offset - 10), (90 + bar_w, y_offset - 2), box_color, -1)
        # Percentage text
        cv2.putText(
            frame,
            f"{score * 100:4.1f}%",
            (200, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        y_offset += 20

    # FPS counter on bottom-right
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 110, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    # Controls instruction footer
    cv2.putText(
        frame,
        "Press [Q] Quit | [C] Freeze/Resume | [S] Save Snapshot",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )


def determine_winner(player_move: str, ai_move: str):
    """Returns (result_code, message)."""
    p = player_move.lower()
    a = ai_move.lower()

    if p == a:
        return "Tie", f"Tie! Both chose {p.capitalize()}"
    elif (
        (p == "rock" and a == "scissors") or
        (p == "paper" and a == "rock") or
        (p == "scissors" and a == "paper")
    ):
        return "Player", f"You Win! {p.capitalize()} beats {a.capitalize()}"
    else:
        return "AI", f"AI Wins! {a.capitalize()} beats {p.capitalize()}"


def run_webcam(
    camera_index: int = 0,
    model_path: Path = MODELS_DIR / "rps_cnn.keras",
    classes_path: Path = MODELS_DIR / "class_indices.json"
):
    """Runs live webcam detection loop with 3-2-1 AI Battle mode."""
    import random
    print("Loading model for webcam inference...")
    model, class_names = load_classifier(model_path, classes_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: Unable to open camera with index {camera_index}.")
        print("Please check camera connections or try a different index with --camera-index 1.")
        return

    print("\nStarting live webcam stream...")
    print("Controls:")
    print("  [SPACE] - Start 3-2-1 Battle against AI")
    print("  [C]     - Pause / Resume continuous prediction")
    print("  [S]     - Save snapshot")
    print("  [R]     - Reset scoreboard")
    print("  [Q]     - Quit")

    prev_time = time.time()
    paused = False
    last_pred = "Waiting..."
    last_conf = 0.0
    last_scores = {c: 0.0 for c in class_names}

    # Battle mode state
    player_score = 0
    ai_score = 0
    ties_count = 0
    battle_active = False
    battle_start_time = 0.0
    battle_result_text = ""
    battle_result_color = (255, 255, 255)
    ai_move = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting webcam stream.")
            break

        # Flip horizontally for intuitive selfie mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Define Region of Interest (Square box centered right)
        box_size = int(min(h, w) * 0.55)
        x1 = int(w * 0.52)
        y1 = int((h - box_size) / 2)
        x2 = x1 + box_size
        y2 = y1 + box_size

        curr_time = time.time()

        # Handle Battle Countdown
        countdown_sec = 3.0
        if battle_active:
            elapsed = curr_time - battle_start_time
            remaining = countdown_sec - elapsed

            if remaining > 0:
                count_display = str(int(np.ceil(remaining)))
                cv2.putText(
                    frame,
                    count_display,
                    (int(w / 2) - 40, int(h / 2) + 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    4.0,
                    (0, 215, 255),
                    8,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    "SHOW YOUR GESTURE!",
                    (int(w / 2) - 200, int(h / 2) + 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            else:
                # Capture and evaluate at countdown end
                roi = frame[y1:y2, x1:x2]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi_resized = cv2.resize(roi_rgb, IMAGE_SIZE)
                    roi_tensor = np.expand_dims(roi_resized, axis=0).astype(np.float32)

                    probs = model.predict(roi_tensor, verbose=0)[0]
                    best_idx = int(np.argmax(probs))
                    last_pred = class_names[best_idx]
                    last_conf = float(probs[best_idx])
                    last_scores = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

                    # AI Move
                    ai_choices = ["rock", "paper", "scissors"]
                    ai_move = random.choice(ai_choices)

                    outcome, expl = determine_winner(last_pred, ai_move)
                    if outcome == "Player":
                        player_score += 1
                        battle_result_text = f"YOU WIN! ({last_pred.upper()} vs {ai_move.upper()})"
                        battle_result_color = (0, 255, 128)
                    elif outcome == "AI":
                        ai_score += 1
                        battle_result_text = f"AI WINS! (AI: {ai_move.upper()} vs {last_pred.upper()})"
                        battle_result_color = (0, 80, 255)
                    else:
                        ties_count += 1
                        battle_result_text = f"TIE! (Both chose {last_pred.upper()})"
                        battle_result_color = (0, 215, 255)

                battle_active = False

        elif not paused:
            # Crop ROI
            roi = frame[y1:y2, x1:x2]
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi_resized = cv2.resize(roi_rgb, IMAGE_SIZE)
                roi_tensor = np.expand_dims(roi_resized, axis=0).astype(np.float32)

                probs = model.predict(roi_tensor, verbose=0)[0]
                best_idx = int(np.argmax(probs))
                last_pred = class_names[best_idx]
                last_conf = float(probs[best_idx])
                last_scores = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        # Calculate FPS
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
        prev_time = curr_time

        # Draw UI
        draw_hud(frame, last_pred, last_conf, last_scores, fps, (x1, y1, x2, y2), paused)

        # Draw Scoreboard Card Top Right
        cv2.rectangle(frame, (w - 300, 15), (w - 15, 95), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"SCORE: YOU {player_score} | AI {ai_score} | TIES {ties_count}",
            (w - 290, 45),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "[SPACE] Play 3s Battle",
            (w - 290, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 215, 255),
            1,
            cv2.LINE_AA
        )

        # Draw Battle Result Banner if available
        if battle_result_text:
            cv2.rectangle(frame, (15, h - 80), (w - 15, h - 35), (20, 20, 20), -1)
            cv2.putText(
                frame,
                battle_result_text,
                (25, h - 48),
                cv2.FONT_HERSHEY_DUPLEX,
                0.75,
                battle_result_color,
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Rock-Paper-Scissors Live Classifier", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            battle_active = True
            battle_start_time = time.time()
            battle_result_text = ""
        elif key == ord("c"):
            paused = not paused
        elif key == ord("r"):
            player_score = 0
            ai_score = 0
            ties_count = 0
            battle_result_text = "Scoreboard Reset"
            battle_result_color = (200, 200, 200)
        elif key == ord("s"):
            snapshot_path = Path.cwd() / f"snapshot_{int(time.time())}.png"
            cv2.imwrite(str(snapshot_path), frame)
            print(f"Snapshot saved to: {snapshot_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Webcam Rock-Paper-Scissors Classifier")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0)")
    args = parser.parse_args()

    run_webcam(camera_index=args.camera_index)

