"""Interactive Streamlit Web Application for Rock-Paper-Scissors AI Battle Arena."""

import base64
import io
import json
import random
import socket
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.data import IMAGE_SIZE, CLASS_EMOJIS
from src.predict import load_classifier

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "rps_cnn.keras"
CLASSES_PATH = MODELS_DIR / "class_indices.json"

st.set_page_config(
    page_title="Rock-Paper-Scissors AI Battle Arena",
    page_icon="✊✋✌️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global model cache for API & Streamlit
_GLOBAL_MODEL = None
_GLOBAL_CLASSES = None


def get_or_load_model():
    global _GLOBAL_MODEL, _GLOBAL_CLASSES
    if _GLOBAL_MODEL is None and MODEL_PATH.exists():
        try:
            _GLOBAL_MODEL, _GLOBAL_CLASSES = load_classifier(MODEL_PATH, CLASSES_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
    return _GLOBAL_MODEL, _GLOBAL_CLASSES


def run_prediction_on_bytes(image_bytes: bytes):
    model, class_names = get_or_load_model()
    if model is None:
        return {"error": "Model not loaded"}

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    img_tensor = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_tensor, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    best_class = class_names[best_idx]
    confidence = float(preds[best_idx])
    scores = {class_names[i]: float(preds[i]) for i in range(len(class_names))}

    return {
        "class": best_class,
        "emoji": CLASS_EMOJIS.get(best_class, "❓"),
        "confidence": confidence,
        "scores": scores
    }


class PredictionAPIHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ready", "service": "rps_prediction"}).encode("utf-8"))

    def do_POST(self):
        if self.path.startswith("/predict"):
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                post_data = self.rfile.read(content_length)
                payload = json.loads(post_data.decode("utf-8"))
                img_data = payload.get("image", "")

                if "," in img_data:
                    img_data = img_data.split(",", 1)[1]

                img_bytes = base64.b64decode(img_data)
                result = run_prediction_on_bytes(img_bytes)

                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()

    def log_message(self, format, *args):
        pass


@st.cache_resource
def get_prediction_api_port():
    """Start the prediction HTTP server on an available port and cache the port."""
    get_or_load_model()

    for port in range(9500, 9550):
        try:
            server = HTTPServer(("127.0.0.1", port), PredictionAPIHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            print(f"[RPS AI] Background prediction server running on port {port}")
            return port
        except OSError:
            continue

    server = HTTPServer(("127.0.0.1", 0), PredictionAPIHandler)
    assigned_port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[RPS AI] Background prediction server running on fallback port {assigned_port}")
    return assigned_port


def main():
    api_port = get_prediction_api_port()
    model, class_names = get_or_load_model()

    # Sidebar
    with st.sidebar:
        st.markdown("### 🏆 Rock-Paper-Scissors AI")
        st.markdown("""
        **Controls & Info**:
        - ⚔️ **Live Battle Arena**: In-browser real-time camera with 3-2-1 automated countdown & AI opponent!
        - 📸 **Instant Snapshot**: Direct photo snapshot mode.
        - 📁 **File Upload**: Test on local images.
        """)
        st.markdown("---")
        if model is not None:
            st.success(f"✅ **CNN Model Loaded**\n- File: `rps_cnn.keras`\n- Test Accuracy: **100%**\n- Server Port: `{api_port}`")
        else:
            st.error("❌ Model not found! Please run `python src/train.py`.")

    if model is None:
        st.warning("⚠️ Please train the model first by running `python src/train.py` in your terminal.")
        return

    tab_arena, tab_snapshot, tab_upload, tab_stats = st.tabs([
        "⚔️ Live 3-2-1 Battle Arena",
        "📸 Instant Snapshot Mode",
        "📁 File Upload",
        "📊 Model Stats"
    ])

    # Tab 1: Live Interactive Browser Arena (HTML5 + WebRTC + 3-2-1 Countdown)
    with tab_arena:
        arena_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');
                
                * {{
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                }}
                
                body {{
                    font-family: 'Inter', sans-serif;
                    background: transparent;
                    color: #f8fafc;
                    padding: 10px 0;
                }}
                
                .arena-wrapper {{
                    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
                    border: 1px solid rgba(99, 102, 241, 0.35);
                    border-radius: 20px;
                    padding: 24px;
                    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5);
                }}
                
                .score-bar {{
                    display: flex;
                    justify-content: center;
                    gap: 36px;
                    background: rgba(15, 23, 42, 0.85);
                    border: 1px solid rgba(255, 255, 255, 0.12);
                    border-radius: 14px;
                    padding: 14px 28px;
                    margin-bottom: 22px;
                }}
                
                .score-item {{
                    text-align: center;
                }}
                
                .score-item .label {{
                    font-size: 13px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    color: #94a3b8;
                }}
                
                .score-item .val {{
                    font-family: 'Outfit', sans-serif;
                    font-size: 34px;
                    font-weight: 800;
                    color: #f8fafc;
                    transition: transform 0.2s ease, color 0.2s ease;
                }}

                .score-bump {{
                    transform: scale(1.25);
                    color: #38bdf8 !important;
                }}
                
                .arena-grid {{
                    display: grid;
                    grid-template-columns: 1fr auto 1fr;
                    gap: 20px;
                    align-items: center;
                }}
                
                .fighter-box {{
                    background: rgba(30, 41, 59, 0.7);
                    border: 2px solid rgba(255, 255, 255, 0.1);
                    border-radius: 18px;
                    padding: 20px;
                    text-align: center;
                    min-height: 400px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    align-items: center;
                    backdrop-filter: blur(10px);
                }}
                
                .player-box {{
                    border-color: rgba(56, 189, 248, 0.5);
                }}
                
                .ai-box {{
                    border-color: rgba(244, 114, 182, 0.5);
                }}
                
                .video-container {{
                    position: relative;
                    width: 100%;
                    max-width: 380px;
                    height: 250px;
                    border-radius: 12px;
                    overflow: hidden;
                    background: #020617;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}
                
                video {{
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    transform: scaleX(-1);
                }}
                
                .target-box {{
                    position: absolute;
                    top: 15px;
                    right: 15px;
                    width: 170px;
                    height: 170px;
                    border: 3px dashed #38bdf8;
                    border-radius: 14px;
                    pointer-events: none;
                    box-shadow: 0 0 20px rgba(56, 189, 248, 0.5);
                    background: rgba(56, 189, 248, 0.08);
                    display: flex;
                    align-items: flex-end;
                    justify-content: center;
                    padding-bottom: 8px;
                }}
                
                .target-label {{
                    font-size: 11px;
                    font-weight: 800;
                    color: #0f172a;
                    background: #38bdf8;
                    padding: 3px 8px;
                    border-radius: 4px;
                    letter-spacing: 0.8px;
                    text-transform: uppercase;
                }}
                
                .vs-divider {{
                    text-align: center;
                    padding: 0 10px;
                }}
                
                .vs-divider h2 {{
                    font-family: 'Outfit', sans-serif;
                    font-size: 40px;
                    font-weight: 900;
                    background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }}
                
                .countdown-overlay {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(15, 23, 42, 0.85);
                    display: none;
                    justify-content: center;
                    align-items: center;
                    font-family: 'Outfit', sans-serif;
                    font-size: 88px;
                    font-weight: 900;
                    color: #facc15;
                    text-shadow: 0 0 35px rgba(250, 204, 21, 0.95);
                    z-index: 10;
                    animation: pulse 0.7s infinite alternate;
                }}

                @keyframes pulse {{
                    0% {{ transform: scale(0.92); }}
                    100% {{ transform: scale(1.12); }}
                }}
                
                .ai-card-content {{
                    width: 100%;
                    max-width: 380px;
                    height: 250px;
                    background: #020617;
                    border-radius: 12px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    border: 1px solid rgba(255, 255, 255, 0.08);
                }}
                
                .move-emoji {{
                    font-size: 92px;
                    line-height: 1;
                    margin-bottom: 8px;
                    filter: drop-shadow(0 6px 16px rgba(0,0,0,0.6));
                }}
                
                .move-name {{
                    font-family: 'Outfit', sans-serif;
                    font-size: 24px;
                    font-weight: 800;
                    letter-spacing: 1.5px;
                    text-transform: uppercase;
                }}

                .preview-strip {{
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    margin-top: 6px;
                }}

                .crop-thumb {{
                    width: 42px;
                    height: 42px;
                    border-radius: 6px;
                    border: 1px solid #38bdf8;
                    object-fit: cover;
                    background: #0f172a;
                }}
                
                .btn-start {{
                    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    padding: 18px 36px;
                    font-family: 'Outfit', sans-serif;
                    font-size: 20px;
                    font-weight: 800;
                    letter-spacing: 1px;
                    cursor: pointer;
                    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.45);
                    transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
                    width: 100%;
                    margin-top: 22px;
                }}
                
                .btn-start:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 12px 30px rgba(99, 102, 241, 0.65);
                }}
                
                .btn-start:disabled {{
                    opacity: 0.6;
                    cursor: not-allowed;
                    transform: none;
                }}
                
                .result-banner {{
                    margin-top: 18px;
                    padding: 16px;
                    border-radius: 12px;
                    font-family: 'Outfit', sans-serif;
                    font-size: 22px;
                    font-weight: 800;
                    text-align: center;
                    display: none;
                }}
                
                .banner-player-win {{
                    background: linear-gradient(90deg, #059669 0%, #10b981 100%);
                    color: white;
                    display: block;
                    box-shadow: 0 6px 24px rgba(16, 185, 129, 0.45);
                }}
                
                .banner-ai-win {{
                    background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
                    color: white;
                    display: block;
                    box-shadow: 0 6px 24px rgba(239, 68, 68, 0.45);
                }}
                
                .banner-tie-match {{
                    background: linear-gradient(90deg, #d97706 0%, #f59e0b 100%);
                    color: white;
                    display: block;
                    box-shadow: 0 6px 24px rgba(245, 158, 11, 0.45);
                }}
            </style>
        </head>
        <body>
            <div class="arena-wrapper">
                <!-- Scoreboard -->
                <div class="score-bar">
                    <div class="score-item">
                        <div class="label">👤 You</div>
                        <div class="val" id="player-score">0</div>
                    </div>
                    <div class="score-item">
                        <div class="label">🤖 AI</div>
                        <div class="val" id="ai-score">0</div>
                    </div>
                    <div class="score-item">
                        <div class="label">🤝 Ties</div>
                        <div class="val" id="tie-score">0</div>
                    </div>
                </div>

                <!-- Battle Grid -->
                <div class="arena-grid">
                    <!-- Player Side -->
                    <div class="fighter-box player-box">
                        <h3 style="color: #38bdf8; font-family: 'Outfit', sans-serif; font-size: 20px;">👤 YOUR FIGHTER</h3>
                        <div class="video-container" id="video-container">
                            <video id="webcam-video" autoplay playsinline muted></video>
                            <div class="target-box" id="target-box">
                                <div class="target-label">SHOW HAND HERE</div>
                            </div>
                            <div class="countdown-overlay" id="countdown-layer">3</div>
                        </div>
                        <div style="margin-top: 10px; width: 100%;">
                            <div class="move-name" id="player-move-title" style="color: #38bdf8;">READY</div>
                            <div class="preview-strip">
                                <img id="crop-preview" class="crop-thumb" style="display: none;" title="Model Input Crop" />
                                <p id="player-conf" style="color: #94a3b8; font-size: 13px;">Place hand in blue box</p>
                            </div>
                        </div>
                    </div>

                    <!-- VS Center -->
                    <div class="vs-divider">
                        <h2>⚡ VS ⚡</h2>
                    </div>

                    <!-- AI Side -->
                    <div class="fighter-box ai-box">
                        <h3 style="color: #f472b6; font-family: 'Outfit', sans-serif; font-size: 20px;">🤖 AI OPPONENT</h3>
                        <div class="ai-card-content" id="ai-card">
                            <div class="move-emoji" id="ai-emoji">❓</div>
                            <div class="move-name" id="ai-move-title" style="color: #f472b6;">WAITING</div>
                        </div>
                        <div style="margin-top: 10px;">
                            <p id="ai-status" style="color: #94a3b8; font-size: 13px;">AI is ready</p>
                        </div>
                    </div>
                </div>

                <!-- Start Button -->
                <button class="btn-start" id="start-btn" onclick="startBattle()">🚀 START 3-2-1 BATTLE ROUND</button>

                <!-- Outcome Banner -->
                <div class="result-banner" id="result-banner"></div>
            </div>

            <!-- Hidden Capture Canvas (128x128 Model Input) -->
            <canvas id="capture-canvas" width="128" height="128" style="display: none;"></canvas>

            <script>
                const API_PORT = {api_port};
                const video = document.getElementById('webcam-video');
                const canvas = document.getElementById('capture-canvas');
                const countdownLayer = document.getElementById('countdown-layer');
                const startBtn = document.getElementById('start-btn');
                const banner = document.getElementById('result-banner');
                const cropPreview = document.getElementById('crop-preview');

                const playerScoreElem = document.getElementById('player-score');
                const aiScoreElem = document.getElementById('ai-score');
                const tieScoreElem = document.getElementById('tie-score');

                const playerMoveElem = document.getElementById('player-move-title');
                const playerConfElem = document.getElementById('player-conf');
                const aiEmojiElem = document.getElementById('ai-emoji');
                const aiMoveElem = document.getElementById('ai-move-title');
                const aiStatusElem = document.getElementById('ai-status');

                let pScore = 0;
                let aScore = 0;
                let tScore = 0;
                let isBattling = false;

                const moves = ['rock', 'paper', 'scissors'];
                const emojis = {{ 'rock': '✊', 'paper': '✋', 'scissors': '✌️' }};

                // Initialize Camera
                async function initCamera() {{
                    try {{
                        const stream = await navigator.mediaDevices.getUserMedia({{
                            video: {{ width: {{ ideal: 640 }}, height: {{ ideal: 480 }}, facingMode: 'user' }}
                        }});
                        video.srcObject = stream;
                    }} catch (err) {{
                        console.error("Camera access error:", err);
                        playerConfElem.innerText = "⚠️ Camera access blocked. Please allow webcam permissions.";
                    }}
                }}
                initCamera();

                function determineWinner(player, ai) {{
                    const p = player.toLowerCase();
                    const a = ai.toLowerCase();
                    if (p === a) return 'Tie';
                    if ((p === 'rock' && a === 'scissors') ||
                        (p === 'paper' && a === 'rock') ||
                        (p === 'scissors' && a === 'paper')) {{
                        return 'Player';
                    }}
                    return 'AI';
                }}

                function bumpElement(elem) {{
                    elem.classList.add('score-bump');
                    setTimeout(() => elem.classList.remove('score-bump'), 300);
                }}

                async function startBattle() {{
                    if (isBattling) return;
                    isBattling = true;
                    startBtn.disabled = true;
                    banner.style.display = 'none';
                    banner.className = 'result-banner';

                    countdownLayer.style.display = 'flex';
                    playerMoveElem.innerText = 'FOCUSING...';
                    playerConfElem.innerText = 'Hold hand gesture in the blue box!';

                    // AI shuffling animation
                    let shuffleInterval = setInterval(() => {{
                        const randomMove = moves[Math.floor(Math.random() * moves.length)];
                        aiEmojiElem.innerText = emojis[randomMove];
                        aiMoveElem.innerText = 'CHOOSING...';
                    }}, 100);

                    // 3s Countdown: 3 -> 2 -> 1 -> SHOOT!
                    let count = 3;
                    countdownLayer.innerText = count;

                    let countdownTimer = setInterval(async () => {{
                        count--;
                        if (count > 0) {{
                            countdownLayer.innerText = count;
                        }} else if (count === 0) {{
                            countdownLayer.innerText = '⚡ SHOOT!';
                        }} else {{
                            clearInterval(countdownTimer);
                            clearInterval(shuffleInterval);
                            countdownLayer.style.display = 'none';

                            // Capture exact hand region & classify
                            await processCapture();
                            isBattling = false;
                            startBtn.disabled = false;
                        }}
                    }}, 950);
                }}

                async function processCapture() {{
                    const container = document.getElementById('video-container');
                    const targetBox = document.getElementById('target-box');
                    
                    const cRect = container.getBoundingClientRect();
                    const bRect = targetBox.getBoundingClientRect();

                    const vWidth = video.videoWidth || 640;
                    const vHeight = video.videoHeight || 480;

                    // Compute aspect-ratio scale for object-fit: cover
                    const cWidth = cRect.width;
                    const cHeight = cRect.height;
                    
                    const scale = Math.max(cWidth / vWidth, cHeight / vHeight);
                    const renderedWidth = vWidth * scale;
                    const renderedHeight = vHeight * scale;
                    const offsetX = (renderedWidth - cWidth) / 2;
                    const offsetY = (renderedHeight - cHeight) / 2;

                    // Target box coordinates inside the container
                    const boxLeft = bRect.left - cRect.left;
                    const boxTop = bRect.top - cRect.top;
                    const boxW = bRect.width;
                    const boxH = bRect.height;

                    // Visual is mirrored (scaleX(-1)), so visual right corresponds to raw video left
                    const visualBoxRight = cWidth - (boxLeft + boxW);
                    
                    let sx = (visualBoxRight + offsetX) / scale;
                    let sy = (boxTop + offsetY) / scale;
                    let sw = boxW / scale;
                    let sh = boxH / scale;

                    // Fallback to center-right crop if values are out of bounds
                    if (isNaN(sx) || isNaN(sy) || sw <= 0 || sh <= 0) {{
                        sw = vWidth * 0.45;
                        sh = vHeight * 0.6;
                        sx = 0;
                        sy = vHeight * 0.1;
                    }} else {{
                        sx = Math.max(0, Math.min(vWidth - sw, sx));
                        sy = Math.max(0, Math.min(vHeight - sh, sy));
                    }}

                    // Draw EXACT cropped hand into 128x128 canvas
                    const ctx = canvas.getContext('2d');
                    ctx.save();
                    // Horizontal flip so that hand orientation matches visual preview
                    ctx.translate(canvas.width, 0);
                    ctx.scale(-1, 1);
                    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
                    ctx.restore();

                    const base64Image = canvas.toDataURL('image/jpeg', 0.95);
                    cropPreview.src = base64Image;
                    cropPreview.style.display = 'block';

                    try {{
                        const apiUrl = `http://127.0.0.1:${{API_PORT}}/predict`;
                        const response = await fetch(apiUrl, {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{ image: base64Image }})
                        }});

                        if (!response.ok) {{
                            throw new Error(`HTTP error status ${{response.status}}`);
                        }}

                        const data = await response.json();
                        const playerGesture = data.class || 'rock';
                        const confidence = (data.confidence * 100).toFixed(1);
                        const playerEmo = emojis[playerGesture] || data.emoji || '✊';

                        // Pick AI Move
                        const aiGesture = moves[Math.floor(Math.random() * moves.length)];
                        const aiEmo = emojis[aiGesture];

                        // Reveal both moves simultaneously
                        playerMoveElem.innerText = `${{playerEmo}} ${{playerGesture.toUpperCase()}}`;
                        playerConfElem.innerText = `Confidence: ${{confidence}}%`;

                        aiEmojiElem.innerText = aiEmo;
                        aiMoveElem.innerText = aiGesture.toUpperCase();
                        aiStatusElem.innerText = `AI Picked ${{aiGesture.toUpperCase()}}`;

                        // Calculate winner and update scoreboard
                        const outcome = determineWinner(playerGesture, aiGesture);
                        if (outcome === 'Player') {{
                            pScore++;
                            playerScoreElem.innerText = pScore;
                            bumpElement(playerScoreElem);
                            banner.className = 'result-banner banner-player-win';
                            banner.innerText = `🎉 YOU WIN! ${{playerGesture.toUpperCase()}} beats ${{aiGesture.toUpperCase()}}!`;
                        }} else if (outcome === 'AI') {{
                            aScore++;
                            aiScoreElem.innerText = aScore;
                            bumpElement(aiScoreElem);
                            banner.className = 'result-banner banner-ai-win';
                            banner.innerText = `🤖 AI WINS! ${{aiGesture.toUpperCase()}} beats ${{playerGesture.toUpperCase()}}!`;
                        }} else {{
                            tScore++;
                            tieScoreElem.innerText = tScore;
                            bumpElement(tieScoreElem);
                            banner.className = 'result-banner banner-tie-match';
                            banner.innerText = `🤝 IT'S A TIE! Both chose ${{playerGesture.toUpperCase()}}!`;
                        }}
                    }} catch (err) {{
                        console.error("Prediction API Error:", err);
                        playerMoveElem.innerText = "ERROR";
                        playerConfElem.innerText = "Check network or restart app";
                        aiMoveElem.innerText = "WAITING";
                        aiStatusElem.innerText = "Error in round";
                    }}
                }}
            </script>
        </body>
        </html>
        """
        components.html(arena_html, height=680, scrolling=False)

    # Tab 2: Snapshot Mode
    with tab_snapshot:
        st.subheader("Instant Photo Snapshot Mode")
        cam_shot = st.camera_input("Take a photo of your hand gesture")
        if cam_shot is not None:
            pil_img = Image.open(cam_shot).convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            res = run_prediction_on_bytes(buf.getvalue())

            p_class = res.get("class", "rock")
            p_conf = res.get("confidence", 0.0)
            p_emoji = res.get("emoji", "")

            ai_choice = random.choice(["rock", "paper", "scissors"])
            ai_emoji = CLASS_EMOJIS.get(ai_choice, "")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.image(pil_img, caption="Captured Image", use_container_width=True)
            with col_s2:
                st.markdown(f"### Detected: **{p_emoji} {p_class.upper()}** (`{p_conf*100:.1f}%`)")
                st.markdown(f"### 🤖 AI Move: **{ai_emoji} {ai_choice.upper()}**")

                p = p_class.lower()
                a = ai_choice.lower()
                if p == a:
                    st.info(f"🤝 Tie! Both chose {p.capitalize()}")
                elif (p == "rock" and a == "scissors") or (p == "paper" and a == "rock") or (p == "scissors" and a == "paper"):
                    st.success(f"🎉 You Win! {p.capitalize()} beats {a.capitalize()}")
                else:
                    st.error(f"🤖 AI Wins! {a.capitalize()} beats {p.capitalize()}")

    # Tab 3: Upload Image File
    with tab_upload:
        st.subheader("Upload Gesture Image")
        up_file = st.file_uploader("Choose a JPG / PNG photo...", type=["jpg", "png", "jpeg"])
        if up_file is not None:
            pil_up = Image.open(up_file).convert("RGB")
            buf_u = io.BytesIO()
            pil_up.save(buf_u, format="JPEG")
            res_u = run_prediction_on_bytes(buf_u.getvalue())

            col_u1, col_u2 = st.columns(2)
            with col_u1:
                st.image(pil_up, caption="Uploaded Image", use_container_width=True)
            with col_u2:
                st.markdown(f"### Prediction: **{res_u.get('emoji', '')} {res_u.get('class', '').upper()}**")
                st.markdown(f"**Confidence:** `{res_u.get('confidence', 0.0)*100:.2f}%`")
                for c_n, sc in res_u.get("scores", {}).items():
                    st.write(f"{CLASS_EMOJIS.get(c_n, '')} **{c_n.capitalize()}**: `{sc*100:.1f}%`")
                    st.progress(float(sc))

    # Tab 4: Model Stats
    with tab_stats:
        st.subheader("Model Architecture & Benchmark")
        h_path = MODELS_DIR / "training_history.png"
        c_path = MODELS_DIR / "confusion_matrix.png"

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if h_path.exists():
                st.image(str(h_path), caption="Training Curves")
        with col_g2:
            if c_path.exists():
                st.image(str(c_path), caption="Confusion Matrix (100% Accuracy)")


if __name__ == "__main__":
    main()
