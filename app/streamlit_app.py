"""
streamlit_app.py
----------------
Full Streamlit web dashboard for the Sign Language Detector.
Features: Live webcam detection, model metrics, gesture guide, sentence builder.
Run: streamlit run app/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import cv2
import numpy as np
import joblib
import json
import time
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sign Language Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main { background: #0a0a14; color: #e8e8f0; }
.stApp { background: linear-gradient(135deg, #0a0a14 0%, #0f0f1e 100%); }

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 8px 0;
    backdrop-filter: blur(10px);
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #00FFB3; font-family: 'JetBrains Mono'; }
.metric-label { font-size: 0.82rem; color: #888; text-transform: uppercase; letter-spacing: 0.1em; }

.prediction-box {
    background: linear-gradient(135deg, rgba(0,255,179,0.08), rgba(0,150,255,0.08));
    border: 1px solid rgba(0,255,179,0.25);
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin: 12px 0;
}
.prediction-text { font-size: 3rem; font-weight: 700; color: #00FFB3; letter-spacing: -0.02em; }
.confidence-text { font-size: 1rem; color: #888; margin-top: 4px; }

.sentence-box {
    background: rgba(255, 220, 100, 0.05);
    border: 1px solid rgba(255,220,100,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    font-family: 'JetBrains Mono';
    font-size: 1.1rem;
    color: #FFDC64;
    min-height: 56px;
}

.gesture-chip {
    display: inline-block;
    background: rgba(0,255,179,0.1);
    border: 1px solid rgba(0,255,179,0.3);
    border-radius: 20px;
    padding: 5px 16px;
    margin: 4px;
    font-size: 0.85rem;
    color: #00FFB3;
}

.stButton>button {
    background: linear-gradient(135deg, #00FFB3, #00B3FF);
    color: #000;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-family: 'Space Grotesk';
    padding: 10px 24px;
    transition: all 0.2s;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,255,179,0.3); }

h1, h2, h3 { color: #e8e8f0; }
.sidebar .sidebar-content { background: #0f0f1e; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_encoder():
    import mediapipe as mp
    from tensorflow import keras

    model, le, classes = None, None, []
    try:
        model = keras.models.load_model("models/sign_model.keras")
        le    = joblib.load("data/label_encoder.pkl")
        classes = list(le.classes_)
    except Exception as e:
        st.sidebar.error(f"Model not found. Run train.py first.\n{e}")

    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    draw  = mp.solutions.drawing_utils
    mp_h  = mp.solutions.hands
    return model, le, classes, hands, draw, mp_h

model, le, classes, hands_mp, mp_draw, mp_hands_mod = load_model_and_encoder()


def normalize_landmarks(hand_lms):
    wrist = hand_lms.landmark[0]
    return np.array([
        v for lm in hand_lms.landmark
        for v in [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
    ], dtype=np.float32)


# ── Session State ──────────────────────────────────────────────────────────
if "sentence"    not in st.session_state: st.session_state.sentence    = []
if "last_word"   not in st.session_state: st.session_state.last_word   = None
if "hold_count"  not in st.session_state: st.session_state.hold_count  = 0
if "history"     not in st.session_state: st.session_state.history     = []
if "running"     not in st.session_state: st.session_state.running     = False
if "total_preds" not in st.session_state: st.session_state.total_preds = 0
if "high_conf"   not in st.session_state: st.session_state.high_conf   = 0


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤟 Sign Language AI")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    CONF_THRESHOLD = st.slider("Confidence Threshold", 0.4, 0.99, 0.70, 0.01)
    HOLD_FRAMES    = st.slider("Hold Frames (sentence)", 10, 60, 25)
    SMOOTH_WINDOW  = st.slider("Smoothing Window", 5, 20, 10)

    st.markdown("---")
    st.markdown("### 🎯 Supported Gestures")
    if classes:
        for c in classes:
            st.markdown(f'<span class="gesture-chip">🤚 {c}</span>', unsafe_allow_html=True)
    else:
        st.info("Train model to see gestures")

    st.markdown("---")
    if os.path.exists("models/model_info.json"):
        with open("models/model_info.json") as f:
            info = json.load(f)
        st.markdown("### 📊 Model Info")
        st.metric("Test Accuracy", f"{info.get('accuracy',0)*100:.1f}%")
        st.metric("Gesture Classes", len(info.get("classes", [])))


# ── Main UI ────────────────────────────────────────────────────────────────
st.markdown("# 🤟 Sign Language Detector")
st.markdown("Real-time ASL gesture recognition powered by MediaPipe + Neural Network")
st.markdown("---")

col_cam, col_info = st.columns([3, 2], gap="large")

with col_cam:
    st.markdown("### 📸 Live Camera Feed")
    start_col, stop_col, clear_col = st.columns(3)
    with start_col:
        if st.button("▶ Start", use_container_width=True):
            st.session_state.running = True
    with stop_col:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.running = False
    with clear_col:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.sentence   = []
            st.session_state.last_word  = None
            st.session_state.hold_count = 0

    cam_placeholder = st.empty()
    cam_placeholder.markdown(
        '<div style="height:400px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);'
        'border-radius:16px;display:flex;align-items:center;justify-content:center;color:#444;font-size:1.1rem;">'
        '📷 Press Start to begin</div>', unsafe_allow_html=True
    )

with col_info:
    st.markdown("### 🎯 Detection")
    pred_placeholder = st.empty()
    pred_placeholder.markdown(
        '<div class="prediction-box"><div class="prediction-text">—</div>'
        '<div class="confidence-text">Waiting for detection...</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("### 💬 Sentence Builder")
    sent_placeholder = st.empty()
    sent_placeholder.markdown(
        '<div class="sentence-box">Your sentence will appear here...</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 📈 Session Stats")
    m1, m2, m3 = st.columns(3)
    stat1 = m1.empty(); stat2 = m2.empty(); stat3 = m3.empty()

    stat1.metric("Detections", 0)
    stat2.metric("High Conf", 0)
    stat3.metric("Words", 0)


# ── Live Detection Loop ─────────────────────────────────────────────────────
if st.session_state.running and model is not None:
    from collections import deque, Counter
    smooth_hist = deque(maxlen=SMOOTH_WINDOW)
    cap = cv2.VideoCapture(0)

    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands_mp.process(rgb)
            rgb.flags.writeable = True

            prediction = "No Hand"; confidence = 0.0

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lms, mp_hands_mod.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,230,118), thickness=2, circle_radius=4),
                        mp_draw.DrawingSpec(color=(255,255,255), thickness=2)
                    )
                    lm = normalize_landmarks(hand_lms).reshape(1, -1)
                    probs     = model.predict(lm, verbose=0)[0]
                    idx       = np.argmax(probs)
                    prediction= le.inverse_transform([idx])[0]
                    confidence= float(probs[idx])

            # Smooth
            smooth_hist.append((prediction, confidence))
            if smooth_hist:
                preds = [p for p, _ in smooth_hist]
                pred_smooth = Counter(preds).most_common(1)[0][0]
                conf_smooth = np.mean([c for p, c in smooth_hist if p == pred_smooth])
            else:
                pred_smooth, conf_smooth = prediction, confidence

            # Sentence builder
            if pred_smooth not in ("No Hand",) and conf_smooth >= CONF_THRESHOLD:
                st.session_state.total_preds += 1
                if conf_smooth >= 0.85:
                    st.session_state.high_conf += 1

                if pred_smooth == st.session_state.last_word:
                    st.session_state.hold_count += 1
                    if st.session_state.hold_count == HOLD_FRAMES:
                        if (not st.session_state.sentence or
                                st.session_state.sentence[-1] != pred_smooth):
                            st.session_state.sentence.append(pred_smooth)
                else:
                    st.session_state.last_word  = pred_smooth
                    st.session_state.hold_count = 0

            # Update UI
            color_hex = "#00FFB3" if conf_smooth > 0.80 else \
                        "#00B3FF" if conf_smooth > 0.60 else "#FF6B6B"
            pred_placeholder.markdown(
                f'<div class="prediction-box">'
                f'<div class="prediction-text" style="color:{color_hex}">{pred_smooth}</div>'
                f'<div class="confidence-text">Confidence: {conf_smooth*100:.1f}%</div>'
                f'</div>', unsafe_allow_html=True
            )

            sentence_str = " ".join(st.session_state.sentence) or "..."
            sent_placeholder.markdown(
                f'<div class="sentence-box">{sentence_str}</div>',
                unsafe_allow_html=True
            )

            stat1.metric("Detections", st.session_state.total_preds)
            stat2.metric("High Conf",  st.session_state.high_conf)
            stat3.metric("Words",      len(st.session_state.sentence))

            # Show frame
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(img_rgb, channels="RGB", use_container_width=True)

    finally:
        cap.release()
