"""
main.py  (OpenCV Real-Time App)
--------------------------------
Runs the sign language detector live from webcam.
Controls:
  Q  → Quit
  C  → Clear sentence
  S  → Save screenshot
Run: python app/main.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import joblib
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
import mediapipe as mp

from src.utils import (normalize_landmarks, draw_styled_landmarks,
                       draw_prediction_overlay, GestureSmoothing,
                       SentenceBuilder)

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH = "models/sign_model.keras"
LE_PATH    = "data/label_encoder.pkl"

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found. Run: python src/train.py")
    sys.exit(1)

model   = keras.models.load_model(MODEL_PATH)
le      = joblib.load(LE_PATH)
print(f"✓ Model loaded | Classes: {list(le.classes_)}")

mp_hands_mod = mp.solutions.hands
hands = mp_hands_mod.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

smoother = GestureSmoothing(window=12)
builder  = SentenceBuilder(hold_frames=25)

cap       = cv2.VideoCapture(0)
prev_time = 0
ss_count  = 0
os.makedirs("screenshots", exist_ok=True)


def main():
    global prev_time, ss_count

    print("\n  Controls: [Q] Quit | [C] Clear sentence | [S] Screenshot\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (960, 720))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        prediction = "No Hand"
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                draw_styled_landmarks(frame, hand_lms)
                lm_array = normalize_landmarks(hand_lms).reshape(1, -1)
                probs     = model.predict(lm_array, verbose=0)[0]
                idx       = np.argmax(probs)
                prediction= le.inverse_transform([idx])[0]
                confidence= float(probs[idx])

        smoother.update(prediction, confidence)
        pred_smooth, conf_smooth = smoother.get_smoothed()

        if pred_smooth != "No Hand":
            builder.update(pred_smooth)

        sentence = builder.get_sentence()
        frame = draw_prediction_overlay(frame, pred_smooth, conf_smooth, sentence)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1]-110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,150,150), 1)

        cv2.imshow("Sign Language Detector — Press Q to Quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            builder.clear()
            print("  Sentence cleared.")
        elif key == ord('s'):
            path = f"screenshots/snap_{ss_count:03d}.png"
            cv2.imwrite(path, frame)
            ss_count += 1
            print(f"  Screenshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Bye!")


if __name__ == "__main__":
    main()
