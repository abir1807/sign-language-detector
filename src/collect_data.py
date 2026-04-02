"""
collect_data.py
---------------
Records hand landmark data from webcam for each gesture.
Press 'S' to start recording, 'Q' to quit current gesture early.
Run: python src/collect_data.py
"""

import cv2
import mediapipe as mp
import csv
import os
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
GESTURES        = ["hello", "thanks", "yes", "no", "iloveyou",
                   "please", "sorry", "help", "more", "stop"]
SAMPLES_PER_GESTURE = 300          # frames per gesture
SEQUENCE_LENGTH     = 1            # set >1 for LSTM temporal sequences
DATA_DIR            = "data/processed"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(DATA_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

LANDMARK_COUNT = 21 * 3   # x, y, z per landmark


def normalize_landmarks(hand_lms):
    """Normalize landmarks relative to wrist position."""
    wrist = hand_lms.landmark[0]
    coords = []
    for lm in hand_lms.landmark:
        coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return coords


def collect_gesture(gesture_name: str):
    cap     = cv2.VideoCapture(0)
    data    = []
    count   = 0
    started = False

    print(f"\n{'='*50}")
    print(f"  Gesture: {gesture_name.upper()}")
    print(f"  Target : {SAMPLES_PER_GESTURE} samples")
    print(f"  Press  : [S] to start | [Q] to quit early")
    print(f"{'='*50}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,230,118), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(255,255,255), thickness=1)
                )
                if started:
                    row = normalize_landmarks(hand_lms) + [gesture_name]
                    data.append(row)
                    count += 1

        status = "RECORDING" if started else "Press [S] to start"
        color  = (0, 255, 100) if started else (0, 200, 255)
        cv2.putText(frame, f"{gesture_name.upper()} | {status} | {count}/{SAMPLES_PER_GESTURE}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Progress bar
        if SAMPLES_PER_GESTURE > 0:
            pct = int((count / SAMPLES_PER_GESTURE) * frame.shape[1])
            cv2.rectangle(frame, (0, frame.shape[0]-8),
                          (pct, frame.shape[0]), (0, 255, 100), -1)

        cv2.imshow("Sign Language — Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and not started:
            started = True
            print("  ▶ Recording started...")
        if count >= SAMPLES_PER_GESTURE or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if data:
        out_path = os.path.join(DATA_DIR, f"{gesture_name}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)
        print(f"  ✓ Saved {len(data)} samples → {out_path}")
    else:
        print(f"  ✗ No data collected for {gesture_name}")


def main():
    print("\n" + "="*50)
    print("  SIGN LANGUAGE DATA COLLECTOR")
    print("="*50)
    print(f"  Gestures to collect: {', '.join(GESTURES)}\n")

    for gesture in GESTURES:
        input(f"  Ready for '{gesture}'? Press [Enter] to open camera...")
        collect_gesture(gesture)
        time.sleep(1)

    print("\n✅ All gestures collected! Run: python src/preprocess.py")


if __name__ == "__main__":
    main()
