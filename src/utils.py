"""
utils.py
--------
Shared utility functions used across the project.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils


def normalize_landmarks(hand_lms):
    """Normalize landmarks relative to wrist (landmark 0)."""
    wrist = hand_lms.landmark[0]
    coords = []
    for lm in hand_lms.landmark:
        coords.extend([lm.x - wrist.x,
                       lm.y - wrist.y,
                       lm.z - wrist.z])
    return np.array(coords, dtype=np.float32)


def draw_styled_landmarks(image, hand_lms):
    """Draw hand landmarks with custom styling."""
    mp_draw.draw_landmarks(
        image, hand_lms, mp_hands.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(0, 230, 118), thickness=2, circle_radius=4),
        mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
    )


def draw_prediction_overlay(image, prediction, confidence, sentence=""):
    """Draw a sleek prediction UI overlay on the frame."""
    h, w = image.shape[:2]

    # Top bar
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    # Prediction text
    color = (0, 255, 140) if confidence > 0.80 else \
            (0, 200, 255) if confidence > 0.60 else (100, 100, 255)

    cv2.putText(image, f"Sign: {prediction}",
                (15, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
    cv2.putText(image, f"Confidence: {confidence*100:.1f}%",
                (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    # Confidence bar
    bar_x, bar_y, bar_h = w - 180, 20, 16
    bar_w = int(160 * confidence)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x+160, bar_y+bar_h), (50,50,50), -1)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), color, -1)

    # Sentence at bottom
    if sentence:
        cv2.rectangle(image, (0, h-50), (w, h), (10, 10, 30), -1)
        cv2.putText(image, f"Sentence: {sentence}",
                    (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 100), 2)

    return image


class GestureSmoothing:
    """Smooth predictions over a rolling window to reduce flicker."""
    def __init__(self, window=10):
        self.window  = window
        self.history = deque(maxlen=window)

    def update(self, prediction, confidence):
        self.history.append((prediction, confidence))

    def get_smoothed(self):
        if not self.history:
            return "...", 0.0
        from collections import Counter
        preds = [p for p, _ in self.history]
        most_common = Counter(preds).most_common(1)[0][0]
        avg_conf = np.mean([c for p, c in self.history if p == most_common])
        return most_common, avg_conf


class SentenceBuilder:
    """Build a sentence from detected signs."""
    def __init__(self, hold_frames=20):
        self.sentence   = []
        self.last_word  = None
        self.hold_count = 0
        self.hold_frames= hold_frames

    def update(self, word):
        if word == self.last_word:
            self.hold_count += 1
            if self.hold_count == self.hold_frames:
                if not self.sentence or self.sentence[-1] != word:
                    self.sentence.append(word)
        else:
            self.last_word  = word
            self.hold_count = 0

    def get_sentence(self):
        return " ".join(self.sentence)

    def clear(self):
        self.sentence = []
        self.last_word = None
        self.hold_count = 0
