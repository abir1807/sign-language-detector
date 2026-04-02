# 🤟 Sign Language Detector
> Real-time ASL gesture recognition using MediaPipe + Neural Network + Streamlit

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red?style=flat-square)

---

## 🚀 Quick Start (Windows)

```bash
# 1. Clone & enter project
cd sign_language_detector

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Collect gesture data (webcam required)
python src/collect_data.py

# 5. Preprocess data
python src/preprocess.py

# 6. Train model
python src/train.py

# 7a. Run OpenCV app
python app/main.py

# 7b. OR run Streamlit web app
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

```
sign_language_detector/
├── data/
│   ├── raw/                  # Raw recordings
│   ├── processed/            # Per-gesture CSVs (landmark data)
│   ├── dataset.csv           # Merged dataset
│   ├── X_train.npy           # Train features
│   ├── X_test.npy            # Test features
│   ├── y_train.npy           # Train labels
│   ├── y_test.npy            # Test labels
│   └── label_encoder.pkl     # Label ↔ gesture mapping
│
├── src/
│   ├── collect_data.py       # Webcam data collection
│   ├── preprocess.py         # Merge + encode + split data
│   ├── train.py              # Train MLP + RandomForest
│   └── utils.py              # Shared helpers
│
├── models/
│   ├── sign_model.keras      # Best MLP model
│   ├── sign_model.tflite     # Edge-deployable model
│   ├── rf_model.pkl          # Baseline RandomForest
│   ├── model_info.json       # Accuracy + metadata
│   ├── training_history.png  # Loss/accuracy curves
│   └── confusion_matrix.png  # Evaluation heatmap
│
├── app/
│   ├── main.py               # OpenCV real-time app
│   └── streamlit_app.py      # Streamlit web dashboard
│
├── notebooks/
│   └── exploration.ipynb     # EDA & experiments
│
├── requirements.txt
└── README.md
```

---

## 🎯 Supported Gestures (Default)

| Gesture   | Meaning       |
|-----------|---------------|
| hello     | Hello         |
| thanks    | Thank you     |
| yes       | Yes           |
| no        | No            |
| iloveyou  | I Love You    |
| please    | Please        |
| sorry     | Sorry         |
| help      | Help          |
| more      | More          |
| stop      | Stop          |

> Add more gestures by editing `GESTURES` list in `src/collect_data.py`

---

## 🧠 Model Architecture

```
Input (63 features = 21 landmarks × 3 coords)
  → BatchNormalization
  → Dense(256, ReLU) + Dropout(0.3)
  → Dense(128, ReLU) + Dropout(0.3)
  → Dense(64, ReLU)
  → Dense(N_classes, Softmax)
```

- **Landmarks normalized** relative to wrist position (translation invariant)
- **Smoothing window** of 10–12 frames reduces prediction flicker
- **Sentence builder** adds words after 25 consistent frames

---

## 📊 Performance

| Model         | Accuracy  |
|---------------|-----------|
| Random Forest | ~96–98%   |
| MLP (Neural)  | ~97–99%   |

---

## 🔮 Future Improvements

- [ ] Full 26-letter ASL alphabet
- [ ] Two-hand gesture support
- [ ] LSTM for temporal gesture sequences
- [ ] Text-to-speech output
- [ ] Mobile deployment (TFLite)
- [ ] Docker containerization

---

## 📝 License
MIT License — free to use, modify, and distribute.
