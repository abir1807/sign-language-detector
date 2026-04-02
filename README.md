<div align="center">

# 🤟 Sign Language Detector

### Real-Time ASL Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.7-00897B?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

**Detects 10 ASL hand gestures in real-time from your webcam using MediaPipe hand landmarks and a trained MLP neural network. Runs entirely on CPU — no GPU required.**

<br/>

[🚀 Setup](#-setup) • [📖 How It Works](#-how-it-works) • [🏗️ Project Structure](#️-project-structure) • [🧠 Model](#-model-architecture) • [📊 Results](#-results) • [🔮 Roadmap](#-roadmap)

</div>

---

## ✨ Features

- 🎯 **Real-time detection** at 25–60 FPS on a standard laptop CPU
- 🤚 **21 hand landmarks** extracted per frame using MediaPipe
- 🧠 **MLP Neural Network** achieving 97–99% accuracy
- 🔤 **Sentence builder** — hold a gesture to add words automatically
- 📉 **Gesture smoothing** — rolling window eliminates prediction flicker
- 📦 **TFLite export** — ready to deploy on Android or Raspberry Pi
- 🌐 **Streamlit web dashboard** with live controls and session stats
- ⚙️ **One-click Windows setup** via `setup.bat`

---

## 🎯 Supported Gestures

| Gesture | Meaning | Gesture | Meaning |
|---------|---------|---------|---------|
| `hello` | 👋 Hello | `please` | 🙏 Please |
| `thanks` | 🙏 Thank You | `sorry` | 😔 Sorry |
| `yes` | ✅ Yes | `help` | 🆘 Help |
| `no` | ❌ No | `more` | ➕ More |
| `iloveyou` | 🤟 I Love You | `stop` | ✋ Stop |

> to add more gestures, edit the `GESTURES` list in `src/collect_data.py` — no other code changes needed.

---

## 🚀 Setup

### Requirements
- Python 3.10+
- A webcam
- Windows (Linux/Mac: skip `setup.bat`, run the pip commands manually)

---

### Step 1 — Install dependencies

**Windows (one-click):**
```bash
setup.bat
```

**Linux / Mac (manual):**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### Step 2 — Collect gesture data
```bash
python src/collect_data.py
```
- Webcam opens for each gesture one at a time
- Press **`S`** to start recording (300 samples per gesture)
- Press **`Q`** to finish early and move to the next gesture
- Takes around 15 minutes total for all 10 gestures

---

### Step 3 — Preprocess the data
```bash
python src/preprocess.py
```
Merges all gesture CSVs, encodes labels, and saves train/test arrays.

---

### Step 4 — Train the model
```bash
python src/train.py
```
Training takes 2–5 minutes. Model, confusion matrix, and training plots are saved to `models/`.

---

### Step 5 — Run the detector

**Option A — Desktop OpenCV app (lightweight):**
```bash
python app/main.py
```

**Option B — Streamlit web dashboard (full UI, opens in browser):**
```bash
streamlit run app/streamlit_app.py
```

---

## 🏗️ Project Structure

```
sign_language_detector/
│
├── 📁 data/
│   ├── processed/              # Per-gesture CSVs (auto-generated after Step 2)
│   │   ├── hello.csv
│   │   ├── thanks.csv
│   │   └── ...
│   ├── X_train.npy             # Train features  (auto-generated after Step 3)
│   ├── X_test.npy              # Test features   (auto-generated after Step 3)
│   ├── y_train.npy             # Train labels    (auto-generated after Step 3)
│   ├── y_test.npy              # Test labels     (auto-generated after Step 3)
│   └── label_encoder.pkl       # Gesture name ↔ integer mapping
│
├── 📁 src/
│   ├── collect_data.py         # Webcam landmark recorder
│   ├── preprocess.py           # Merge CSVs, encode labels, split data
│   ├── train.py                # Train MLP + RandomForest, export TFLite
│   └── utils.py                # Normalization, smoothing, sentence builder
│
├── 📁 models/                  # Auto-generated after Step 4
│   ├── sign_model.keras        # Best trained MLP model
│   ├── sign_model.tflite       # Edge-deployable compressed model
│   ├── rf_model.pkl            # Baseline RandomForest model
│   ├── model_info.json         # Accuracy + class metadata
│   ├── training_history.png    # Loss & accuracy curves
│   └── confusion_matrix.png    # Per-class evaluation heatmap
│
├── 📁 app/
│   ├── main.py                 # OpenCV real-time desktop app
│   └── streamlit_app.py        # Streamlit web dashboard
│
├── setup.bat                   # One-click Windows environment setup
├── requirements.txt            # All dependencies with pinned versions
└── README.md
```

---

## 📖 How It Works

Every webcam frame passes through a 5-stage pipeline:

```
Webcam Frame
     │
     ▼
┌──────────────────────────────┐
│ 1. MediaPipe Hand Detection  │  →  Finds hand, extracts 21 landmarks (x, y, z)
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│ 2. Landmark Normalization    │  →  Subtract wrist → position-independent features
└──────────────────────────────┘
     │  63 values (21 landmarks × 3 coords)
     ▼
┌──────────────────────────────┐
│ 3. MLP Neural Network        │  →  Outputs probability score per gesture class
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│ 4. Gesture Smoothing         │  →  Rolling window majority vote (last 10–12 frames)
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│ 5. Sentence Builder          │  →  Adds word after gesture is held for ~1 second
└──────────────────────────────┘
     │
     ▼
  Display on Screen
```

**Why normalize landmarks?**
Raw coordinates change depending on where your hand sits in the frame. Subtracting the wrist position from every landmark makes the features position-independent — the model sees the same gesture whether your hand is top-left or bottom-right.

**Why gesture smoothing?**
Single-frame predictions flicker during transitions. A rolling window of the last 10–12 frames with a majority vote delivers stable results with no added lag.

---

## 🧠 Model Architecture

```
Input: 63 features (21 landmarks × x, y, z)
        │
  BatchNormalization
        │
   Dense(256, ReLU)
        │
    Dropout(0.3)
        │
   Dense(128, ReLU)
        │
    Dropout(0.3)
        │
    Dense(64, ReLU)
        │
  Dense(N, Softmax)    ←  N = number of gesture classes
        │
    Prediction
```

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr = 0.001) |
| Loss function | Sparse Categorical Crossentropy |
| Max epochs | 100 (with EarlyStopping, patience = 10) |
| Batch size | 32 |
| Validation split | 15% of training data |
| Export formats | `.keras` and `.tflite` |

---

## 📊 Results

| Model | Test Accuracy | Notes |
|-------|-------------|-------|
| Random Forest (baseline) | ~96–98% | Trained on same landmark features |
| MLP Neural Network | ~97–99% | Main model used in the app |
| MLP TFLite (quantized) | ~97% | For mobile / edge deployment |

> Accuracy depends on recording quality. More samples and varied lighting improve generalization.

---

## ⌨️ Controls

### Desktop App — `python app/main.py`

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `C` | Clear the sentence buffer |
| `S` | Save a screenshot |

### Streamlit Dashboard — `streamlit run app/streamlit_app.py`

| Control | What It Does |
|---------|-------------|
| Confidence Threshold | Minimum confidence required to register a prediction |
| Hold Frames | Stable frames required before a word is added to the sentence |
| Smoothing Window | Number of recent frames used for majority vote |
| Start / Stop | Toggle the live webcam feed |
| Clear | Reset the sentence builder |

---

## 🛠️ Tech Stack

| Library | Version | Role |
|---------|---------|------|
| [MediaPipe](https://mediapipe.dev) | 0.10.7 | Hand landmark detection (21 keypoints per frame) |
| [OpenCV](https://opencv.org) | 4.8.1 | Webcam capture, frame processing, display |
| [TensorFlow / Keras](https://tensorflow.org) | 2.15.0 | MLP training, evaluation, TFLite export |
| [Scikit-learn](https://scikit-learn.org) | 1.3.2 | RandomForest baseline, LabelEncoder, data splitting |
| [Streamlit](https://streamlit.io) | 1.29.0 | Web dashboard UI |
| [NumPy](https://numpy.org) | 1.26.2 | Feature arrays and numerical operations |
| [Pandas](https://pandas.pydata.org) | 2.1.3 | CSV loading and data preprocessing |
| [Joblib](https://joblib.readthedocs.io) | 1.3.2 | Saving and loading model/encoder files |
| [Matplotlib](https://matplotlib.org) | 3.8.2 | Training history plots |
| [Seaborn](https://seaborn.pydata.org) | 0.13.0 | Confusion matrix heatmap |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not opening | Change `VideoCapture(0)` → `VideoCapture(1)` |
| Low FPS | Resize frame to `(640, 480)`, set `model_complexity=0` in MediaPipe |
| Hand not detected | Improve lighting, keep hand fully within the frame |
| `Model not found` error | Run `python src/train.py` before launching the app |
| `ModuleNotFoundError` | Activate venv first: `venv\Scripts\activate` |
| Streamlit camera blocked | Open in Chrome and allow camera permissions |
| Low accuracy | Set `SAMPLES_PER_GESTURE = 500` and re-record gestures |

---

## 🔮 Roadmap

- [ ] Full ASL alphabet (26 letters)
- [ ] Two-hand gesture support
- [ ] LSTM model for motion-based gestures
- [ ] Text-to-speech output
- [ ] Android deployment via TFLite
- [ ] Docker support
- [ ] REST API with FastAPI

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

Built with ❤️ using MediaPipe · TensorFlow · OpenCV · Streamlit

⭐ **Star this repo if you found it helpful!**

</div>
