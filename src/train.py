"""
train.py
--------
Trains an MLP (Dense Neural Network) on landmark data.
Also trains a RandomForest as baseline comparison.
Run: python src/train.py
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR  = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    le      = joblib.load(os.path.join(DATA_DIR, "label_encoder.pkl"))
    return X_train, X_test, y_train, y_test, le


def build_mlp(input_dim, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ], name="SignLangMLP")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Model Training History", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["accuracy"],    label="Train Acc", color="#00C9A7")
    axes[0].plot(history.history["val_accuracy"],label="Val Acc",   color="#FF6B6B")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],    label="Train Loss", color="#00C9A7")
    axes[1].plot(history.history["val_loss"],label="Val Loss",   color="#FF6B6B")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_history.png"), dpi=150)
    print("  ✓ Saved training_history.png")


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print("  ✓ Saved confusion_matrix.png")


def train_random_forest(X_train, X_test, y_train, y_test, le):
    print("\n── Baseline: Random Forest ─────────────────────")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"  RF Accuracy: {acc:.4f}")
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))
    print("  ✓ Saved rf_model.pkl")
    return rf


def train_mlp(X_train, X_test, y_train, y_test, le):
    print("\n── Main Model: MLP Neural Network ──────────────")
    num_classes = len(le.classes_)
    model = build_mlp(X_train.shape[1], num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "best_mlp.keras"),
            save_best_only=True, monitor="val_accuracy"
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  MLP Test Accuracy: {acc:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))

    plot_training(history)
    plot_confusion_matrix(y_test, y_pred, le.classes_)

    # Save full model + TFLite
    model.save(os.path.join(MODEL_DIR, "sign_model.keras"))
    print("  ✓ Saved sign_model.keras")

    # TFLite for edge deployment
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(os.path.join(MODEL_DIR, "sign_model.tflite"), "wb") as f:
        f.write(tflite_model)
    print("  ✓ Saved sign_model.tflite (edge-ready)")

    # Save accuracy info
    info = {"accuracy": float(acc), "classes": list(le.classes_),
            "num_features": int(X_train.shape[1])}
    import json
    with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    return model


def main():
    print("\n" + "="*50)
    print("  TRAINING SIGN LANGUAGE MODELS")
    print("="*50)

    X_train, X_test, y_train, y_test, le = load_data()
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    train_random_forest(X_train, X_test, y_train, y_test, le)
    train_mlp(X_train, X_test, y_train, y_test, le)

    print("\n✅ Training complete! Run: python app/main.py  OR  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
