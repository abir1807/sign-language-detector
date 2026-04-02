"""
preprocess.py
-------------
Merges all gesture CSVs, encodes labels, splits into train/test sets,
and saves numpy arrays ready for model training.
Run: python src/preprocess.py
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_DIR    = "data/processed"
OUTPUT_DIR  = "data"
TEST_SIZE   = 0.2
RANDOM_SEED = 42


def load_and_merge():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}. Run collect_data.py first.")

    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None)
        gesture = os.path.splitext(os.path.basename(f))[0]
        print(f"  Loaded: {gesture:<15} → {len(df):>4} samples")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.dropna(inplace=True)

    num_features = combined.shape[1] - 1
    feature_cols = [f"lm_{i}" for i in range(num_features)]
    combined.columns = feature_cols + ["label"]
    return combined


def preprocess():
    print("\n" + "="*50)
    print("  PREPROCESSING DATA")
    print("="*50)

    df = load_and_merge()
    print(f"\n  Total samples : {len(df)}")
    print(f"  Features      : {df.shape[1]-1}")
    print(f"  Classes       : {df['label'].nunique()} → {sorted(df['label'].unique())}")

    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=y_encoded
    )

    # Save arrays
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)
    joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    print(f"\n  Train : {len(X_train)} samples")
    print(f"  Test  : {len(X_test)} samples")
    print(f"\n  ✓ Saved to data/  →  X_train, X_test, y_train, y_test, label_encoder.pkl")
    print("\n  Next: python src/train.py")

    return X_train, X_test, y_train, y_test, le


if __name__ == "__main__":
    preprocess()
