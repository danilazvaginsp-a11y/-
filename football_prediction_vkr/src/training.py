from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss

try:
    from src.build_dataset import build_dataset
    from src.data_processing import create_features
except ModuleNotFoundError:
    # Supports direct script run: python3 ./src/training.py
    from build_dataset import build_dataset
    from data_processing import create_features


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "catboost_model.joblib"

CATEGORICAL_FEATURES = ["home_team", "away_team", "league", "season"]


def train() -> None:
    # Always rebuild dataset from source CSV files before training.
    raw_df = build_dataset()
    features_df = create_features(raw_df)

    features_df = features_df.sort_values("date").reset_index(drop=True)
    numeric_cols = features_df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "target"]
    for col in numeric_cols:
        features_df[col] = features_df[col].fillna(features_df[col].mean())
    features_df = features_df.dropna(subset=["target"]).reset_index(drop=True)
    split_idx = int(len(features_df) * 0.8)
    if split_idx == 0 or split_idx >= len(features_df):
        raise ValueError("Not enough rows after feature engineering for 80/20 split.")

    train_df = features_df.iloc[:split_idx].copy()
    test_df = features_df.iloc[split_idx:].copy()

    X_train = train_df.drop(columns=["target", "date"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target", "date"])
    y_test = test_df["target"]

    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=100,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50,
    )

    y_pred = model.predict(X_test).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = model.predict_proba(X_test)
    class_order = list(model.classes_)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba, labels=class_order)

    print(f"Accuracy: {acc:.4f}")
    print(f"Log loss: {ll:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": X_train.columns.tolist(),
        "cat_features": CATEGORICAL_FEATURES,
        "class_order": class_order,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()

