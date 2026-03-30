"""
Train sklearn Gradient Boosting on Train_Dataset.csv.

Folder layout:
  models/<ALGO_NAME>/train.py
  data/raw/Train_Dataset.csv
  outputs/<ALGO_NAME>/trained_models/<ALGO_NAME>.joblib
  outputs/<ALGO_NAME>/metrics/<ALGO_NAME>.txt
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ALGO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ALGO_DIR.parent.parent
ALGO_NAME = ALGO_DIR.name

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Train_Dataset.csv"
TARGET_COL = "Default"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / ALGO_NAME
TRAINED_DIR = OUTPUT_DIR / "trained_models"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_OUT = TRAINED_DIR / f"{ALGO_NAME}.joblib"
METRICS_OUT = METRICS_DIR / f"{ALGO_NAME}.txt"


def _stratify_if_ok(y: pd.Series):
    if y.nunique() < 2:
        return None
    if y.value_counts().min() < 2:
        return None
    return y


def main():
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found in {DATA_PATH}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X = pd.get_dummies(X, dummy_na=True)

    strat = _stratify_if_ok(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    METRICS_OUT.write_text(f"accuracy={acc:.4f}\n", encoding="utf-8")
    print(f"Saved model to {MODEL_OUT}")
    print(f"Saved metrics to {METRICS_OUT}")


if __name__ == "__main__":
    main()
