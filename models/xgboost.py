"""Train XGBoost classifier on Train_Dataset.csv."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/raw/Train_Dataset.csv"
TARGET_COL = "target"  # Change to the real label column in Train_Dataset.csv
MODEL_OUT = ROOT / "outputs/trained_models/xgboost.joblib"
METRICS_OUT = ROOT / "outputs/metrics/xgboost.txt"


def main():
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found in {DATA_PATH}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X = pd.get_dummies(X, dummy_na=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(random_state=42, eval_metric="logloss")
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
