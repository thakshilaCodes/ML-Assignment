"""
Train XGBoost on the full Train_Dataset.csv (no random train/validation split).

Evaluation uses stratified k-fold CV on the training set. The official ``Test_Dataset.csv``
has no ``Default`` column (Kaggle-style); we score it and save predictions only.

See ``preprocessing_pipeline.py`` for the pipeline definition.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

from preprocessing_pipeline import build_xgb_pipeline, align_features_for_pipeline, prepare_raw_features

ALGO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ALGO_DIR.parent.parent
ALGO_NAME = ALGO_DIR.name

DATA_TRAIN = PROJECT_ROOT / "data" / "raw" / "Train_Dataset.csv"
DATA_TEST = PROJECT_ROOT / "data" / "raw" / "Test_Dataset.csv"
TARGET_COL = "Default"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / ALGO_NAME
TRAINED_DIR = OUTPUT_DIR / "trained_models"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_OUT = TRAINED_DIR / f"{ALGO_NAME}.joblib"
METRICS_OUT = METRICS_DIR / f"{ALGO_NAME}.txt"
TEST_PREDS_OUT = METRICS_DIR / f"{ALGO_NAME}_test_predictions.csv"

CV_FOLDS = 5
SELECT_PERCENTILE = 40.0
RANDOM_STATE = 42


def _scale_pos_weight(y: pd.Series) -> float:
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    if n_pos < 1:
        return 1.0
    return float(n_neg / n_pos)


def main():
    df = pd.read_csv(DATA_TRAIN, low_memory=False)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found in {DATA_TRAIN}")

    X = prepare_raw_features(df.drop(columns=[TARGET_COL]))
    y = df[TARGET_COL].astype(int)

    spw = _scale_pos_weight(y)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    base = build_xgb_pipeline(
        X, select_percentile=SELECT_PERCENTILE, scale_pos_weight=spw
    )

    print("Running stratified cross-validation on training data...")
    cv_acc = cross_val_score(
        base, X, y, cv=cv, scoring="accuracy", n_jobs=1
    )
    cv_auc = cross_val_score(
        base, X, y, cv=cv, scoring="roc_auc", n_jobs=1
    )

    model = build_xgb_pipeline(
        X, select_percentile=SELECT_PERCENTILE, scale_pos_weight=spw
    )
    model.fit(X, y)

    train_pred = model.predict(X)
    train_acc = accuracy_score(y, train_pred)

    lines = [
        f"train_rows={len(y)}",
        f"positive_rate={(y == 1).mean():.4f}",
        f"scale_pos_weight={spw:.4f}",
        f"cv_folds={CV_FOLDS}",
        f"cv_accuracy_mean={cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})",
        f"cv_roc_auc_mean={cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})",
        f"train_fit_accuracy={train_acc:.4f}  (in-sample; optimistic)",
        "",
        "Note: Test_Dataset.csv has no Default labels in this repo; test predictions are saved separately.",
        "",
        "Classification report (in-sample, for reference):",
        classification_report(y, train_pred, zero_division=0),
    ]

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    METRICS_OUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved model to {MODEL_OUT}")
    print(f"Saved metrics to {METRICS_OUT}")
    print(f"CV accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
    print(f"CV ROC-AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

    if DATA_TEST.is_file():
        df_test = pd.read_csv(DATA_TEST, low_memory=False)
        if TARGET_COL in df_test.columns:
            df_test = df_test.drop(columns=[TARGET_COL])
        missing = [c for c in X.columns if c not in df_test.columns]
        extra = [c for c in df_test.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Test CSV missing columns: {missing[:20]}")
        if extra:
            df_test = df_test.drop(columns=extra, errors="ignore")
        X_test = align_features_for_pipeline(model, df_test[X.columns])
        proba = model.predict_proba(X_test)
        p_default = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        out = pd.DataFrame(
            {
                "ID": X_test["ID"] if "ID" in X_test.columns else np.arange(len(X_test)),
                "pred_default": model.predict(X_test),
                "p_default": p_default,
            }
        )
        out.to_csv(TEST_PREDS_OUT, index=False)
        print(f"Saved test predictions to {TEST_PREDS_OUT}")
    else:
        print(f"No test file at {DATA_TEST}; skip test predictions.")


if __name__ == "__main__":
    main()
