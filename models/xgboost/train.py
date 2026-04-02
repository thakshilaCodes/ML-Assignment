"""
Train XGBoost on ``Train_Dataset.csv``.

- **Stratified train/test split** on the labeled CSV yields ``test_accuracy`` and related
  metrics (Kaggle ``Test_Dataset.csv`` often has no ``Default`` labels).
- **5-fold stratified CV** runs only on the **training** portion of that split so folds
  do not overlap the test set.
- The saved ``.joblib`` model is **re-fit on all rows** afterward for deployment.

See ``preprocessing_pipeline.py`` for the pipeline definition.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

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
# Labeled test fraction from Train_Dataset (Test_Dataset.csv has no Default in this repo)
TEST_SPLIT_SIZE = 0.2


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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SPLIT_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    spw_train = _scale_pos_weight(y_train)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    base = build_xgb_pipeline(
        X_train, select_percentile=SELECT_PERCENTILE, scale_pos_weight=spw_train
    )

    print(
        f"Running {CV_FOLDS}-fold stratified CV on training split "
        f"({len(y_train)} train rows, {len(y_test)} test rows)..."
    )
    cv_acc = cross_val_score(
        base, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1
    )
    cv_auc = cross_val_score(
        base, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
    )

    eval_model = build_xgb_pipeline(
        X_train, select_percentile=SELECT_PERCENTILE, scale_pos_weight=spw_train
    )
    eval_model.fit(X_train, y_train)
    test_pred = eval_model.predict(X_test)
    test_proba = eval_model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)
    test_prec_w = precision_score(
        y_test, test_pred, average="weighted", zero_division=0
    )
    test_rec_w = recall_score(
        y_test, test_pred, average="weighted", zero_division=0
    )
    test_f1_w = f1_score(y_test, test_pred, average="weighted", zero_division=0)
    test_prec_1 = precision_score(
        y_test, test_pred, pos_label=1, average="binary", zero_division=0
    )
    test_rec_1 = recall_score(
        y_test, test_pred, pos_label=1, average="binary", zero_division=0
    )
    test_f1_1 = f1_score(
        y_test, test_pred, pos_label=1, average="binary", zero_division=0
    )

    spw_full = _scale_pos_weight(y)
    model = build_xgb_pipeline(
        X, select_percentile=SELECT_PERCENTILE, scale_pos_weight=spw_full
    )
    model.fit(X, y)

    train_pred = model.predict(X)
    train_acc = accuracy_score(y, train_pred)

    lines = [
        f"total_labeled_rows={len(y)}",
        f"test_split_size={TEST_SPLIT_SIZE}",
        f"train_split_rows={len(y_train)}",
        f"test_split_rows={len(y_test)}",
        f"random_state={RANDOM_STATE}",
        f"positive_rate_full={(y == 1).mean():.4f}",
        f"scale_pos_weight_train_split={spw_train:.4f}",
        f"scale_pos_weight_full_fit={spw_full:.4f}",
        "",
        f"cv_folds={CV_FOLDS}  (CV only on train split, not on test split)",
        f"cv_accuracy_mean={cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})",
        f"cv_roc_auc_mean={cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})",
        "",
        "Test set (stratified split from Train_Dataset.csv):",
        f"test_accuracy={test_acc:.4f}",
        f"test_roc_auc={test_auc:.4f}",
        f"test_precision_weighted={test_prec_w:.4f}",
        f"test_recall_weighted={test_rec_w:.4f}",
        f"test_f1_weighted={test_f1_w:.4f}",
        f"test_precision_default_class={test_prec_1:.4f}",
        f"test_recall_default_class={test_rec_1:.4f}",
        f"test_f1_default_class={test_f1_1:.4f}",
        "",
        f"train_fit_accuracy_full_data={train_acc:.4f}  (in-sample on all rows; optimistic)",
        "",
        "Classification report — test set:",
        classification_report(y_test, test_pred, zero_division=0),
        "",
        "Classification report — full data in-sample (for reference):",
        classification_report(y, train_pred, zero_division=0),
        "",
        "Note: Kaggle Test_Dataset.csv often has no Default; test metrics above use a split from Train_Dataset.",
        "Saved model is fit on ALL labeled rows for deployment / Streamlit.",
    ]

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    METRICS_OUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved model to {MODEL_OUT}")
    print(f"Saved metrics to {METRICS_OUT}")
    print(f"CV accuracy (train split): {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
    print(f"CV ROC-AUC (train split): {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")
    print(f"test_accuracy: {test_acc:.4f}")
    print(f"test_roc_auc: {test_auc:.4f}")

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
        X_unlabeled = align_features_for_pipeline(model, df_test[X.columns])
        proba = model.predict_proba(X_unlabeled)
        p_default = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        out = pd.DataFrame(
            {
                "ID": X_unlabeled["ID"] if "ID" in X_unlabeled.columns else np.arange(len(X_unlabeled)),
                "pred_default": model.predict(X_unlabeled),
                "p_default": p_default,
            }
        )
        out.to_csv(TEST_PREDS_OUT, index=False)
        print(f"Saved test predictions to {TEST_PREDS_OUT}")
    else:
        print(f"No test file at {DATA_TEST}; skip test predictions.")


if __name__ == "__main__":
    main()
