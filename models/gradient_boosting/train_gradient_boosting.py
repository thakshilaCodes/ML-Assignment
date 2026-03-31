"""
LoanPulse — Gradient Boosting Classifier
=========================================
Folder layout (project root):
  models/gradient_boosting/train.py          ← this file
  data/raw/Train_Dataset.csv
  data/processed/train_processed_gb.csv      ← saved by this script
  outputs/gradient_boosting/trained_models/  ← model + preprocessors
  outputs/gradient_boosting/metrics/         ← txt, png, csv artefacts

Run from any directory:
  python models/gradient_boosting/train.py
"""

# ── Stdlib ────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Preprocessing ─────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer

# ── Class imbalance ───────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ── Model ─────────────────────────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier

# ── Evaluation ────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
)

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

# ── Persistence ───────────────────────────────────────────────────────────────
import joblib

# ══════════════════════════════════════════════════════════════════════════════
#  PATH SETUP
# ══════════════════════════════════════════════════════════════════════════════
ALGO_DIR   = Path(__file__).resolve().parent
ROOT       = ALGO_DIR.parent.parent           # project root

DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
OUTPUT_DIR      = ROOT / "outputs" / "gradient_boosting"
METRICS_DIR     = OUTPUT_DIR / "metrics"
TRAINED_DIR     = OUTPUT_DIR / "trained_models"

for d in [DATA_PROCESSED, METRICS_DIR, TRAINED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_RAW / "Train_Dataset.csv"
TEST_CSV  = DATA_RAW / "Test_Dataset.csv"

TARGET    = "Default"
DROP_COLS = [
    "ID", "Mobile_Tag", "Homephone_Tag", "Workphone_Working",
    "Client_Permanent_Match_Tag", "Client_Contact_Work_Tag",
]

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _stratify_if_ok(y: pd.Series):
    if y.nunique() < 2 or y.value_counts().min() < 2:
        return None
    return y


def preprocess_train(df: pd.DataFrame):
    """Clean, encode, impute and scale the training dataframe.
    Returns (X, y, le_map, num_imputer, scaler, num_cols, cat_cols).
    """
    df = df.copy()
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    # Label encode
    le_map = {}
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("Missing")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_map[col] = le

    # Impute numerics
    num_imputer = SimpleImputer(strategy="median")
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

    # Scale
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y, le_map, num_imputer, scaler, num_cols, cat_cols


def preprocess_test(df_test: pd.DataFrame, X_cols, le_map, num_imputer,
                    scaler, num_cols):
    """Apply fitted preprocessing to the test set."""
    df_test = df_test.copy()
    ids = df_test["ID"].copy()
    df_test.drop(columns=[c for c in DROP_COLS if c in df_test.columns],
                 inplace=True)

    test_cat_cols = df_test.select_dtypes(include="object").columns.tolist()
    for col in test_cat_cols:
        df_test[col] = df_test[col].astype(str).fillna("Missing")
        if col in le_map:
            known = set(le_map[col].classes_)
            df_test[col] = df_test[col].apply(
                lambda x: x if x in known else "Missing"
            )
            if "Missing" not in known:
                le_map[col].classes_ = np.append(le_map[col].classes_, "Missing")
            df_test[col] = le_map[col].transform(df_test[col])
        else:
            le_new = LabelEncoder()
            df_test[col] = le_new.fit_transform(df_test[col])

    test_num_cols = [c for c in num_cols if c in df_test.columns]
    df_test[test_num_cols] = num_imputer.transform(df_test[test_num_cols])

    for col in X_cols:
        if col not in df_test.columns:
            df_test[col] = 0
    df_test = df_test[X_cols]
    df_test[test_num_cols] = scaler.transform(df_test[test_num_cols])

    return ids, df_test


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Load ─────────────────────────────────────────────────────────────────
    print("Loading data …")
    train_raw = pd.read_csv(TRAIN_CSV, low_memory=False)
    test_raw  = pd.read_csv(TEST_CSV,  low_memory=False)

    # ── Preprocess ───────────────────────────────────────────────────────────
    print("Preprocessing …")
    X, y, le_map, num_imputer, scaler, num_cols, cat_cols = preprocess_train(train_raw)

    # SMOTE
    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = sm.fit_resample(X, y)
        print(f"  SMOTE → {pd.Series(y_res).value_counts().to_dict()}")
    else:
        X_res, y_res = X, y
        print("  SMOTE skipped (install imbalanced-learn)")

    # Save processed
    proc_df = X_res.copy()
    proc_df[TARGET] = y_res.values
    proc_df.to_csv(DATA_PROCESSED / "train_processed_gb.csv", index=False)

    # ── Train / val split ────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42,
        stratify=_stratify_if_ok(pd.Series(y_res))
    )
    print(f"  Train {X_train.shape[0]:,} | Val {X_val.shape[0]:,}")

    # ── Hyperparameter tuning ─────────────────────────────────────────────────
    print("Running GridSearchCV …")
    param_grid = {
        "n_estimators" : [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth"    : [3, 4],
        "subsample"    : [0.8, 1.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(
        GradientBoostingClassifier(min_samples_split=20, random_state=42),
        param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    best_gb = gs.best_estimator_
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV AUC : {gs.best_score_:.4f}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = best_gb.predict(X_val)
    y_prob = best_gb.predict_proba(X_val)[:, 1]

    acc    = accuracy_score(y_val, y_pred)
    roc    = roc_auc_score(y_val, y_prob)
    report = classification_report(y_val, y_pred,
                                   target_names=["No Default", "Default"])

    print(f"\nValidation Accuracy : {acc:.4f}")
    print(f"Validation ROC-AUC  : {roc:.4f}")
    print(report)

    # ── Plots ─────────────────────────────────────────────────────────────────
    # Confusion matrix + ROC
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cm   = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix — Gradient Boosting", fontweight="bold")

    fpr, tpr, _ = roc_curve(y_val, y_prob)
    axes[1].plot(fpr, tpr, lw=2, color="#1565C0",
                 label=f"GB (AUC = {auc(fpr, tpr):.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve — Gradient Boosting", fontweight="bold")
    axes[1].legend(loc="lower right"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(METRICS_DIR / "confusion_matrix_roc.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importance
    feat_imp = (pd.Series(best_gb.feature_importances_, index=X.columns)
                .sort_values(ascending=False).head(20))
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp[::-1].plot(kind="barh", ax=ax, color="#1976D2", edgecolor="white")
    ax.set_title("Top-20 Feature Importances — Gradient Boosting", fontweight="bold")
    ax.set_xlabel("Importance"); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(METRICS_DIR / "feature_importance.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Training deviance
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(best_gb.train_score_) + 1), best_gb.train_score_,
            lw=1.5, color="#43A047")
    ax.set_xlabel("Boosting iterations"); ax.set_ylabel("Deviance")
    ax.set_title("Training Deviance — Gradient Boosting", fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(METRICS_DIR / "training_deviance.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save model & preprocessors ────────────────────────────────────────────
    joblib.dump(best_gb,    TRAINED_DIR / "gradient_boosting.joblib")
    joblib.dump(scaler,     TRAINED_DIR / "scaler.joblib")
    joblib.dump(num_imputer,TRAINED_DIR / "num_imputer.joblib")
    joblib.dump(le_map,     TRAINED_DIR / "label_encoders.joblib")

    # ── Save metrics text ─────────────────────────────────────────────────────
    metrics_text = (
        f"Gradient Boosting — Validation Metrics\n"
        f"========================================\n"
        f"Accuracy  : {acc:.4f}\n"
        f"ROC-AUC   : {roc:.4f}\n\n"
        f"Best Hyperparameters:\n{gs.best_params_}\n\n"
        f"Classification Report:\n{report}\n"
    )
    (METRICS_DIR / "gradient_boosting.txt").write_text(
        metrics_text, encoding="utf-8"
    )

    # ── Predict on test set ───────────────────────────────────────────────────
    ids, X_test = preprocess_test(
        test_raw, X.columns, le_map, num_imputer, scaler, num_cols
    )
    test_preds = best_gb.predict(X_test)
    test_probs = best_gb.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        "ID"                  : ids.values,
        "Default_Predicted"   : test_preds,
        "Default_Probability" : test_probs.round(4),
    })
    submission.to_csv(METRICS_DIR / "test_predictions.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n═══ Outputs saved ═══")
    print(f"  Model       → {TRAINED_DIR / 'gradient_boosting.joblib'}")
    print(f"  Metrics txt → {METRICS_DIR / 'gradient_boosting.txt'}")
    print(f"  Plots       → {METRICS_DIR}")
    print(f"  Test preds  → {METRICS_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
