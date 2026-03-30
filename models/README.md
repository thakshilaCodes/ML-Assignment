# Models Folder Guide

There are four training scripts (one algorithm each):

| Script | Algorithm |
|--------|-----------|
| `random_forest.py` | Random Forest |
| `xgboost.py` | XGBoost |
| `logistic_regression.py` | Logistic Regression |
| `gradient_boosting.py` | Gradient Boosting (sklearn) |

All scripts:

1. Read `data/raw/Train_Dataset.csv` (paths resolve from project root).
2. Save the trained model to `outputs/trained_models/<name>.joblib`.
3. Save metrics to `outputs/metrics/<name>.txt`.
4. Set `TARGET_COL` in each file to the correct label column in `Train_Dataset.csv`.

Run from project root: `python models/random_forest.py` (and similarly for the other three).
