# Notebooks — XGBoost

- **`xgboost_training.ipynb`** — Load training data, **stratified k-fold CV** (accuracy + ROC-AUC), **fit on the full training set** (no random split), in-sample diagnostics, feature importances, score **`Test_Dataset.csv`**, save model + metrics + test predictions under `outputs/xgboost/`.

**Data:** place `Train_Dataset.csv` and `Test_Dataset.csv` under `data/raw/`. The test file has no `Default` column; the notebook saves predictions only (no test accuracy from the CSV).

**Script:** `models/xgboost/train.py` mirrors the same workflow for CLI use.

Open Jupyter from the project root or `notebooks/xgboost/` (paths auto-resolve). After changing `preprocessing_pipeline.py`, use **Restart Kernel & Run All**.
