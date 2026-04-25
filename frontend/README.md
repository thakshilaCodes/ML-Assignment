# Frontend (XGBoost predictions)

Streamlit app for **loan default** predictions using the trained pipeline saved under `outputs/xgboost/trained_models/`.

## Prerequisites

1. Train XGBoost and save the model — either:

   - **Script (default in the app):** `python models/xgboost/train.py` → `outputs/xgboost/trained_models/xgboost.joblib`  
   - **Notebook:** `notebooks/xgboost/xgboost_training.ipynb` → `outputs/xgboost/trained_models/xgboost_notebook.joblib` (set the path in the sidebar or upload)

2. Optional: `data/raw/Train_Dataset.csv` is used only to fill **default values** in the form (medians / modes). If the file is missing, defaults fall back to zeros / empty strings.

## Run

From the **project root** (`ML-ASSIGNMENT/`):

```bash
python -m pip install -r requirements.txt
python -m streamlit run frontend/app.py
```

If `streamlit` alone is not found (common on Windows), always use **`python -m streamlit`** so the active interpreter runs the app. Add `Scripts` to `PATH` if you prefer the `streamlit` command.

## What it does

- **Sidebar:** default path is `xgboost.joblib` from `train.py`; point to `xgboost_notebook.joblib` or **upload** if you trained only in the notebook.
- **Single application:** numeric fields; **continuous** pipeline-categorical fields use **free number inputs**; other text categories use a **dropdown** of training values when the list is small enough, plus an **optional custom** text line to override. **Application ID**, **application day**, and **application hour** are auto-filled (`AUTOFILL_HIDDEN_COLUMNS` in `predict_utils.py`).
- **Batch CSV:** score many rows; columns must match training features (`Default` is ignored if present). Download results with `pred_default` and `p_default`.

## Notes

- Model loading uses `joblib` directly from `outputs/xgboost/trained_models/xgboost.joblib`; no extra `sys.path` setup is required.
- **scikit-learn:** train/save and the Streamlit app should use the **same** `scikit-learn` version when possible. If you see `SimpleImputer` / `_fill_dtype` errors on old `.joblib` files, the app patches common cases; when in doubt, re-save the model from the same environment as Streamlit.
- If you move the project, keep the same layout or adjust the path in the sidebar.

## Files

| File | Role |
|------|------|
| `app.py` | Streamlit UI (single + batch prediction) |
| `feature_labels.py` | Human-readable labels for dataset columns (UI only; CSV headers stay technical) |
| `predict_utils.py` | Model column extraction, row building, `predict_proba` helpers |
