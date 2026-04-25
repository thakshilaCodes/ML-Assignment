# Loan Prediction Project Structure

This repository is organized so each team member trains a different ML algorithm on the same dataset.

Dataset used: `Train_Dataset.csv` from Kaggle (Dish Network Hackathon dataset).

## Folder Structure

```text
ML-ASSIGNMENT/
  data/
    README.md
    raw/
    processed/
  notebooks/
    README.md
    random_forest/
    xgboost/
    logistic_regression/
    gradient_boosting/
  frontend/
    README.md
    app.py
  models/
    README.md
    random_forest/train.py
    xgboost/train.py
    xgboost/GUIDELINES.md
    logistic_regression/train.py
    gradient_boosting/train.py
  outputs/
    README.md
    random_forest/trained_models/  metrics/
    xgboost/trained_models/        metrics/
    logistic_regression/...
    gradient_boosting/...
  requirements.txt
```

Empty or placeholder folders may include a short `README.md` so Git tracks them.

## Team Workflow

1. Place Kaggle file as `data/raw/Train_Dataset.csv`.
2. Each algorithm runs from its folder under `models/<algorithm>/train.py`.
3. Outputs go to `outputs/<algorithm>/trained_models/` and `outputs/<algorithm>/metrics/`.
4. Notebooks for each algorithm live under `notebooks/<algorithm>/`.

Target column for this dataset: **`Default`** (1 = defaulted, 0 = otherwise).

## Run Example

From the project root (`ML-ASSIGNMENT/`):

```bash
python models/random_forest/train.py
python models/xgboost/train.py
python models/logistic_regression/train.py
python models/gradient_boosting/train.py
```

## Frontend (XGBoost predictions)

After training XGBoost (default app path: `outputs/xgboost/trained_models/xgboost.joblib` from `train.py`), run the Streamlit UI from the project root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run frontend/app.py
```

Use `python -m streamlit` (not bare `streamlit`) if Windows reports “streamlit is not recognized” — the `Scripts` folder is often missing from `PATH`.

See `frontend/README.md` for upload vs path, batch CSV, and layout details.
