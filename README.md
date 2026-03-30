# Loan Prediction Project Structure

This repository is organized so each team member trains a different ML algorithm on the same dataset.

Dataset used: `Train_Dataset.csv` from Kaggle (Dish Network Hackathon dataset).

## Folder Structure

```text
ML-ASSIGNMENT/
  data/
    raw/
      Train_Dataset.csv
    processed/
  frontend/
    app.py
  models/
    README.md
    random_forest.py
    xgboost.py
    logistic_regression.py
    gradient_boosting.py
  outputs/
    trained_models/
    metrics/
  requirements.txt
```

## Team Workflow

1. Place Kaggle file as `data/raw/Train_Dataset.csv`.
2. In each training script, set `TARGET_COL` to your label column name.
3. Each member works on one of the four scripts in `models/`.
4. Trained models are saved to `outputs/trained_models/`.
5. Metrics are saved to `outputs/metrics/`.

## Run Example

From the project root (`ML-ASSIGNMENT/`):

```bash
python models/random_forest.py
python models/xgboost.py
python models/logistic_regression.py
python models/gradient_boosting.py
```
