# Outputs (one folder per algorithm)

Each algorithm writes only under its own directory:

```text
outputs/
  random_forest/
    trained_models/   # .joblib files
    metrics/          # accuracy, reports
  xgboost/
    trained_models/
    metrics/
  logistic_regression/
    trained_models/
    metrics/
  gradient_boosting/
    trained_models/
    metrics/
```

This keeps models, metrics, and experiments separated by algorithm. Shared raw data stays in `data/raw/`.
