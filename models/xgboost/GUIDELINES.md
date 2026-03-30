# XGBoost (`models/xgboost/`)

## Files

| File | Role |
|------|------|
| `preprocessing_pipeline.py` | Normalization (numeric `StandardScaler`), imputation, **sparse** one-hot (`ColumnTransformer` + `OneHotEncoder`), **variance filter**, **ANOVA F-score feature selection** (`SelectPercentile` with `f_classif`), `XGBClassifier` |
| `train.py` | CLI training using the pipeline above; saves `outputs/xgboost/` |

## Run

```bash
python models/xgboost/train.py
```

## Tuning (performance)

In `train.py` or `build_xgb_pipeline()`, adjust `select_percentile` (default **40**):

- **Lower** (e.g. 25–30): fewer features, can reduce overfitting; may lose signal.
- **Higher** (e.g. 50–60): more features retained.

Re-run and compare test accuracy / ROC-AUC on a fixed `random_state`.

## Notebook

`notebooks/xgboost/xgboost_training.ipynb` uses the same `preprocessing_pipeline` module.
