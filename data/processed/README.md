# Processed data

Use this folder for **derived** tables after cleaning, encoding, imputation, or train/validation splits saved as CSV/Parquet.

Examples:

- `train_clean.csv`, `test_clean.csv`
- One-hot-encoded or scaled features ready for modeling

Add scripts or notebooks that write here under `notebooks/` or a future `src/` pipeline. The default `models/*.py` scripts read from `data/raw/`; switch paths in code if you standardize on processed files.
