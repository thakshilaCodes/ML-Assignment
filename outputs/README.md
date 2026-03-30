# Outputs

Generated artifacts from training and evaluation—not hand-edited source.

- **`trained_models/`** — Serialized models (e.g. `.joblib`) produced by `models/*.py`.
- **`metrics/`** — Text or JSON logs: accuracy, confusion matrix summaries, etc.

Add these paths to `.gitignore` if artifacts must not be pushed; otherwise keep small metric files for the report.
