# Models (one folder per algorithm)

Each algorithm has its own directory and a `train.py` entry point.

| Folder | Run from project root |
|--------|------------------------|
| `random_forest/` | `python models/random_forest/train.py` |
| `xgboost/` | `python models/xgboost/train.py` |
| `logistic_regression/` | `python models/logistic_regression/train.py` |
| `gradient_boosting/` | `python models/gradient_boosting/train.py` |

Artifacts are written under `outputs/<algorithm_name>/` (see `outputs/README.md`). Each `train.py` sets `ALGO_NAME` from its folder name (`Path(__file__).parent.name`), so paths stay in sync if folders are renamed consistently.

XGBoost notes: `xgboost/GUIDELINES.md`.
