"""
Preprocessing + XGBoost pipeline for loan default prediction.

Memory-safe for large categorical cardinality:
- OneHotEncoder uses **sparse** output (avoids dense  n_samples × n_ohe  matrices).
- ColumnTransformer uses **sparse** stacking when supported.
- Feature selection uses **ANOVA F-score** (works on sparse inputs); mutual information
  would densify the matrix and can cause MemoryError on wide OHE.

- CastCategoricalToString: object/category columns → str.
- Numeric: median imputation + StandardScaler.
- Categorical: most-frequent imputation + sparse one-hot.
- VarianceThreshold, SelectPercentile, XGBoost (sparse-capable).
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier


class CastCategoricalToString(BaseEstimator, TransformerMixin):
    """
    Coerce ``object`` / ``category`` columns to plain strings.

    Kaggle-style CSVs often mix numbers and text in the same column (e.g. ``1`` vs ``"1"``),
    which makes ``OneHotEncoder`` fail with *uniformly strings or numbers*.
    Non-null values become ``str``; missing values stay missing for the imputer.
    """

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.columns_ = X.columns.tolist()
        self.cat_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_)
        else:
            X = X.copy()
        for c in self.cat_cols_:
            if c in X.columns:
                X[c] = X[c].map(lambda v: v if pd.isna(v) else str(v))
        return X


def _one_hot_encoder() -> OneHotEncoder:
    """Sparse OHE to avoid allocating dense (n_samples × n_categories) matrices."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _column_transformer(**kwargs) -> ColumnTransformer:
    """Prefer sparse output so numeric + OHE can stack without densifying."""
    try:
        return ColumnTransformer(n_jobs=1, sparse_output=True, **kwargs)
    except TypeError:
        try:
            return ColumnTransformer(
                n_jobs=1, sparse_threshold=1.0, **kwargs
            )
        except TypeError:
            return ColumnTransformer(n_jobs=1, **kwargs)


def _anova_f_scores(X, y):
    """
    ANOVA F-score per feature for ``SelectPercentile``.

    Defined at module scope (not inside ``build_xgb_pipeline``) so the fitted
    pipeline can be pickled with ``joblib`` / ``pickle``.
    """
    return f_classif(X, y)[0]


def build_xgb_pipeline(
    X: pd.DataFrame,
    *,
    select_percentile: float = 40.0,
    variance_threshold: float = 1e-8,
    random_state: int = 42,
) -> Pipeline:
    """
    Build sklearn Pipeline: cast → preprocess (sparse) → variance → ANOVA selection → XGBoost.

    Parameters
    ----------
    select_percentile : float
        Keep top ``select_percentile`` percent of features by ANOVA F-score (0–100).
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )

    preprocessor = _column_transformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
    )

    return Pipeline(
        steps=[
            ("cast_categorical", CastCategoricalToString()),
            ("prep", preprocessor),
            ("var", VarianceThreshold(threshold=variance_threshold)),
            (
                "select",
                SelectPercentile(
                    score_func=_anova_f_scores, percentile=select_percentile
                ),
            ),
            ("clf", clf),
        ]
    )


def feature_importance_series(model: Pipeline) -> pd.Series:
    """
    After ``model.fit(...)``, return feature importances aligned to selected feature names.
    """
    clf = model.named_steps["clf"]
    prep = model.named_steps["prep"]
    var = model.named_steps["var"]
    sel = model.named_steps["select"]
    names = prep.get_feature_names_out()
    names = names[var.get_support()]
    names = names[sel.get_support()]
    if len(names) != len(clf.feature_importances_):
        return pd.Series(clf.feature_importances_)
    return pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False)
