"""
Helpers for loading the XGBoost sklearn Pipeline and building prediction inputs.

The training script saves a standard sklearn/XGBoost pipeline model under
`outputs/xgboost/trained_models/`, and the app loads it directly via ``joblib``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


TARGET_COL = "Default"

# Above this, a ``selectbox`` is usually too large to scroll comfortably; use free-text.
MAX_SELECTBOX_OPTIONS = 5000

# sklearn may route these as categorical (mixed/object dtypes in CSV) but values are
# continuous numbers — use free ``number_input``, not a training-value dropdown.
FREE_NUMERIC_AS_CATEGORICAL_UI: frozenset[str] = frozenset(
    {
        "Client_Income",
        "Credit_Amount",
        "Loan_Annuity",
        "Population_Region_Relative",
        "Age_Days",
        "Employed_Days",
        "Registration_Days",
        "ID_Days",
        "Score_Source_3",
    }
)

# Not shown in the borrower form: internal / process metadata. Still passed to the model
# using median/mode from the training reference (`default_values_for_row`).
AUTOFILL_HIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "ID",
        "Application_Process_Day",
        "Application_Process_Hour",
    }
)


def split_cat_for_ui(cat_cols: list[str]) -> tuple[list[str], list[str]]:
    """(free_numeric_entry, dropdown_or_text) for categorical pipeline columns."""
    free = [c for c in cat_cols if c in FREE_NUMERIC_AS_CATEGORICAL_UI]
    rest = [c for c in cat_cols if c not in FREE_NUMERIC_AS_CATEGORICAL_UI]
    return free, rest


def visible_columns_for_borrower_form(
    num_cols: list[str],
    cat_free_num: list[str],
    cat_discrete: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Drop autofill-only columns from the single-application form."""
    h = AUTOFILL_HIDDEN_COLUMNS
    return (
        [c for c in num_cols if c not in h],
        [c for c in cat_free_num if c not in h],
        [c for c in cat_discrete if c not in h],
    )


def merge_form_with_autofill(
    num_cols: list[str],
    cat_cols: list[str],
    *,
    form_values: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Start from training defaults, override with user-entered visible fields only."""
    out = {**defaults}
    for k, v in form_values.items():
        out[k] = v
    # Ensure every pipeline column is present
    for c in num_cols + cat_cols:
        if c not in out:
            out[c] = defaults.get(c, 0.0 if c in num_cols else "")
    return out


def _value_to_ohe_string(v: Any) -> Any:
    """String for OHE; normalize numeric-like values for stable category text."""
    if v is None or (isinstance(v, str) and not str(v).strip()):
        return np.nan
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if pd.isna(v):
            return np.nan
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
        return str(fv)
    return str(v).strip() or np.nan


def repair_sklearn_pipeline_after_unpickle(estimator: Any) -> None:
    """
    Older ``joblib`` dumps of ``SimpleImputer`` may not include ``_fill_dtype``, which
    newer scikit-learn expects in ``transform``. Set it from ``statistics_.dtype``
    so predictions work after upgrading sklearn (or mixing notebook vs app env).
    """
    def walk(obj: Any) -> None:
        if isinstance(obj, SimpleImputer):
            if getattr(obj, "_fill_dtype", None) is None:
                st = getattr(obj, "statistics_", None)
                if st is not None and getattr(st, "size", 0) > 0:
                    obj._fill_dtype = st.dtype
                else:
                    obj._fill_dtype = np.dtype(np.float64)
            return
        if isinstance(obj, Pipeline):
            for _, step in obj.steps:
                walk(step)
            return
        if isinstance(obj, ColumnTransformer):
            for _name, trans, _cols in getattr(obj, "transformers_", []) or []:
                if trans is not None and trans != "drop":
                    walk(trans)
            return

    walk(estimator)


def extract_num_cat_columns(model) -> tuple[list[str], list[str]]:
    """Read numeric / categorical column lists from a fitted ``ColumnTransformer``."""
    prep = model.named_steps["prep"]
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for name, _trans, cols in prep.transformers_:
        if name in ("remainder", "drop"):
            continue
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)
    if not num_cols and not cat_cols:
        raise ValueError("Could not read column groups from the loaded model.")
    return num_cols, cat_cols


def load_reference_X(
    data_path: Path, *, nrows: int | None = 15_000
) -> pd.DataFrame:
    """Sample of training data for sensible defaults (medians / modes)."""
    df = pd.read_csv(data_path, low_memory=False, nrows=nrows)
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    return df


def load_categorical_uniques(data_path: Path, cat_cols: list[str]) -> dict[str, list[str]]:
    """
    Distinct values per categorical column from the full training CSV (``usecols`` only).

    Sorted string values for stable dropdown order. Empty list if the file or column is missing.
    """
    if not data_path.is_file() or not cat_cols:
        return {c: [] for c in cat_cols}
    header = pd.read_csv(data_path, nrows=0)
    cols = [c for c in cat_cols if c in header.columns]
    if not cols:
        return {c: [] for c in cat_cols}
    df = pd.read_csv(data_path, usecols=cols, low_memory=False)
    out: dict[str, list[str]] = {}
    for c in cat_cols:
        if c not in df.columns:
            out[c] = []
            continue
        u = df[c].dropna().astype(str).str.strip()
        u = u[u != ""]
        out[c] = sorted(set(u.tolist()))
    return out


def default_values_for_row(
    X_ref: pd.DataFrame, num_cols: list[str], cat_cols: list[str]
) -> dict[str, Any]:
    """Median for numeric pipeline columns, mode string for categorical."""
    row: dict[str, Any] = {}
    for c in num_cols:
        if c not in X_ref.columns:
            row[c] = 0.0
            continue
        s = pd.to_numeric(X_ref[c], errors="coerce")
        m = float(np.nanmedian(s))
        row[c] = m
    for c in cat_cols:
        if c not in X_ref.columns:
            row[c] = "" if c not in FREE_NUMERIC_AS_CATEGORICAL_UI else 0.0
            continue
        if c in FREE_NUMERIC_AS_CATEGORICAL_UI:
            s = pd.to_numeric(X_ref[c], errors="coerce")
            row[c] = float(np.nanmedian(s))
        else:
            mode = X_ref[c].mode(dropna=True)
            row[c] = str(mode.iloc[0]) if len(mode) else ""
    return row


def build_single_row(
    num_cols: list[str],
    cat_cols: list[str],
    values: dict[str, Any],
) -> pd.DataFrame:
    """Build one-row DataFrame with dtypes compatible with the fitted pipeline."""
    row: dict[str, Any] = {}
    for c in num_cols:
        v = values.get(c)
        if v is None or (isinstance(v, str) and not str(v).strip()):
            row[c] = np.nan
        else:
            try:
                row[c] = float(v)
            except (TypeError, ValueError):
                row[c] = np.nan
    for c in cat_cols:
        v = values.get(c)
        row[c] = _value_to_ohe_string(v)
    df = pd.DataFrame([row])
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in cat_cols:
        df[c] = df[c].astype("object")
    return df


def load_model(path: Path):
    m = joblib.load(path)
    repair_sklearn_pipeline_after_unpickle(m)
    return m


def predict_binary(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return class labels and probability of the positive class (Default=1)."""
    proba = model.predict_proba(X)
    if proba.shape[1] < 2:
        pos = proba[:, 0]
    else:
        pos = proba[:, 1]
    pred = model.predict(X)
    return pred, pos
