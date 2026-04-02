"""
Loan default prediction UI — loads the trained XGBoost pipeline from ``outputs/xgboost/``.

Run from project root::

    streamlit run frontend/app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from feature_labels import (  # noqa: E402
    SECTION_CAPTIONS,
    SECTION_ICONS,
    SECTION_TITLES,
    YES_NO_NUMERIC_COLUMNS,
    bounded_number_input_kwargs,
    display_label,
    field_help,
    field_help_cat_numeric,
    group_by_section,
    ordered_form_fields,
    yes_no_radio_index,
)
from predict_utils import (  # noqa: E402
    AUTOFILL_HIDDEN_COLUMNS,
    MAX_SELECTBOX_OPTIONS,
    TARGET_COL,
    build_single_row,
    default_values_for_row,
    extract_num_cat_columns,
    load_categorical_uniques,
    load_model,
    load_reference_X,
    merge_form_with_autofill,
    predict_binary,
    repair_sklearn_pipeline_after_unpickle,
    split_cat_for_ui,
    visible_columns_for_borrower_form,
)

DEFAULT_MODEL_PATH = _PROJECT_ROOT / "outputs" / "xgboost" / "trained_models" / "xgboost.joblib"
DATA_PATH = _PROJECT_ROOT / "data" / "raw" / "Train_Dataset.csv"


def _css():
    st.markdown(
        """
        <style>
        html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        /* Use full main column width (removes the old narrow 920px column and large side gutters) */
        div[data-testid="stAppViewContainer"] .main div.block-container,
        .main > div.block-container {
            max-width: 100% !important;
            padding-left: clamp(1rem, 3vw, 2.5rem) !important;
            padding-right: clamp(1rem, 3vw, 2.5rem) !important;
        }
        .block-container { padding-top: 1.25rem; padding-bottom: 3rem; }
        /* Slightly more air between the two form columns */
        div[data-testid="column"] { padding-left: 0.65rem !important; padding-right: 0.65rem !important; }
        h1, h2, h3 { font-weight: 600; letter-spacing: -0.02em; color: #0f172a !important; }
        [data-testid="stHeader"] { background: linear-gradient(180deg, #fafbfc 0%, #f1f5f9 100%); border-bottom: 1px solid #e2e8f0; }
        div[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
        div[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
        div[data-testid="stMetricValue"] { font-size: 1.65rem; font-weight: 600; color: #0f172a; }
        div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #64748b; text-transform: none; letter-spacing: 0; }
        .app-hero { background: linear-gradient(145deg, #ecfeff 0%, #e0f2fe 40%, #f8fafc 100%); border: 1px solid #bae6fd; border-radius: 16px; padding: 1.5rem 1.75rem; margin-bottom: 1.75rem; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }
        .app-hero h1 { margin: 0 0 0.5rem 0; font-size: 1.85rem; font-weight: 700; }
        .app-hero p { margin: 0; color: #475569; font-size: 1.05rem; line-height: 1.55; }
        .form-section-title { font-size: 1.05rem; font-weight: 600; color: #0f172a; margin: 0 0 0.15rem 0; }
        .form-section-wrap { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.1rem 1.25rem 1.25rem; margin-bottom: 1.35rem; box-shadow: 0 1px 3px rgba(15,23,42,0.04); }
        [data-baseweb="tab-list"] { gap: 0.35rem; background: #f1f5f9; padding: 0.4rem; border-radius: 12px; }
        [data-baseweb="tab"] { border-radius: 8px !important; }
        button[kind="primary"] { background: #0284c7 !important; border: none !important; font-weight: 600 !important; border-radius: 10px !important; padding: 0.65rem 1.25rem !important; box-shadow: 0 1px 2px rgba(2,132,199,0.25); }
        button[kind="primary"]:hover { background: #0369a1 !important; }
        label[data-testid="stWidgetLabel"] p { font-size: 0.95rem; color: #334155; }
        .steps-row { display: flex; flex-wrap: wrap; gap: 1rem; margin: 0 0 1.25rem 0; }
        .step-pill { flex: 1; min-width: 140px; background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 0.85rem 1rem; text-align: center; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }
        .step-pill strong { display: block; color: #0f172a; font-size: 0.95rem; margin-bottom: 0.25rem; }
        .step-pill span { font-size: 0.8rem; color: #64748b; line-height: 1.35; }
        div[data-testid="stExpander"] details { border-radius: 12px; border: 1px solid #e2e8f0; background: #fafbfc; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="app-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def _steps_banner() -> None:
    """Short 3-step guide so users know what to do next."""
    st.markdown(
        """
        <div class="steps-row">
          <div class="step-pill"><strong>1 · Model</strong><span>Loaded from the sidebar (or upload)</span></div>
          <div class="step-pill"><strong>2 · Details</strong><span>Fill the sections below — skip what you don’t know</span></div>
          <div class="step-pill"><strong>3 · Result</strong><span>Submit to see default risk as a %</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def _cached_model(path_str: str):
    return load_model(Path(path_str))


@st.cache_data
def _cached_reference():
    if not DATA_PATH.is_file():
        return None
    return load_reference_X(DATA_PATH)


@st.cache_data
def _cached_cat_uniques(cat_cols: tuple[str, ...]) -> dict[str, list[str]]:
    """Unique categorical values from full training CSV (for dropdowns)."""
    if not DATA_PATH.is_file():
        return {}
    return load_categorical_uniques(DATA_PATH, list(cat_cols))


def main():
    st.set_page_config(
        page_title="Loan risk check",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _css()

    _hero(
        "See how risky this loan looks",
        "Answer a few questions about the applicant. The tool estimates the chance they might default — "
        "it’s a guide only, not a lending decision. Hover the **ⓘ** icons if a question is unclear.",
    )

    # —— Sidebar: model source ——
    with st.sidebar:
        st.markdown("### Prediction model")
        st.caption("The app needs your trained `.joblib` file once.")
        use_upload = st.checkbox("Upload a model file instead", value=False)
        if use_upload:
            up = st.file_uploader(
                "Choose file",
                type=["joblib", "pkl", "pickle"],
                help="Select the saved model file from your training run.",
            )
            model_path = None
            model_obj = None
            if up is not None:
                try:
                    model_obj = joblib.load(io.BytesIO(up.getvalue()))
                    repair_sklearn_pipeline_after_unpickle(model_obj)
                    st.success("Model ready.")
                except Exception as e:
                    st.error(f"Could not load this file: {e}")
        else:
            default_txt = str(DEFAULT_MODEL_PATH)
            model_path = st.text_input(
                "Where is the model saved?",
                value=default_txt,
                help="Full path to the .joblib file on this PC. The default matches the usual project folder.",
            )
            model_obj = None
            p = Path(model_path)
            if p.is_file():
                try:
                    model_obj = _cached_model(str(p.resolve()))
                    st.success("Model ready.")
                except Exception as e:
                    st.error(f"Could not load: {e}")
            else:
                st.warning("Model file not found. Check the path or upload a file above.")

        with st.expander("How this app works", expanded=False):
            st.markdown(
                """
1. **Load the model** — same file produced by `python models/xgboost/train.py`.
2. **Enter one applicant** (or upload a CSV on the second tab).
3. **Read the %** — higher = model thinks default is more likely. Not a credit decision.
"""
            )

    if model_obj is None:
        st.info(
            "Choose or upload a trained model in the **sidebar** to run a check. "
            "If you don’t have the file on this computer, get it from whoever trained the model."
        )
        return

    try:
        num_cols, cat_cols = extract_num_cat_columns(model_obj)
    except Exception as e:
        st.error(f"Could not read model columns: {e}")
        return

    ref = _cached_reference()
    if ref is not None:
        defaults = default_values_for_row(ref, num_cols, cat_cols)
    else:
        defaults = {c: 0.0 for c in num_cols}
        defaults.update({c: "" for c in cat_cols})

    cat_uniques = _cached_cat_uniques(tuple(cat_cols))
    cat_free_num, cat_discrete = split_cat_for_ui(cat_cols)

    _steps_banner()

    tab1, tab2 = st.tabs(["One applicant", "Many rows (CSV)"])

    with tab1:
        _single_form_ui(
            model_obj,
            num_cols,
            cat_cols,
            cat_free_num,
            cat_discrete,
            defaults,
            cat_uniques,
        )

    with tab2:
        _batch_csv_ui(model_obj, num_cols, cat_cols, defaults)


def _single_form_ui(
    model,
    num_cols: list[str],
    cat_cols: list[str],
    cat_free_num: list[str],
    cat_discrete: list[str],
    defaults: dict,
    cat_uniques: dict[str, list[str]],
):
    vis_num, vis_free_cat, vis_discrete = visible_columns_for_borrower_form(
        num_cols, cat_free_num, cat_discrete
    )
    ordered = ordered_form_fields(vis_num, vis_free_cat, vis_discrete)
    sections = group_by_section(ordered)

    def _put_field(kind: str, col: str, values: dict) -> None:
        if kind == "num":
            d = defaults.get(col, 0.0)
            if col in YES_NO_NUMERIC_COLUMNS:
                choice = st.radio(
                    display_label(col),
                    ("No", "Yes"),
                    index=yes_no_radio_index(d),
                    horizontal=True,
                    key=f"yn_{col}",
                    help=field_help(col),
                )
                values[col] = 1.0 if choice == "Yes" else 0.0
            else:
                kw = bounded_number_input_kwargs(col, d)
                values[col] = st.number_input(
                    display_label(col),
                    key=f"num_{col}",
                    help=field_help(col),
                    **kw,
                )
            return
        if kind == "free_cat":
            d = defaults.get(col, 0.0)
            try:
                dv = float(d) if pd.notna(d) else 0.0
            except (TypeError, ValueError):
                dv = 0.0
            values[col] = st.number_input(
                display_label(col),
                value=dv,
                format="%.6g",
                key=f"catnum_{col}",
                help=field_help(col) + " " + field_help_cat_numeric(),
            )
            return
        d = defaults.get(col, "")
        d_str = "" if pd.isna(d) else str(d)
        opts = cat_uniques.get(col, [])
        h_base = field_help(col)
        if opts and len(opts) <= MAX_SELECTBOX_OPTIONS:
            idx = opts.index(d_str) if d_str in opts else 0
            values[f"_cat_sel_{col}"] = st.selectbox(
                display_label(col),
                options=opts,
                index=min(idx, len(opts) - 1),
                key=f"cat_sel_{col}",
                help=h_base + " Pick the closest match, or type your own wording below.",
            )
            values[f"_cat_ov_{col}"] = st.text_input(
                "Or type your own value",
                value="",
                key=f"cat_ov_{col}",
                label_visibility="visible",
                placeholder="Leave empty to use the list above",
                help="Only fill this if the dropdown doesn’t have the right wording.",
            )
        elif opts:
            values[col] = st.text_input(
                display_label(col),
                value=d_str,
                key=f"cat_{col}",
                placeholder="Type the value as it appears in your records",
                help=h_base + f" ({len(opts)} options exist in past data — spelling should be similar.)",
            )
        else:
            values[col] = st.text_input(
                display_label(col),
                value=d_str,
                key=f"cat_{col}",
                placeholder="Enter the value",
                help=h_base + " No list loaded — enter something sensible.",
            )

    with st.form("predict_form", clear_on_submit=False):
        st.markdown("#### Enter applicant details")
        st.info(
            "**Tips:** Use **ⓘ** next to any field for examples. "
            "Defaults come from typical training data — change only what you know. "
            "When a list doesn’t match, use the text box under it."
        )
        st.caption(
            "Each topic is in its own **section** below — click a section title to open or close it. "
            "Fill every section before you submit."
        )

        values: dict = {}
        for sec_idx, (sec_key, pairs) in enumerate(sections):
            icon = SECTION_ICONS.get(sec_key, "•")
            title = SECTION_TITLES.get(sec_key, "Details")
            cap = SECTION_CAPTIONS.get(sec_key, "")
            # Expand first section by default so the form doesn’t feel like a wall; others open too for scanning.
            exp_label = f"{icon}  {title}"
            # First section open by default; others collapsed to reduce scrolling — click to expand.
            with st.expander(exp_label, expanded=(sec_idx == 0)):
                st.caption(cap)
                left, right = st.columns(2)
                mid = (len(pairs) + 1) // 2
                for j, (kind, col) in enumerate(pairs):
                    with (left if j < mid else right):
                        _put_field(kind, col, values)

        submitted = st.form_submit_button("Calculate default risk", type="primary", use_container_width=True)

    if submitted:
        try:
            form_values = {}
            for col in vis_num:
                form_values[col] = values[col]
            for col in vis_free_cat:
                form_values[col] = values[col]
            for col in vis_discrete:
                opts = cat_uniques.get(col, [])
                if opts and len(opts) <= MAX_SELECTBOX_OPTIONS:
                    custom = (values.get(f"_cat_ov_{col}", "") or "").strip()
                    if custom:
                        form_values[col] = custom
                    else:
                        form_values[col] = values[f"_cat_sel_{col}"]
                else:
                    form_values[col] = values[col]

            merged = merge_form_with_autofill(
                num_cols, cat_cols, form_values=form_values, defaults=defaults
            )
            X = build_single_row(num_cols, cat_cols, merged)
            pred, pos = predict_binary(model, X)
            label = int(pred[0])
            p_default = float(pos[0])

            st.divider()
            st.markdown("#### Result")
            st.caption("Model output — interpret as a rough score, not a decision.")

            m1, m2, m3 = st.columns(3)
            with m1:
                outcome = "Higher risk profile" if label == 1 else "Lower risk profile"
                st.metric("Summary", outcome)
            with m2:
                st.metric("Estimated chance of default", f"{p_default:.1%}")
            with m3:
                st.metric("Estimated chance of on-time repayment", f"{1 - p_default:.1%}")

            st.progress(min(float(p_default), 1.0))
            st.caption(
                f"Risk bar: more filled = higher estimated default chance ({p_default:.1%}). "
                "0% = left, 100% = full."
            )

            if label == 1:
                st.warning(
                    "**Higher predicted default risk** for this input. "
                    "Use alongside policy and human review — not as approve/decline by itself."
                )
            else:
                st.success(
                    "**Lower predicted default risk** for this input. "
                    "Still not a guarantee of repayment."
                )
            st.caption(
                "Educational / internal use. Does not replace underwriting, fair lending, or human judgment."
            )
        except Exception as e:
            st.exception(e)


def _batch_csv_ui(
    model,
    num_cols: list[str],
    cat_cols: list[str],
    defaults: dict,
):
    st.markdown("**Batch scoring** — one row per applicant, same columns as training.")
    st.info(
        "1. Export your table as **CSV**.  \n"
        "2. Column names must **match** the training file (technical names).  \n"
        "3. A `Default` column is optional — it will be ignored if present."
    )
    if AUTOFILL_HIDDEN_COLUMNS:
        st.caption(
            "You can omit application ID and timing columns — we fill sensible defaults."
        )
    f = st.file_uploader(
        "Drop your CSV here or browse",
        type=["csv"],
        help="Same structure as Train_Dataset.csv (without needing the Default column for new applicants).",
    )
    if f is None:
        return

    try:
        raw = pd.read_csv(f, low_memory=False)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    if TARGET_COL in raw.columns:
        raw = raw.drop(columns=[TARGET_COL])

    expected = [c for c in num_cols + cat_cols]
    missing = [c for c in expected if c not in raw.columns]
    auto_fill = [c for c in missing if c in AUTOFILL_HIDDEN_COLUMNS]
    must_have = [c for c in missing if c not in AUTOFILL_HIDDEN_COLUMNS]
    if must_have:
        pretty = "\n".join(
            f"- {display_label(c)} (`{c}`)" for c in must_have[:30]
        )
        more = f"\n… and {len(must_have) - 30} more." if len(must_have) > 30 else ""
        st.error(
            f"Missing **{len(must_have)}** required column(s). Add these headers to your CSV:\n\n{pretty}{more}"
        )
        return

    for c in auto_fill:
        raw[c] = defaults[c]

    raw = raw[expected]
    for c in num_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    for c in cat_cols:
        raw[c] = raw[c].apply(
            lambda v: np.nan
            if (pd.isna(v) or (isinstance(v, str) and not str(v).strip()))
            else str(v).strip()
        )

    n = len(raw)
    if n > 50_000:
        st.warning("Large file — only first 50,000 rows will be scored.")
        raw = raw.iloc[:50_000]

    if st.button("Score all rows", type="primary", use_container_width=False):
        try:
            proba = model.predict_proba(raw)
            pos = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            pred = model.predict(raw)
            out = raw.copy()
            out["pred_default"] = pred.astype(int)
            out["p_default"] = pos
            st.dataframe(out.head(50), use_container_width=True)
            st.write(f"Scored **{len(out)}** rows.")
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results",
                data=csv_bytes,
                file_name="loan_default_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.exception(e)


if __name__ == "__main__":
    main()
