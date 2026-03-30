import streamlit as st

st.set_page_config(page_title="Loan Prediction Frontend", layout="centered")
st.title("Loan Prediction - Team Model Frontend")

st.write(
    "Trained models load from per-algorithm folders, e.g. "
    "`outputs/xgboost/trained_models/`. See `outputs/README.md`."
)

model_choice = st.selectbox(
    "Select model",
    [
        "Random Forest",
        "XGBoost",
        "Logistic Regression",
        "Gradient Boosting",
    ],
)

st.info(f"Selected: {model_choice}")
st.write("Next step: add input fields and prediction logic.")
