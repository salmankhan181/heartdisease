import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from src.utils import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Heart Disease Predictor")
st.title("Heart Disease Prediction")

model_name = st.selectbox(
    "Choose Model",
    ["logistic_regression", "random_forest"]
)

model = joblib.load(MODEL_DIR / f"{model_name}.joblib")

user_data = {}
for col in FEATURE_COLUMNS:
    user_data[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    df_input = pd.DataFrame([user_data])
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    if pred == 1:
        st.error(f"High Risk ({prob:.2%})")
    else:
        st.success(f"Low Risk ({prob:.2%})")
