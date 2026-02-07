from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "heart.csv"
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "heart_disease_model.pkl"

FEATURES = ["age", "sex", "trestbps", "chol", "cp", "thalach", "fbs", "restecg", "exang"]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["target"]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


@st.cache_resource

def get_or_train_model() -> RandomForestClassifier:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    X, y = load_data()
    return train_model(X, y)


def categorize_blood_pressure(bp: int) -> str:
    if bp < 90:
        return "Low"
    if bp <= 120:
        return "Normal"
    return "High"


def categorize_cholesterol(chol: int) -> str:
    if chol < 200:
        return "Normal"
    if chol <= 240:
        return "Borderline High"
    return "High"


def predict_risk(model: RandomForestClassifier, features: np.ndarray) -> int:
    return int(model.predict(features)[0])


def main() -> None:
    st.set_page_config(
        page_title="Heart Disease Risk Chatbot",
        page_icon="🫀",
        layout="centered",
    )

    st.title("AI-Driven Heart Disease Risk Assessment")
    st.write(
        "Enter your health details to estimate risk and get lifestyle recommendations. "
        "This tool is for educational use and not a medical diagnosis."
    )

    with st.sidebar:
        st.header("Input Details")
        age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        trestbps = st.number_input(
            "Resting Blood Pressure (mmHg)", min_value=50, max_value=250, value=120, step=1
        )
        chol = st.number_input(
            "Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1
        )
        cp = st.selectbox(
            "Chest Pain Type",
            options=[
                ("None", 0),
                ("Mild", 1),
                ("Moderate", 2),
                ("Severe", 3),
            ],
            format_func=lambda x: x[0],
        )[1]
        thalach = st.number_input(
            "Max Heart Rate Achieved", min_value=60, max_value=230, value=150, step=1
        )
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox(
            "Resting ECG",
            options=[
                ("Normal", 0),
                ("ST-T wave abnormality", 1),
                ("Possible LVH", 2),
            ],
            format_func=lambda x: x[0],
        )[1]
        exang = st.selectbox(
            "Exercise-Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0]
        )[1]

        run = st.button("Assess Risk")

    if not run:
        st.info("Fill in the inputs and click 'Assess Risk'.")
        return

    model = get_or_train_model()
    features = np.array([[age, sex, trestbps, chol, cp, thalach, fbs, restecg, exang]])

    bp_category = categorize_blood_pressure(int(trestbps))
    chol_category = categorize_cholesterol(int(chol))

    manual_high_risk = trestbps > 120 or chol > 240 or cp >= 2 or exang == 1
    if manual_high_risk:
        risk_label = "High"
    else:
        prediction = predict_risk(model, features)
        risk_label = "High" if prediction == 1 else "No Risk, Healthy"

    st.subheader("Health Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Blood Pressure", bp_category)
    with col2:
        st.metric("Cholesterol", chol_category)

    st.subheader("Risk Prediction")
    if risk_label == "High":
        st.error("High Risk")
    else:
        st.success("No Risk, Healthy")

    st.subheader("Recommended Health Tips")
    recommendations = {
        "High": [
            "Consult a doctor as soon as possible.",
            "Follow a diet low in saturated fats and high in fiber.",
            "Exercise at least 30 minutes daily.",
            "Reduce stress through meditation or yoga.",
            "Quit smoking and limit alcohol intake.",
            "Schedule regular checkups for blood pressure and cholesterol.",
        ],
        "No Risk, Healthy": [
            "Maintain a balanced diet and regular exercise.",
            "Continue periodic monitoring of blood pressure and cholesterol.",
            "Stay active to reduce future risk.",
        ],
    }
    for tip in recommendations[risk_label]:
        st.write(f"- {tip}")


if __name__ == "__main__":
    main()
