from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "heart.csv"
MODEL_DIR = APP_DIR / "models"

ANN_PATH = MODEL_DIR / "salks_ann.h5"
KNN_PATH = MODEL_DIR / "salks_knn.pkl"
META_PATH = MODEL_DIR / "salks_meta.pkl"
SCALER_PATH = MODEL_DIR / "salks_scaler.pkl"

FEATURES = ["age", "sex", "trestbps", "chol", "cp", "thalach", "fbs", "restecg", "exang"]


@st.cache_resource

def get_models():
    if not all(path.exists() for path in [ANN_PATH, KNN_PATH, META_PATH, SCALER_PATH]):
        train_and_save_models()

    ann_model = tf.keras.models.load_model(ANN_PATH)
    knn_model = joblib.load(KNN_PATH)
    meta_model = joblib.load(META_PATH)
    scaler = joblib.load(SCALER_PATH)
    return ann_model, knn_model, meta_model, scaler


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    X = df[FEATURES]
    y = df["target"]
    return X, y


def build_ann(input_dim: int) -> tf.keras.Model:
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_save_models() -> None:
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    ann_model = build_ann(X_train_scaled.shape[1])
    ann_model.fit(
        X_train_scaled,
        y_train_resampled,
        epochs=40,
        batch_size=16,
        verbose=0,
    )

    knn_model = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="manhattan")
    knn_model.fit(X_train_scaled, y_train_resampled)

    ann_probs = ann_model.predict(X_test_scaled, verbose=0).flatten()
    knn_probs = knn_model.predict_proba(X_test_scaled)[:, 1]
    meta_features = np.column_stack((ann_probs, knn_probs))
    meta_model = LogisticRegressionCV(cv=5, max_iter=1000)
    meta_model.fit(meta_features, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ann_model.save(ANN_PATH, include_optimizer=False)
    joblib.dump(knn_model, KNN_PATH)
    joblib.dump(meta_model, META_PATH)
    joblib.dump(scaler, SCALER_PATH)


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


def predict_risk(ann_model, knn_model, meta_model, scaler, features: np.ndarray) -> int:
    scaled = scaler.transform(features)
    ann_prob = ann_model.predict(scaled, verbose=0).flatten()
    knn_prob = knn_model.predict_proba(scaled)[:, 1]
    meta_features = np.column_stack((ann_prob, knn_prob))
    return int(meta_model.predict(meta_features)[0])


def main() -> None:
    st.set_page_config(
        page_title="Heart Disease Risk Chatbot (SALKS)",
        page_icon="🫀",
        layout="centered",
    )

    st.title("AI-Driven Heart Disease Risk Assessment")
    st.write(
        "This Streamlit UI uses the SALKS ensemble (ANN + KNN + Logistic Regression with SMOTE). "
        "It is for educational use only and not a medical diagnosis."
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

    ann_model, knn_model, meta_model, scaler = get_models()
    features = np.array([[age, sex, trestbps, chol, cp, thalach, fbs, restecg, exang]])

    bp_category = categorize_blood_pressure(int(trestbps))
    chol_category = categorize_cholesterol(int(chol))

    manual_high_risk = trestbps > 120 or chol > 240 or cp >= 2 or exang == 1
    if manual_high_risk:
        risk_label = "High"
    else:
        prediction = predict_risk(ann_model, knn_model, meta_model, scaler, features)
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
