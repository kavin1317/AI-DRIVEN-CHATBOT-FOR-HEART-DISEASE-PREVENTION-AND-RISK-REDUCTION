from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "heart.csv"
MODEL_DIR = APP_DIR / "models"

ANN_PATH = MODEL_DIR / "salks_ann.h5"
KNN_PATH = MODEL_DIR / "salks_knn.pkl"
META_PATH = MODEL_DIR / "salks_meta.pkl"
SCALER_PATH = MODEL_DIR / "salks_scaler.pkl"

FEATURES = ["age", "sex", "trestbps", "chol", "cp", "thalach", "fbs", "restecg", "exang"]
FAST_EPOCHS = 12
FULL_EPOCHS = 40
FAST_BATCH = 32
FULL_BATCH = 16


def models_exist() -> bool:
    """Check if all model files exist."""
    return all(path.exists() for path in [ANN_PATH, KNN_PATH, META_PATH, SCALER_PATH])


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the heart disease dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    X = df[FEATURES]
    y = df["target"]
    return X, y


def build_ann(input_dim: int, fast_mode: bool) -> tf.keras.Model:
    """Build Artificial Neural Network model."""
    hidden_units = (64, 32) if fast_mode else (128, 64)
    model = Sequential(
        [
            Dense(hidden_units[0], activation="relu", input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(hidden_units[1], activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_save_models(fast_mode: bool, progress_bar=None, status_text=None) -> None:
    """Train all models and save them to disk."""
    if status_text:
        status_text.text("📊 Loading dataset...")
    X, y = load_data()

    if status_text:
        status_text.text("🔀 Splitting data and balancing classes...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    if progress_bar:
        progress_bar.progress(0.2)

    if status_text:
        status_text.text("📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    if progress_bar:
        progress_bar.progress(0.3)

    if status_text:
        status_text.text("🧠 Training Neural Network (ANN)...")
    ann_model = build_ann(X_train_scaled.shape[1], fast_mode=fast_mode)
    epochs = FAST_EPOCHS if fast_mode else FULL_EPOCHS
    batch_size = FAST_BATCH if fast_mode else FULL_BATCH
    callbacks = [EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)]
    ann_model.fit(
        X_train_scaled,
        y_train_resampled,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    
    if progress_bar:
        progress_bar.progress(0.6)

    if status_text:
        status_text.text("🎯 Training K-Nearest Neighbors (KNN)...")
    knn_model = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="manhattan")
    knn_model.fit(X_train_scaled, y_train_resampled)
    
    if progress_bar:
        progress_bar.progress(0.8)

    if status_text:
        status_text.text("🔗 Training Meta-Model (Logistic Regression)...")
    ann_probs = ann_model.predict(X_test_scaled, verbose=0).flatten()
    knn_probs = knn_model.predict_proba(X_test_scaled)[:, 1]
    meta_features = np.column_stack((ann_probs, knn_probs))
    meta_model = LogisticRegression(max_iter=300)
    meta_model.fit(meta_features, y_test)
    
    if progress_bar:
        progress_bar.progress(0.9)

    if status_text:
        status_text.text("💾 Saving models...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ann_model.save(ANN_PATH, include_optimizer=False)
    joblib.dump(knn_model, KNN_PATH)
    joblib.dump(meta_model, META_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    if progress_bar:
        progress_bar.progress(1.0)
    
    if status_text:
        status_text.text("✅ Training complete!")


@st.cache_resource(show_spinner=False)
def get_models(fast_mode: bool):
    """Load or train models. Cached to avoid retraining on every run."""
    if not models_exist():
        # Models don't exist, need to train
        return None  # Signal that training is needed
    
    # Models exist, load them
    ann_model = tf.keras.models.load_model(ANN_PATH)
    knn_model = joblib.load(KNN_PATH)
    meta_model = joblib.load(META_PATH)
    scaler = joblib.load(SCALER_PATH)
    return ann_model, knn_model, meta_model, scaler


def categorize_blood_pressure(bp: int) -> str:
    """Categorize blood pressure level."""
    if bp < 120:
        return "Normal"
    elif bp < 140:
        return "Prehypertension"
    else:
        return "High"


def categorize_cholesterol(chol: int) -> str:
    """Categorize cholesterol level."""
    if chol < 200:
        return "Normal"
    elif chol < 240:
        return "Borderline High"
    else:
        return "High"


def predict_risk(ann_model, knn_model, meta_model, scaler, features: np.ndarray) -> tuple[int, float]:
    """Predict heart disease risk using ensemble model."""
    scaled = scaler.transform(features)
    ann_prob = ann_model.predict(scaled, verbose=0).flatten()[0]
    knn_prob = knn_model.predict_proba(scaled)[0, 1]
    meta_features = np.column_stack(([ann_prob], [knn_prob]))
    prediction = int(meta_model.predict(meta_features)[0])
    confidence = float(meta_model.predict_proba(meta_features)[0, prediction])
    return prediction, confidence


def main() -> None:
    st.set_page_config(
        page_title="SALKS Chatbot - Heart Disease Risk",
        page_icon="🫀",
        layout="centered",
    )

    # Ensure model directory exists for saved artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    st.title("🫀 AI-Driven Heart Disease Chatbot (SALKS)")
    st.write(
        "This chatbot uses the **SALKS ensemble** (ANN + KNN + Logistic Regression with SMOTE) "
        "to estimate heart disease risk and provide lifestyle tips."
    )
    st.info("**For educational purposes only - not a medical diagnosis.**")

    with st.sidebar:
        st.header("⚙️ Configuration")
        fast_mode = st.checkbox("⚡ Fast training mode (recommended)", value=True)
        st.caption("Fast mode: 12 epochs, smaller network. Full mode: 40 epochs, larger network.")
        st.caption("Models are saved locally in `models/` and reused on next run.")
        retrain = st.button("🔁 Train/Update Models")
        
        st.divider()
        st.header("📝 Input Details")
        
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
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL", 
            options=[("No", 0), ("Yes", 1)], 
            format_func=lambda x: x[0]
        )[1]
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
            "Exercise-Induced Angina", 
            options=[("No", 0), ("Yes", 1)], 
            format_func=lambda x: x[0]
        )[1]
        
        st.divider()
        run = st.button("🔍 Assess Risk", type="primary", use_container_width=True)

    if retrain:
        st.warning("⚠️ Training models (this happens once and will be reused).")
        progress_bar = st.progress(0)
        status_text = st.empty()
        train_and_save_models(fast_mode, progress_bar, status_text)
        st.success("✅ Models trained and saved. Re-run assessment.")
        get_models.clear()
        return

    if not run:
        st.info("👈 Fill in your health details in the sidebar and click '🔍 Assess Risk' to begin.")
        return

    # Check if models need to be trained
    models = get_models(fast_mode)
    
    if models is None:
        # Need to train models
        st.warning("⚠️ Models not found. Training models for the first time...")
        st.info("This will take 30-60 seconds depending on your system. Models will be saved for future use.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            train_and_save_models(fast_mode, progress_bar, status_text)
            st.success("✅ Models trained successfully! Loading models...")
            # Clear cache and reload
            get_models.clear()
            models = get_models(fast_mode)
        except Exception as e:
            st.error(f"❌ Error training models: {str(e)}")
            st.stop()
    
    ann_model, knn_model, meta_model, scaler = models
    
    # Prepare features
    features = np.array([[age, sex, trestbps, chol, cp, thalach, fbs, restecg, exang]])

    # Get prediction
    with st.spinner("🤔 Analyzing your health data..."):
        prediction, confidence = predict_risk(ann_model, knn_model, meta_model, scaler, features)
    
    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    bp_category = categorize_blood_pressure(int(trestbps))
    chol_category = categorize_cholesterol(int(chol))

    # Display chatbot-style conversation
    st.subheader("📋 Your Health Information")
    chat_pairs = [
        ("Age", f"{age} years"),
        ("Sex", "Male" if sex == 1 else "Female"),
        ("Resting Blood Pressure", f"{trestbps} mmHg"),
        ("Cholesterol Level", f"{chol} mg/dL"),
        ("Chest Pain Type", ["None", "Mild", "Moderate", "Severe"][cp]),
        ("Max Heart Rate Achieved", f"{thalach} bpm"),
        ("Fasting Blood Sugar > 120 mg/dL", "Yes" if fbs == 1 else "No"),
        ("Resting ECG", ["Normal", "ST-T wave abnormality", "Possible LVH"][restecg]),
        ("Exercise-Induced Angina", "Yes" if exang == 1 else "No"),
    ]
    
    for question, answer in chat_pairs:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

    # Health Analysis
    st.subheader("🔍 Health Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Blood Pressure", bp_category)
    with col2:
        st.metric("Cholesterol", chol_category)

    # Risk Prediction
    st.subheader("⚠️ Risk Prediction")
    if prediction == 1:
        st.error(f"**{risk_label}** - Model Confidence: {confidence:.1%}")
    else:
        st.success(f"**{risk_label}** - Model Confidence: {confidence:.1%}")

    # Recommendations
    st.subheader("💡 Recommended Health Tips")
    if prediction == 1:
        tips = [
            "🏥 **Consult a doctor as soon as possible** for comprehensive evaluation.",
            "🥗 Follow a diet **low in saturated fats and high in fiber**.",
            "🏃 Exercise at least **30 minutes daily** with moderate intensity.",
            "🧘 Reduce stress through **meditation, yoga, or deep breathing**.",
            "🚭 **Quit smoking** and limit alcohol intake.",
            "📅 Schedule **regular checkups** for blood pressure and cholesterol.",
        ]
    else:
        tips = [
            "🥗 Maintain a **balanced diet** and regular exercise.",
            "📊 Continue **periodic monitoring** of blood pressure and cholesterol.",
            "😊 Stay active to reduce future risk.",
        ]
    
    for tip in tips:
        st.markdown(tip)
    
    # Model info
    with st.expander("ℹ️ About the SALKS Ensemble"):
        st.write("""
        The SALKS model combines three machine learning approaches:
        
        1. **ANN (Artificial Neural Network)**: Deep learning model with batch normalization and dropout
        2. **KNN (K-Nearest Neighbors)**: Distance-weighted classifier
        3. **Meta-Model (Logistic Regression)**: Combines predictions from ANN and KNN
        
        The models are trained on the UCI Heart Disease dataset with SMOTE for class balancing.
        """)


if __name__ == "__main__":
    main()
