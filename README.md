# AI-Driven Chatbot for Heart Disease Prevention and Risk Reduction

## Overview
Cardiovascular diseases (CVDs) remain a leading cause of mortality worldwide. This project proposes an AI-powered chatbot that assesses heart disease risk and provides personalized recommendations on exercise, diet, and stress management. The approach combines data analytics and machine learning to make risk assessment more accessible and interactive.

## Key Idea (SALKS)
The proposed model, SALKS, integrates:
- Stratified k-fold cross validation for robust evaluation
- Artificial Neural Networks (ANNs) for non-linear feature learning
- Logistic Regression for explainability and meta-learning
- K-Nearest Neighbors (KNN) for proximity-based classification
- SMOTE to handle class imbalance

This ensemble is designed to improve predictive performance over traditional single models.

## Dataset
The report states the dataset was sourced from public medical repositories, primarily the UCI Heart Disease dataset and Kaggle medical datasets. The data includes clinical and lifestyle attributes such as:
- Age, sex, chest pain type (cp)
- Resting blood pressure (trestbps), serum cholesterol (chol)
- Fasting blood sugar (fbs), resting ECG (restecg)
- Max heart rate (thalach), exercise-induced angina (exang)
- ST depression (oldpeak), slope, ca, thal

The repository includes `data/heart.csv` for experiments.

## Methods
- Data cleaning and duplicate removal
- Normalization and scaling (StandardScaler, MinMaxScaler)
- Statistical tests (chi-square, ANOVA, F-test)
- PCA and feature selection
- SMOTE for class balancing
- Stratified k-fold cross validation

## Results (from the report)
SALKS achieved 87.02% accuracy, outperforming:
- ANN: 81.97%
- XRSS (XGBoost Random Classifier with Stratified K-Fold SMOTE): 81.30%
- KNN: 77.05%

## Notebooks
- `Notebooks/Proj_2_CB.ipynb`
- `Notebooks/Proj_2_SALKS.ipynb`
- `Notebooks/Proj_2_XGB_RC.ipynb`
- `Notebooks/proj_2_ANN.ipynb`
- `Notebooks/proj_2_KNN.ipynb`

## Python Scripts
- `proj_2_ann.py`
- `proj_2_knn.py`
- `proj_2_own.py`
- `proj_2_xgb_rc.py`

## Other Files
- Dataset: `data/heart.csv`
- Report: `reports/Kavin Proj Report Final.pdf`
- Streamlit UI: `streamlit_app.py`

## Streamlit UI
Run the web interface locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## How to Run Notebooks
1. Open a notebook in Jupyter or Google Colab.
2. Make sure `data/heart.csv` is available (adjust paths if needed).
3. Run the cells top to bottom.

## Report Source
This README is derived from the project report: `reports/Kavin Proj Report Final.pdf`.
