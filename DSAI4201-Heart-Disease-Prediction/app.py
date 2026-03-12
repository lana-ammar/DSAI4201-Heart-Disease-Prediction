import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("AI-Powered Heart Disease Prediction System")
st.markdown("**DSAI4201 Project** – Early detection saves lives")

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/heart_model.pkl")
METRICS_PATH = Path("reports/metrics.json")

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    st.error("Model not found. Please run `python src/train_model.py` first.")
    model_loaded = False
except Exception as e:
    st.error(f"Unexpected error loading model: {e}")
    model_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Model Info")
st.sidebar.write("**Algorithm:** Random Forest")
st.sidebar.write("**Dataset:** Heart Disease Dataset (Kaggle)")

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Performance")
    st.sidebar.metric("Accuracy",  f"{metrics['accuracy']:.1%}")
    st.sidebar.metric("ROC-AUC",   f"{metrics['roc_auc']:.3f}")
    st.sidebar.metric("Precision", f"{metrics['precision']:.1%}")
    st.sidebar.metric("Recall",    f"{metrics['recall']:.1%}")
    st.sidebar.metric("F1 Score",  f"{metrics['f1']:.1%}")
    st.sidebar.metric("CV ROC-AUC (mean ± std)",
                      f"{metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")

st.sidebar.markdown("---")
st.sidebar.caption("⚕️ For clinical decision support only. Always consult a doctor.")

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("🩺 Enter Patient Information")
col1, col2, col3 = st.columns(3)

with col1:
    age      = st.number_input("Age", 20, 80, 50)
    sex      = st.selectbox("Sex", [0, 1],
                            format_func=lambda x: "Female" if x == 0 else "Male")
    cp       = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                            format_func=lambda x: ["Typical Angina", "Atypical Angina",
                                                   "Non-anginal Pain", "Asymptomatic"][x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol     = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

with col2:
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                            format_func=lambda x: "No" if x == 0 else "Yes")
    restecg  = st.selectbox("Resting ECG Results", [0, 1, 2],
                            format_func=lambda x: ["Normal", "ST-T Abnormality",
                                                   "Left Ventricular Hypertrophy"][x])
    thalach  = st.number_input("Max Heart Rate Achieved", 70, 220, 150)
    exang    = st.selectbox("Exercise Induced Angina", [0, 1],
                            format_func=lambda x: "No" if x == 0 else "Yes")

with col3:
    oldpeak  = st.number_input("ST Depression (Exercise vs Rest)", 0.0, 6.0, 1.0, 0.1)
    slope    = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2],
                            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca       = st.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal     = st.selectbox("Thalassemia", [1, 2, 3],
                            format_func=lambda x: ["Normal", "Fixed Defect",
                                                   "Reversible Defect"][x - 1])

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Heart Disease Risk", type="primary", disabled=not model_loaded):
    input_data = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg,
          thalach, exang, oldpeak, slope, ca, thal]],
        columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
    )

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"**HIGH RISK** — {probability * 100:.1f}% probability of Heart Disease")
    else:
        st.success(f"**LOW RISK** — {probability * 100:.1f}% probability of Heart Disease")

    # Risk gauge bar
    st.progress(float(probability), text=f"Risk Score: {probability:.1%}")

    # Feature importance chart
    st.subheader("Top Factors Influencing This Prediction")
    importance = (
        pd.Series(model.feature_importances_, index=input_data.columns)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e74c3c" if f in ["ca", "thal", "cp", "thalach"] else "#3498db"
              for f in importance.index]
    sns.barplot(x=importance.values, y=importance.index, palette=colors, ax=ax)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance (Random Forest)")
    st.pyplot(fig)
    plt.close(fig)

    st.info("This tool is for decision support only. Always consult a qualified doctor.")