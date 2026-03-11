import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("AI-Powered Heart Disease Prediction System")
st.markdown("**DSAI4201 Project** – Early detection saves lives")

# Load model
try:
    model = joblib.load("models/heart_model.pkl")
except:
    st.error("Model not found. Train the model first.")

# Input form
st.sidebar.header("Model Info")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Dataset: Heart Disease Dataset")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 20, 80, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", [0,1,2,3], format_func=lambda x: ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"][x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0,1,2])
    thalach = st.number_input("Max Heart Rate Achieved", 70, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0,1,2])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0,1,2,3])
    thal = st.selectbox("Thalassemia", [1,2,3], format_func=lambda x: ["Normal","Fixed Defect","Reversible Defect"][x-1])

if st.button("🔮 Predict Heart Disease Risk", type="primary"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f" HIGH RISK: {probability*100:.1f}% probability of Heart Disease")
    else:
        st.success(f"LOW RISK: {probability*100:.1f}% probability of Heart Disease")
    
    # Feature importance plot
    importance = pd.Series(model.feature_importances_, index=input_data.columns).sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=importance.values, y=importance.index, ax=ax)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Top Factors Influencing Prediction")
    st.pyplot(fig)
    
    st.info("This is a decision-support tool. Always consult a doctor.")