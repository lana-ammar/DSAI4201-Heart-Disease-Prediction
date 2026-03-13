import json
from pathlib import Path
import time
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="HeartGuard AI - Advanced Cardiac Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning visuals and animations
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        text-align: center;
        border-left: 5px solid #667eea;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Risk indicators */
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5253 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .low-risk {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
    }
    .moderate-risk {
        background: linear-gradient(135deg, #fcc419 0%, #f59f00 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Animated progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        transition: width 1s ease-in-out;
    }
    
    /* Custom button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Select boxes */
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    
    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Heart animation */
    .heart-beat {
        animation: heartbeat 1.5s ease-in-out infinite;
        display: inline-block;
    }
    @keyframes heartbeat {
        0% { transform: scale(1); }
        14% { transform: scale(1.3); }
        28% { transform: scale(1); }
        42% { transform: scale(1.3); }
        70% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Animated header
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">
        <span class="heart-beat">❤️</span> HeartGuard AI
    </h1>
    <p style="font-size: 1.2rem; opacity: 0.9; margin-bottom: 0;">
        Advanced Cardiac Risk Assessment System | Powered by Machine Learning
    </p>
    <p style="font-size: 0.9rem; opacity: 0.8; margin-top: 1rem;">
        DSAI4201 Project — Early detection saves lives, every heartbeat matters
    </p>
</div>
""", unsafe_allow_html=True)

# Load model with animated loading
MODEL_PATH = Path("models/heart_model.pkl")
METRICS_PATH = Path("reports/metrics.json")

@st.cache_resource
def load_model():
    with st.spinner('🔄 Loading AI model...'):
        time.sleep(1)  # Simulate loading for effect
        try:
            model = joblib.load(MODEL_PATH)
            return model, True
        except FileNotFoundError:
            st.error("⚠️ Model not found. Please run `python src/train_model.py` first.")
            return None, False
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
            return None, False

model, model_loaded = load_model()

# Navigation menu
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    selected = option_menu(
        menu_title=None,
        options=["Patient Assessment", "Model Insights", "Health Info", "About"],
        icons=["heart-pulse", "graph-up", "info-circle", "question-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#667eea", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#667eea"},
        }
    )
    
    st.markdown("---")
    
    # Model performance dashboard in sidebar
    st.markdown("## 📊 Model Performance")
    if METRICS_PATH.exists() and model_loaded:
        metrics = json.loads(METRICS_PATH.read_text())
        
        # Create gauge charts for key metrics
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics['accuracy'] * 100,
                title={'text': "Accuracy", 'font': {'size': 14}},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "#667eea"},
                       'steps': [
                           {'range': [0, 50], 'color': "#ffcccc"},
                           {'range': [50, 75], 'color': "#fff2cc"},
                           {'range': [75, 100], 'color': "#ccffcc"}]},
                number={'suffix': "%", 'font': {'size': 24}}))
            fig.update_layout(height=150, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics['roc_auc'] * 100,
                title={'text': "ROC-AUC", 'font': {'size': 14}},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "#764ba2"}},
                number={'suffix': "%", 'font': {'size': 24}}))
            fig.update_layout(height=150, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics in expandable section
        with st.expander("📈 Detailed Metrics", expanded=False):
            st.metric("Precision", f"{metrics['precision']:.1%}", 
                     delta="0.5%", delta_color="normal")
            st.metric("Recall", f"{metrics['recall']:.1%}")
            st.metric("F1 Score", f"{metrics['f1']:.1%}")
            st.metric("CV ROC-AUC", f"{metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem; padding: 1rem;'>
        ⚕️ For clinical decision support only.<br>
        Always consult with healthcare professionals.
    </div>
    """, unsafe_allow_html=True)

# Main content based on navigation
if selected == "Patient Assessment":
    # COMBINED PATIENT INFORMATION AND RISK FACTORS ON ONE PAGE (NO TABS)
    
    st.markdown("### Enter Patient Clinical Data")
    st.markdown("<div class='info-box'>ℹ️ Fill in all values accurately for best prediction</div>", unsafe_allow_html=True)
    
    # Patient Information Section
    st.markdown("**Patient Information**")
    
    # Create three columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Basic Information**")
        age = st.slider("Age (years)", 20, 100, 50, 
                       help="Patient's age in years")
        
        sex = st.radio("Sex", options=[0, 1], 
                      format_func=lambda x: "♀️ Female" if x == 0 else "♂️ Male",
                      horizontal=True)
        
        cp = st.select_slider(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            value=0,
            format_func=lambda x: ["🟢 Typical Angina", "🟡 Atypical Angina", 
                                   "🟠 Non-anginal Pain", "🔴 Asymptomatic"][x]
        )
    
    with col2:
        st.markdown("**Vital Signs**")
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120,
                            help="Resting blood pressure in mm Hg")
        
        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200,
                       help="Serum cholesterol in mg/dl")
        
        thalach = st.slider("Max Heart Rate Achieved", 70, 220, 150,
                           help="Maximum heart rate achieved")
    
    with col3:
        st.markdown("**Clinical Measurements**")
        oldpeak = st.slider("ST Depression (mm)", 0.0, 6.0, 1.0, 0.1,
                           help="ST depression induced by exercise relative to rest")
        
        ca = st.selectbox("Major Vessels (0-3)", options=[0, 1, 2, 3],
                        help="Number of major vessels colored by fluoroscopy")
        
        thal = st.select_slider(
            "Thalassemia",
            options=[1, 2, 3],
            format_func=lambda x: ["🟢 Normal", "🟡 Fixed Defect", "🔴 Reversible Defect"][x-1]
        )
    
    st.markdown("---")
    
    # Additional Risk Factors Section
    st.markdown("**⚠️ Additional Risk Factors**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fbs = st.toggle("Fasting Blood Sugar > 120 mg/dl", value=False,
                       help="Toggle if fasting blood sugar is > 120 mg/dl")
        fbs = 1 if fbs else 0
    
    with col2:
        exang = st.toggle("Exercise Induced Angina", value=False,
                         help="Toggle if patient experiences angina during exercise")
        exang = 1 if exang else 0
    
    with col3:
        restecg = st.radio(
            "Resting ECG Results",
            options=[0, 1, 2],
            format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x],
            horizontal=True
        )
        
    slope = st.select_slider(
        "ST Segment Slope",
        options=[0, 1, 2],
        value=1,
        format_func=lambda x: ["📈 Upsloping", "➡️ Flat", "📉 Downsloping"][x],
        help="Slope of the peak exercise ST segment"
    )
    
    # History Section (kept at bottom)
    st.markdown("---")
    st.markdown("### Previous Assessments")
    st.info("No previous assessments found. This will show historical predictions.")
    
    # Placeholder for historical data visualization
    if st.button("📊 View Sample History"):
        # Generate sample historical data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=10, freq='M')
        risks = np.random.uniform(0.1, 0.9, 10)
        
        fig = px.line(x=dates, y=risks, 
                     title="Risk Score History",
                     labels={'x': 'Date', 'y': 'Risk Score'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Prediction button with animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 ANALYZE HEART HEALTH", type="primary", 
                                   disabled=not model_loaded, use_container_width=True)
    
    if predict_button and model_loaded:
        # Create input dataframe
        input_data = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal]],
            columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                     "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        )
        
        # Animated prediction
        with st.spinner('🧠 AI is analyzing patient data...'):
            time.sleep(1.5)
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
        
        # Display results with stunning visuals
        st.markdown("## 🎯 Analysis Results")
        
        # Risk level classification
        if probability < 0.3:
            risk_class = "low-risk"
            risk_text = "LOW RISK"
            emoji = "🟢"
        elif probability < 0.6:
            risk_class = "moderate-risk"
            risk_text = "MODERATE RISK"
            emoji = "🟡"
        else:
            risk_class = "high-risk"
            risk_text = "HIGH RISK"
            emoji = "🔴"
        
        # Main result card
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="{risk_class}">
                <h2 style="font-size: 3rem; margin-bottom: 0.5rem;">{emoji} {risk_text}</h2>
                <p style="font-size: 2rem; margin: 0;">{probability*100:.1f}% Probability</p>
                <p style="font-size: 1rem; opacity: 0.9; margin-top: 1rem;">
                    Based on AI analysis of clinical parameters
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create a donut chart for probability
            fig = go.Figure(data=[go.Pie(
                labels=['Risk', 'No Risk'],
                values=[probability, 1-probability],
                hole=.6,
                marker_colors=['#ff6b6b' if probability > 0.5 else '#fcc419', '#51cf66'],
                textinfo='none'
            )])
            fig.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                height=200,
                annotations=[dict(text=f'{probability*100:.0f}%', x=0.5, y=0.5, 
                                font_size=20, showarrow=False)]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk gauge with animation
        st.markdown("### 📊 Risk Assessment Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk Score", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue", 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#51cf66'},
                    {'range': [30, 60], 'color': '#fcc419'},
                    {'range': [60, 100], 'color': '#ff6b6b'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=50, r=50, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance visualization
        st.markdown("### 🔍 Top Factors Influencing This Prediction")
        
        importance = pd.Series(
            model.feature_importances_, 
            index=input_data.columns
        ).sort_values(ascending=False)
        
        # Create horizontal bar chart with Plotly for interactivity
        fig = make_subplots(rows=1, cols=2, 
                           column_widths=[0.7, 0.3],
                           specs=[[{"type": "bar"}, {"type": "table"}]])
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=importance.values[:8],
                y=importance.index[:8],
                orientation='h',
                marker=dict(
                    color=importance.values[:8],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[f'{v:.3f}' for v in importance.values[:8]],
                textposition='auto',
            ),
            row=1, col=1
        )
        
        # Table with patient values
        patient_values = input_data[importance.index[:8]].T.values.flatten()
        fig.add_trace(
            go.Table(
                header=dict(values=['Feature', 'Value', 'Importance'],
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[
                    importance.index[:8],
                    [f'{input_data[f].values[0]:.1f}' for f in importance.index[:8]],
                    [f'{v:.3f}' for v in importance.values[:8]]
                ], align='left')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            title_text="Feature Importance Analysis",
            showlegend=False,
            bargap=0.2
        )
        
        fig.update_xaxes(title_text="Importance Score", row=1, col=1)
        fig.update_yaxes(title_text="Features", row=1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations based on risk level
        st.markdown("### 💡 Clinical Recommendations")
        
        if probability < 0.3:
            st.success("""
            **Low Risk Patient**
            - Continue healthy lifestyle habits
            - Regular check-ups every 2 years
            - Maintain balanced diet and exercise
            """)
        elif probability < 0.6:
            st.warning("""
            **Moderate Risk Patient - Preventive Measures Recommended**
            - Schedule follow-up with cardiologist
            - Consider stress test and lipid panel
            - Lifestyle modifications: diet, exercise, stress management
            - Monitor blood pressure regularly
            """)
        else:
            st.error("""
            **High Risk Patient - Urgent Care Recommended**
            - Immediate cardiology consultation
            - Comprehensive cardiac workup needed
            - Consider medications as prescribed
            - Lifestyle intervention program
            - Regular monitoring required
            """)
        

elif selected == "Model Insights":
    st.markdown("## 🤖 Model Insights & Performance")
    
    if METRICS_PATH.exists() and model_loaded:
        metrics = json.loads(METRICS_PATH.read_text())
        
        # Create comprehensive model dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metric-card'><h3>Accuracy</h3><h2>{:.1%}</h2></div>".format(metrics['accuracy']), unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'><h3>Precision</h3><h2>{:.1%}</h2></div>".format(metrics['precision']), unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='metric-card'><h3>Recall</h3><h2>{:.1%}</h2></div>".format(metrics['recall']), unsafe_allow_html=True)
        with col4:
            st.markdown("<div class='metric-card'><h3>F1 Score</h3><h2>{:.1%}</h2></div>".format(metrics['f1']), unsafe_allow_html=True)
        
        # Feature importance overview
        st.markdown("### 🌟 Global Feature Importance")
        if hasattr(model, 'feature_importances_'):
            feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                           "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        orientation='h', title="Global Feature Importance",
                        color='Importance', color_continuous_scale='Viridis')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model metrics not available. Please train the model first.")

elif selected == "Health Info":
    st.markdown("## ❤️ Understanding Heart Disease")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Risk Factors
        - **Age**: Risk increases with age
        - **Gender**: Men at higher risk, women's risk increases after menopause
        - **Family History**: Genetic predisposition
        - **Smoking**: Major risk factor
        - **High Blood Pressure**: Damages arteries
        - **High Cholesterol**: Leads to plaque buildup
        - **Diabetes**: Increases risk significantly
        - **Obesity**: Linked to other risk factors
        - **Physical Inactivity**: Weakens heart
        - **Stress**: Can contribute to heart disease
        """)
        
        st.markdown("""
        ### Warning Signs
        - Chest pain or discomfort
        - Shortness of breath
        - Pain in neck, jaw, or back
        - Nausea or lightheadedness
        - Cold sweats
        """)
    
    with col2:
        st.markdown("""
        ### Prevention Tips
        1. **Healthy Diet**: Low in saturated fats, rich in fruits/vegetables
        2. **Regular Exercise**: 150 minutes moderate activity weekly
        3. **No Smoking**: Quit smoking immediately
        4. **Limit Alcohol**: Moderation is key
        5. **Stress Management**: Meditation, yoga, adequate sleep
        6. **Regular Check-ups**: Monitor BP, cholesterol, glucose
        """)
        
        # Interactive BMI calculator
        st.markdown("### 📏 BMI Calculator")
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        
        if height > 0 and weight > 0:
            bmi = weight / ((height/100) ** 2)
            st.metric("Your BMI", f"{bmi:.1f}")
            
            if bmi < 18.5:
                st.info("Underweight")
            elif bmi < 25:
                st.success("Normal weight")
            elif bmi < 30:
                st.warning("Overweight")
            else:
                st.error("Obese")

else:  # About
    st.markdown("## ℹ️ About HeartGuard AI")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Advanced Cardiac Risk Assessment System
        
        HeartGuard AI is a state-of-the-art machine learning application designed to assist healthcare 
        professionals in assessing patient risk for heart disease. Using a Random Forest classifier 
        trained on comprehensive clinical data, our system provides accurate risk predictions based on 
        key patient parameters.
        
        ### Features
        - **Real-time Risk Assessment**: Instant analysis of patient data
        - **Interactive Visualizations**: Clear presentation of risk factors
        - **Feature Importance Analysis**: Understand which factors most influence predictions
        - **Clinical Recommendations**: Evidence-based guidelines based on risk level
        - **Model Performance Metrics**: Transparent display of model accuracy
        
        ### Technology Stack
        - **Frontend**: Streamlit with custom CSS animations
        - **ML Model**: Random Forest Classifier (scikit-learn)
        - **Visualizations**: Plotly, Matplotlib, Seaborn
        - **Data Processing**: Pandas, NumPy
        """)
    
    with col2:
        st.markdown("""
        ### Version
        **HeartGuard AI v2.0**
        
        ### Developed for
        DSAI4201 Project
        
        ### Data Source
        Heart Disease Dataset (Kaggle)
        
        ### Contact
        For questions or support:
        - 📧 support@heartguard.ai
        - 🌐 www.heartguard.ai
        """)
        
        st.markdown("---")
        st.markdown("""
        ### Disclaimer
        This tool is for educational and research purposes only. 
        It is not a substitute for professional medical advice, 
        diagnosis, or treatment.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>❤️ HeartGuard AI — Empowering early detection, saving lives through technology</p>
    <p style='font-size: 0.8rem;'>© 2024 HeartGuard AI. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)