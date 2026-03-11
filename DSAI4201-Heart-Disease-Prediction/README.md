# AI-Powered Heart Disease Prediction System
**DSAI4201 Course Project**  
Group: Ragad Ziyada , Rasha Fadulallah, Lana Mukhtar 

## Problem
Heart disease is the #1 cause of death globally (WHO). Early prediction can save lives.

## Solution
Random Forest classifier (92% accuracy) deployed as an interactive Streamlit web app.  
Doctors/patients enter 13 medical parameters, instant prediction and explanation.

## How to Run
```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py