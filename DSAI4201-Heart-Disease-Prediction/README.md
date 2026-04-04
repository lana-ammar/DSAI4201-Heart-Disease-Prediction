# AI-Powered Heart Disease Prediction System
**DSAI4201 Course Project**

Group:
- Ragad Ziyada - 60301042
- Rasha Fadulallah - 60301813
- Lana Mukhtar - 60107216

## Problem
Heart disease is a leading global cause of death. Early risk screening can improve outcomes, but access to specialist evaluation is not always immediate.

## Solution
This project provides an end-to-end ML pipeline and Streamlit app for heart disease risk prediction from 13 clinical features.

This upgraded version adds:
- Model benchmarking (Random Forest vs Logistic Regression vs Gradient Boosting).
- Local interpretability with LIME for each patient prediction.
- Global interpretability with permutation importance and aggregated LIME importance.
- Better overfitting visibility via CV-test ROC-AUC gap tracking.

## Performance Reporting
The pipeline reports:
- Accuracy, Precision, Recall, F1
- ROC-AUC and PR-AUC
- Brier score
- Cross-validated ROC-AUC mean and std
- Multi-model comparison table in `reports/model_comparison.csv`

## Repository Structure
```text
├── data/
│   └── Heart Disease Dataset/
│       └── heart.csv
├── src/
│   └── train_model.py
├── models/
│   ├── heart_model.pkl
│   └── features.json
├── reports/
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── lime_global_importance.csv
│   └── data_source.txt
├── app.py
├── requirements.txt
└── README.md
```

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place dataset file at:
```text
data/Heart Disease Dataset/heart.csv
```

3. Train and generate artifacts:
```bash
python src/train_model.py
```

4. Launch the app:
```bash
streamlit run app.py
```

## What Was Added for Depth
1. Comparison extension:
- Instead of only one tuned model, the project now compares three model families with the same evaluation protocol.
- Results are saved and viewable in the app under Model Insights.

2. Interpretability extension:
- Local LIME explanation appears for each patient prediction.
- Global LIME summary is generated to complement permutation importance.

These additions directly address the feedback about technical depth and novelty.

## Disclaimer
This tool is for educational and decision-support purposes only. It is not a substitute for professional clinical diagnosis.

## Data Source
- Kaggle Heart Disease Dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- Original source: Cleveland Clinic Foundation (UCI Heart Disease Repository)
