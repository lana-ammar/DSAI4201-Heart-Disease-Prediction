# AI-Powered Heart Disease Prediction System
**DSAI4201 Course Project**  
Group: 
Ragad Ziyada - 60301042 , Rasha Fadulallah - 60301813, Lana Mukhtar - 60107216


##  Problem

Heart disease is the **#1 cause of death globally**, responsible for ~17.9 million deaths per year (WHO). Early detection dramatically improves patient outcomes, but traditional screening requires specialist expertise and is not always accessible.

##  Solution

A **Random Forest classifier** trained on 13 clinical measurements that predicts heart disease risk with a cross-validated ROC-AUC. Deployed as an interactive Streamlit web app, doctors or patients enter values and receive an instant risk assessment with explanation.

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 77.0% |
| Precision | 78.8% |
| Recall | 78.8% |
| F1 Score | 78.8% |
| ROC-AUC (test) | 0.863 |
| ROC-AUC (CV mean ± std) | 0.916 ± 0.013 |

---

## Repository Structure

```
├── data/
│   └── Heart Disease Dataset/
│       └── heart.csv           # Raw dataset (download from Kaggle)
├── src/
│   └── train_model.py          # Training pipeline
├── models/
│   ├── heart_model.pkl         # Trained model (generated after training)
│   └── features.json           # Feature list
├── reports/
│   ├── metrics.json            # Evaluation metrics
│   ├── feature_importance.csv  # Permutation importance scores
│   └── data_source.txt         # Data provenance
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone <https://github.com/lana-ammar/DSAI4201-Heart-Disease-Prediction>
cd DSAI4201-HEART-DISEASE-PREDICTION
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `heart.csv` from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) and place it at:
```
data/Heart Disease Dataset/heart.csv
```

### 4. Train the model
```bash
python src/train_model.py
```
This will save the trained model to `models/` and evaluation reports to `reports/`.

### 5. Launch the app
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## Input Features

| Feature | Description |
|---------|-------------|
| age | Patient age (years) |
| sex | Biological sex (0 = Female, 1 = Male) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0–2) |
| thalach | Max heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression (exercise vs rest) |
| slope | Slope of peak exercise ST segment |
| ca | Major vessels coloured by fluoroscopy (0–3) |
| thal | Thalassemia type (1–3) |

---

## How It Works

1. **Data Cleaning** — duplicates removed, column validation performed
2. **Hyperparameter Tuning** — 5-fold stratified GridSearchCV over 48 parameter combinations, optimising ROC-AUC
3. **Evaluation** — test set metrics + separate 5-fold cross-validation for robust generalisation estimate
4. **Explainability** — permutation importance computed on held-out test set to show which features drive each prediction

---

##  Disclaimer

This tool is for **clinical decision support only**. It is not a substitute for professional medical diagnosis. Always consult a qualified doctor.

---

## Data Source

- Dataset: [Heart Disease Dataset — Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Original source: Cleveland Clinic Foundation (UCI Heart Disease Repository)
- Detrano, R., et al. (1989). *American Journal of Cardiology*, 64(5), 304–310.