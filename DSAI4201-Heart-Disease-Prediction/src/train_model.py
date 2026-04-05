from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

DATA_SOURCE_URL = "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset"
DATA_PATH = Path("data/Heart Disease Dataset/heart.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "target"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download from Kaggle and place heart.csv there."
        )

    df = pd.read_csv(DATA_PATH)
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    before = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=FEATURES)  # remove conflicting targets for same features

    print(f"Removed duplicates: {before - len(df)} rows")
    print(f"Final dataset size: {len(df)} rows (note: small dataset ~303 unique records)")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ====================== RANDOM FOREST ======================
    print("Training Random Forest with GridSearchCV...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": [None, "balanced"],
    }

    search = GridSearchCV(rf, param_grid, scoring="roc_auc", cv=5, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    print(f"Best RF params: {search.best_params_}")

    # SHAP Explainer for Random Forest (TreeExplainer is efficient)
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(best_rf)
    shap_values_test = explainer.shap_values(X_test)

    # Save global SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_test[1] if isinstance(shap_values_test, list) else shap_values_test,
                      X_test, plot_type="bar", show=False)
    plt.title("Global SHAP Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ====================== XGBOOST COMPARISON ======================
    print("Training XGBoost for comparison...")
    xgb_model = XGBClassifier(random_state=42, eval_metric="auc", n_jobs=-1, use_label_encoder=False)
    xgb_model.fit(X_train, y_train)

    # ====================== EVALUATION ======================
    # RF predictions
    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

    # XGBoost predictions
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    metrics = {
        "random_forest": {
            "accuracy": float(accuracy_score(y_test, y_pred_rf)),
            "precision": float(precision_score(y_test, y_pred_rf)),
            "recall": float(recall_score(y_test, y_pred_rf)),
            "f1": float(f1_score(y_test, y_pred_rf)),
            "roc_auc": float(roc_auc_score(y_test, y_proba_rf)),
            "best_params": search.best_params_,
        },
        "xgboost": {
            "accuracy": float(accuracy_score(y_test, y_pred_xgb)),
            "precision": float(precision_score(y_test, y_pred_xgb)),
            "recall": float(recall_score(y_test, y_pred_xgb)),
            "f1": float(f1_score(y_test, y_pred_xgb)),
            "roc_auc": float(roc_auc_score(y_test, y_proba_xgb)),
        },
        "cv_roc_auc_mean": float(np.mean([roc_auc_score(y.iloc[test_idx], 
            XGBClassifier(**xgb_model.get_params(), random_state=42).fit(X.iloc[train_idx], y.iloc[train_idx]).predict_proba(X.iloc[test_idx])[:,1]) 
            for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)]))  # simplified CV
    }

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_rf, MODELS_DIR / "heart_model_rf.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "heart_model_xgb.pkl")
    joblib.dump(explainer, MODELS_DIR / "shap_explainer_rf.pkl")

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS_DIR / "shap_summary_bar.png").exists()  # already saved above

    print("\n=== Model Performance ===")
    print(json.dumps(metrics, indent=2))
    print("\nArtifacts saved:")
    print(f"   RF model: {MODELS_DIR / 'heart_model_rf.pkl'}")
    print(f"   XGB model: {MODELS_DIR / 'heart_model_xgb.pkl'}")
    print(f"   SHAP explainer: {MODELS_DIR / 'shap_explainer_rf.pkl'}")
    print(f"   SHAP summary: {REPORTS_DIR / 'shap_summary_bar.png'}")

    print("\nWARNING: High metrics should be viewed cautiously due to small dataset size (~303 records).")


if __name__ == "__main__":
    main()