from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

DATA_SOURCE_URL = "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset"
DATA_PATH = Path("data/heart.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET = "target"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            "Dataset not found. Download the Kaggle dataset and place the CSV at "
            f"{DATA_PATH.as_posix()} (expected file name: heart.csv)."
        )

    df = pd.read_csv(DATA_PATH)

    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def main() -> None:
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "best_params": search.best_params_,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODELS_DIR / "heart_model.pkl")
    (MODELS_DIR / "features.json").write_text(json.dumps(FEATURES, indent=2))
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS_DIR / "data_source.txt").write_text(
        "Data source:\n"
        f"- {DATA_SOURCE_URL}\n"
        f"- Accessed: {date.today().isoformat()}\n"
        "File used: data/heart.csv\n"
    )

    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=42,
        scoring="roc_auc",
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    print("Model training complete.")
    print(json.dumps(metrics, indent=2))
    print("Saved:")
    print(f"- {MODELS_DIR / 'heart_model.pkl'}")
    print(f"- {MODELS_DIR / 'features.json'}")
    print(f"- {REPORTS_DIR / 'metrics.json'}")
    print(f"- {REPORTS_DIR / 'feature_importance.csv'}")
    print(f"- {REPORTS_DIR / 'data_source.txt'}")


if __name__ == "__main__":
    main()
