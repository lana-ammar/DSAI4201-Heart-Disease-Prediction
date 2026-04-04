from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_SOURCE_URL = "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset"
DATA_PATH = Path("data/Heart Disease Dataset/heart.csv")
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
    """Load and clean the heart disease dataset."""
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            "Dataset not found. Download the Kaggle dataset and place the CSV at "
            f"{DATA_PATH.as_posix()} (expected file name: heart.csv)."
        )

    df = pd.read_csv(DATA_PATH)

    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    before_feat = len(df)
    df = df.drop_duplicates(subset=FEATURES)
    removed_feat = before_feat - len(df)

    print(f"Removed duplicate rows: {removed}")
    print(f"Removed duplicate feature rows: {removed_feat}")
    print(f"Final dataset size: {len(df)} rows")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def build_model_spaces() -> dict[str, tuple[object, dict[str, list[object]]]]:
    """Model families and their hyperparameter grids for a fair benchmark."""
    return {
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "class_weight": [None, "balanced"],
            },
        ),
        "LogisticRegression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            random_state=42,
                            max_iter=2000,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.1, 1.0, 3.0, 10.0],
                "model__solver": ["lbfgs", "liblinear"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3],
                "min_samples_leaf": [1, 2],
                "subsample": [0.8, 1.0],
            },
        ),
    }


def evaluate_model(model: object, X_eval: pd.DataFrame, y_eval: pd.Series) -> dict[str, float]:
    """Compute a richer set of classification metrics."""
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
        "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
        "f1": float(f1_score(y_eval, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_eval, y_proba)),
        "pr_auc": float(average_precision_score(y_eval, y_proba)),
        "brier": float(brier_score_loss(y_eval, y_proba)),
    }


def summarize_lime_global_importance(
    model: object,
    X_train: pd.DataFrame,
    X_reference: pd.DataFrame,
    max_rows: int = 40,
) -> pd.DataFrame:
    """Aggregate absolute LIME weights over a subset of rows for global insight."""
    explainer = LimeTabularExplainer(
        training_data=X_train.to_numpy(),
        feature_names=FEATURES,
        class_names=["No Disease", "Disease"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    sample_size = min(max_rows, len(X_reference))
    sampled = X_reference.sample(n=sample_size, random_state=42)

    weight_sums = {feature: 0.0 for feature in FEATURES}

    for _, row in sampled.iterrows():
        explanation = explainer.explain_instance(
            row.to_numpy(),
            model.predict_proba,
            num_features=len(FEATURES),
        )
        for condition, weight in explanation.as_list(label=1):
            matched_feature = next((f for f in FEATURES if f in condition), None)
            if matched_feature is not None:
                weight_sums[matched_feature] += abs(weight)

    lime_df = pd.DataFrame(
        {
            "feature": list(weight_sums.keys()),
            "mean_abs_lime_weight": [value / sample_size for value in weight_sums.values()],
        }
    ).sort_values("mean_abs_lime_weight", ascending=False)

    return lime_df


def main() -> None:
    """Train, benchmark, evaluate, and save the heart disease model artifacts."""
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_spaces = build_model_spaces()

    comparison_rows: list[dict[str, object]] = []
    tuned_models: dict[str, object] = {}

    print("Running multi-model benchmark with GridSearchCV...")
    for model_name, (estimator, param_grid) in model_spaces.items():
        print(f"  -> Tuning {model_name}")
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0,
            return_train_score=False,
        )
        search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        tuned_models[model_name] = best_estimator

        test_metrics = evaluate_model(best_estimator, X_test, y_test)
        comparison_rows.append(
            {
                "model": model_name,
                "cv_roc_auc_mean": float(search.best_score_),
                "cv_roc_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_brier": test_metrics["brier"],
                "cv_test_roc_auc_gap": float(search.best_score_ - test_metrics["roc_auc"]),
                "best_params": json.dumps(search.best_params_),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["test_roc_auc", "cv_roc_auc_mean"],
        ascending=False,
    )

    deployment_model_name = "RandomForest" if "RandomForest" in tuned_models else comparison_df.iloc[0]["model"]
    deployment_model = tuned_models[deployment_model_name]

    y_pred = deployment_model.predict(X_test)
    y_proba = deployment_model.predict_proba(X_test)[:, 1]

    deployment_cv_scores = cross_val_score(
        clone(deployment_model),
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    metrics = {
        "deployment_model": deployment_model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "brier": float(brier_score_loss(y_test, y_proba)),
        "cv_roc_auc_mean": float(np.mean(deployment_cv_scores)),
        "cv_roc_auc_std": float(np.std(deployment_cv_scores)),
        "rows_used": int(len(X)),
        "comparison_rows": int(len(comparison_df)),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(deployment_model, MODELS_DIR / "heart_model.pkl")
    (MODELS_DIR / "features.json").write_text(json.dumps(FEATURES, indent=2))
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS_DIR / "model_comparison.csv").write_text(comparison_df.to_csv(index=False))
    (REPORTS_DIR / "data_source.txt").write_text(
        "Data source:\n"
        f"- {DATA_SOURCE_URL}\n"
        f"- Accessed: {date.today().isoformat()}\n"
        "File used: data/Heart Disease Dataset/heart.csv\n"
    )

    perm = permutation_importance(
        deployment_model,
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

    lime_importance = summarize_lime_global_importance(
        model=deployment_model,
        X_train=X_train,
        X_reference=X_test,
    )
    lime_importance.to_csv(REPORTS_DIR / "lime_global_importance.csv", index=False)

    print("\nModel training complete.")
    print(json.dumps(metrics, indent=2))

    if metrics["accuracy"] == 1.0 and metrics["roc_auc"] == 1.0:
        print(
            "WARNING: Perfect test metrics detected. "
            "This can happen with small or duplicated datasets."
        )

    print("\nSaved:")
    print(f"  - {MODELS_DIR / 'heart_model.pkl'}")
    print(f"  - {MODELS_DIR / 'features.json'}")
    print(f"  - {REPORTS_DIR / 'metrics.json'}")
    print(f"  - {REPORTS_DIR / 'model_comparison.csv'}")
    print(f"  - {REPORTS_DIR / 'feature_importance.csv'}")
    print(f"  - {REPORTS_DIR / 'lime_global_importance.csv'}")
    print(f"  - {REPORTS_DIR / 'data_source.txt'}")


if __name__ == "__main__":
    main()
