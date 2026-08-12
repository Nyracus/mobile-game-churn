"""Train baseline churn models and write evaluation and interpretation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SEED = 437
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models"


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [PROCESSED_DIR / f"{split}_features.csv" for split in ("train", "dev", "test")]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed feature files: {missing}. Run build_features.py first.")
    return tuple(pd.read_csv(path) for path in paths)  # type: ignore[return-value]


def model_registry(n_features: int) -> dict[str, Pipeline]:
    scaled = ColumnTransformer(
        [("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(range(n_features)))],
        remainder="drop",
    )
    imputed = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), list(range(n_features)))],
        remainder="drop",
    )

    return {
        "logistic_regression": Pipeline(
            [("preprocess", scaled), ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=SEED))]
        ),
        "random_forest": Pipeline(
            [("preprocess", imputed), ("model", RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            ))]
        ),
        "hist_gradient_boosting": Pipeline(
            [("preprocess", imputed), ("model", HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=300,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=SEED,
            ))]
        ),
        "xgboost": Pipeline(
            [("preprocess", imputed), ("model", XGBClassifier(
                n_estimators=500,
                learning_rate=0.04,
                max_depth=5,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=SEED,
            ))]
        ),
    }


def metrics_at_threshold(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
    }


def best_f1_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    candidates = np.linspace(0.10, 0.90, 161)
    scores = [f1_score(y_true, probability >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def save_class_distribution(train: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=train, x="label", hue="label", palette="Set2", legend=False)
    ax.set(title="Training-set class distribution", xlabel="Churn label (0=retained, 1=churned)", ylabel="Players")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=180)
    plt.close()


def save_model_comparison(metrics: pd.DataFrame) -> None:
    plot_data = metrics.melt(id_vars="model", value_vars=["roc_auc", "pr_auc", "f1", "recall"], var_name="metric", value_name="score")
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=plot_data, x="model", y="score", hue="metric")
    ax.set(title="Development-set model comparison", xlabel="Model", ylabel="Score", ylim=(0, 1))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison.png", dpi=180)
    plt.close()


def extract_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)


def save_importance_plot(importance: pd.DataFrame) -> None:
    top = importance.head(20).sort_values("importance")
    plt.figure(figsize=(8, 7))
    plt.barh(top["feature"], top["importance"], color="#4c78a8")
    plt.title("Top 20 model feature importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_importance.png", dpi=180)
    plt.close()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, dev, test = load_splits()
    excluded = {"user_id", "split", "label"}
    feature_names = [column for column in train.columns if column not in excluded]
    x_train, y_train = train[feature_names], train["label"]
    x_dev, y_dev = dev[feature_names], dev["label"]
    x_test = test[feature_names]

    results: list[dict[str, float | str]] = []
    fitted: dict[str, Pipeline] = {}
    probabilities: dict[str, np.ndarray] = {}

    for name, model in model_registry(len(feature_names)).items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_dev)[:, 1]
        row: dict[str, float | str] = {"model": name}
        row.update(metrics_at_threshold(y_dev, probability, threshold=0.5))
        results.append(row)
        fitted[name] = model
        probabilities[name] = probability

    metrics = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    metrics.to_csv(REPORT_DIR / "metrics.csv", index=False)
    save_class_distribution(train)
    save_model_comparison(metrics)

    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted[best_name]
    best_probability = probabilities[best_name]
    threshold = best_f1_threshold(y_dev, best_probability)
    tuned_metrics = metrics_at_threshold(y_dev, best_probability, threshold)

    prediction = (best_probability >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_dev, prediction, cmap="Blues", colorbar=False)
    plt.title(f"{best_name}: development confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    RocCurveDisplay.from_predictions(y_dev, best_probability)
    plt.title(f"{best_name}: development ROC curve")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curve.png", dpi=180)
    plt.close()

    importance = extract_importance(best_model, feature_names)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    if not importance.empty:
        save_importance_plot(importance)

    test_probability = best_model.predict_proba(x_test)[:, 1]
    pd.DataFrame({
        "user_id": test["user_id"],
        "churn_probability": test_probability,
        "predicted_label": (test_probability >= threshold).astype(int),
    }).to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    joblib.dump(best_model, MODEL_DIR / "best_churn_model.joblib")
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps({
        "model": best_name,
        "features": feature_names,
        "decision_threshold": threshold,
        "development_metrics_at_tuned_threshold": tuned_metrics,
        "random_seed": SEED,
    }, indent=2), encoding="utf-8")

    top_features = importance.head(10)["feature"].tolist()
    summary = f"""# Initial model results

- Best model by development ROC-AUC: **{best_name}**
- Tuned decision threshold: **{threshold:.3f}**
- ROC-AUC: **{tuned_metrics['roc_auc']:.4f}**
- PR-AUC: **{tuned_metrics['pr_auc']:.4f}**
- Precision: **{tuned_metrics['precision']:.4f}**
- Recall: **{tuned_metrics['recall']:.4f}**
- F1-score: **{tuned_metrics['f1']:.4f}**

Top model features: {', '.join(top_features)}.

These are baseline development-set results. The next research stage should add cross-validation, formal hyperparameter optimization, SHAP explanations, and error analysis before making final claims.
"""
    (REPORT_DIR / "project_summary.md").write_text(summary, encoding="utf-8")

    print("\nDevelopment metrics:")
    print(metrics.to_string(index=False))
    print(f"\nSelected {best_name} with tuned threshold {threshold:.3f}.")


if __name__ == "__main__":
    main()

