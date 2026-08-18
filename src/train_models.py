"""Tune churn models with training-only CV and evaluate once on the development set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SEED = 437
CV_FOLDS = 5
SEARCH_ITERATIONS = 15
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models"
EXCLUDED = {"user_id", "split", "label"}


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [PROCESSED_DIR / f"{split}_features.csv" for split in ("train", "dev", "test")]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed feature files: {missing}. Run build_features.py first.")
    return tuple(pd.read_csv(path) for path in paths)  # type: ignore[return-value]


def _preprocessors(feature_names: list[str]) -> tuple[ColumnTransformer, ColumnTransformer]:
    scaled = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                feature_names,
            )
        ],
        remainder="drop",
    )
    imputed = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), feature_names)],
        remainder="drop",
    )
    return scaled, imputed


def search_registry(feature_names: list[str], positive_ratio: float) -> dict[str, tuple[Pipeline, dict[str, Any]]]:
    scaled, imputed = _preprocessors(feature_names)
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("preprocess", scaled),
                    ("model", LogisticRegression(max_iter=4_000, random_state=SEED)),
                ]
            ),
            {
                "model__C": loguniform(1e-3, 1e2),
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            Pipeline(
                [
                    ("preprocess", imputed),
                    (
                        "model",
                        RandomForestClassifier(n_jobs=1, random_state=SEED),
                    ),
                ]
            ),
            {
                "model__n_estimators": randint(300, 801),
                "model__max_depth": [None, 8, 12, 18, 24],
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": ["sqrt", 0.4, 0.6, 0.8],
                "model__class_weight": [None, "balanced", "balanced_subsample"],
            },
        ),
        "hist_gradient_boosting": (
            Pipeline(
                [
                    ("preprocess", imputed),
                    ("model", HistGradientBoostingClassifier(random_state=SEED)),
                ]
            ),
            {
                "model__learning_rate": loguniform(0.02, 0.20),
                "model__max_iter": randint(200, 601),
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": randint(10, 51),
                "model__l2_regularization": loguniform(1e-3, 20),
                "model__class_weight": [None, "balanced"],
            },
        ),
        "xgboost": (
            Pipeline(
                [
                    ("preprocess", imputed),
                    (
                        "model",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            tree_method="hist",
                            n_jobs=1,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": randint(250, 751),
                "model__learning_rate": loguniform(0.015, 0.15),
                "model__max_depth": randint(2, 8),
                "model__min_child_weight": randint(1, 10),
                "model__subsample": uniform(0.65, 0.35),
                "model__colsample_bytree": uniform(0.65, 0.35),
                "model__reg_lambda": loguniform(0.1, 20),
                "model__reg_alpha": loguniform(1e-4, 2),
                "model__scale_pos_weight": [1.0, positive_ratio],
            },
        ),
    }


def metrics_at_threshold(y_true: pd.Series | np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
    }


def best_f1_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    candidates = np.linspace(0.05, 0.95, 181)
    rows = [metrics_at_threshold(y_true, probability, threshold) for threshold in candidates]
    curve = pd.DataFrame(rows)
    best = curve.sort_values(["f1", "recall"], ascending=False).iloc[0]
    return float(best["threshold"]), curve


def bootstrap_intervals(y_true: pd.Series, probability: np.ndarray, threshold: float, samples: int = 1_000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    y = np.asarray(y_true)
    rows: list[dict[str, float]] = []
    for _ in range(samples):
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            continue
        rows.append(metrics_at_threshold(y[indices], probability[indices], threshold))
    boot = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "metric": ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
            "lower_95": [boot[column].quantile(0.025) for column in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]],
            "upper_95": [boot[column].quantile(0.975) for column in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]],
        }
    )


def _save_model_comparison(summary: pd.DataFrame) -> None:
    plot_data = summary.melt(
        id_vars="model",
        value_vars=["cv_roc_auc", "dev_roc_auc", "dev_pr_auc", "dev_f1"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=plot_data, x="model", y="score", hue="metric")
    ax.set(title="Cross-validation selection and untouched development performance", xlabel="Model", ylabel="Score", ylim=(0, 1))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tuned_model_comparison.png", dpi=180)
    plt.close()


def _save_threshold_curve(curve: pd.DataFrame, threshold: float) -> None:
    plt.figure(figsize=(8, 5))
    for metric in ["precision", "recall", "f1"]:
        plt.plot(curve["threshold"], curve[metric], label=metric.title())
    plt.axvline(threshold, color="#333333", linestyle="--", label=f"Selected: {threshold:.3f}")
    plt.title("Training out-of-fold threshold selection")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "threshold_selection.png", dpi=180)
    plt.close()


def _save_evaluation_plots(y_dev: pd.Series, probability: np.ndarray, prediction: np.ndarray, model_name: str) -> None:
    ConfusionMatrixDisplay.from_predictions(y_dev, prediction, cmap="Blues", colorbar=False)
    plt.title(f"{model_name}: development confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    RocCurveDisplay.from_predictions(y_dev, probability)
    plt.title(f"{model_name}: development ROC curve")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curve.png", dpi=180)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_dev, probability)
    plt.title(f"{model_name}: development precision-recall curve")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "precision_recall_curve.png", dpi=180)
    plt.close()

    CalibrationDisplay.from_predictions(y_dev, probability, n_bins=10, strategy="quantile")
    plt.title(f"{model_name}: development calibration")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "calibration_curve.png", dpi=180)
    plt.close()


def _positive_class_explanation(values: shap.Explanation) -> shap.Explanation:
    if values.values.ndim != 3:
        return values
    base_values = values.base_values
    if np.asarray(base_values).ndim == 2:
        base_values = np.asarray(base_values)[:, 1]
    return shap.Explanation(
        values=values.values[:, :, 1],
        base_values=base_values,
        data=values.data,
        feature_names=values.feature_names,
    )


def generate_shap_artifacts(model: Pipeline, x_dev: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    transformed = np.asarray(preprocessor.transform(x_dev))
    rng = np.random.default_rng(SEED)
    sample_indices = rng.choice(len(x_dev), size=min(1_000, len(x_dev)), replace=False)
    background_indices = rng.choice(len(x_dev), size=min(300, len(x_dev)), replace=False)
    sample = transformed[sample_indices]
    background = transformed[background_indices]

    if isinstance(estimator, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(estimator, feature_names=feature_names)
    else:
        explainer = shap.Explainer(estimator, background, feature_names=feature_names)
    values = _positive_class_explanation(explainer(sample))
    shap_table = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": np.abs(values.values).mean(axis=0),
            "mean_signed_shap": values.values.mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    shap_table.to_csv(REPORT_DIR / "shap_importance.csv", index=False)

    shap.plots.beeswarm(values, max_display=20, show=False)
    plt.title("SHAP effects on predicted churn")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    shap.plots.bar(values, max_display=20, show=False)
    plt.title("Global mean absolute SHAP importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_importance.png", dpi=180, bbox_inches="tight")
    plt.close()

    largest_explanation = int(np.argmax(np.abs(values.values).sum(axis=1)))
    shap.plots.waterfall(values[largest_explanation], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_example_player.png", dpi=180, bbox_inches="tight")
    plt.close()
    return shap_table


def save_error_analysis(
    dev: pd.DataFrame,
    probability: np.ndarray,
    prediction: np.ndarray,
    important_features: list[str],
) -> pd.DataFrame:
    result = dev[["user_id", "label"] + important_features].copy()
    result.insert(2, "churn_probability", probability)
    result.insert(3, "predicted_label", prediction)
    result["outcome"] = np.select(
        [
            (result["label"] == 1) & (result["predicted_label"] == 1),
            (result["label"] == 0) & (result["predicted_label"] == 0),
            (result["label"] == 0) & (result["predicted_label"] == 1),
            (result["label"] == 1) & (result["predicted_label"] == 0),
        ],
        ["true_positive", "true_negative", "false_positive", "false_negative"],
        default="unknown",
    )
    result.to_csv(REPORT_DIR / "development_error_cases.csv", index=False)
    summary = result.groupby("outcome").agg(
        players=("user_id", "size"),
        mean_probability=("churn_probability", "mean"),
        **{f"median_{feature}": (feature, "median") for feature in important_features[:8]},
    )
    summary.to_csv(REPORT_DIR / "error_analysis_summary.csv")
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, dev, test = load_splits()
    feature_names = [column for column in train.columns if column not in EXCLUDED]
    x_train, y_train = train[feature_names], train["label"]
    x_dev, y_dev = dev[feature_names], dev["label"]
    x_test = test[feature_names]
    positive_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    search_summaries: list[dict[str, Any]] = []
    all_search_results: list[pd.DataFrame] = []
    searches: dict[str, RandomizedSearchCV] = {}

    for name, (pipeline, parameters) in search_registry(feature_names, positive_ratio).items():
        print(f"Tuning {name} ({SEARCH_ITERATIONS} candidates x {CV_FOLDS} folds)...", flush=True)
        search = RandomizedSearchCV(
            pipeline,
            parameters,
            n_iter=SEARCH_ITERATIONS,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"},
            refit="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=SEED,
            return_train_score=False,
            verbose=0,
        )
        search.fit(x_train, y_train)
        searches[name] = search
        result = pd.DataFrame(search.cv_results_)
        result.insert(0, "model", name)
        all_search_results.append(result)
        best_index = int(search.best_index_)
        search_summaries.append(
            {
                "model": name,
                "cv_roc_auc": float(result.loc[best_index, "mean_test_roc_auc"]),
                "cv_roc_auc_std": float(result.loc[best_index, "std_test_roc_auc"]),
                "cv_pr_auc": float(result.loc[best_index, "mean_test_pr_auc"]),
                "cv_f1_at_0_5": float(result.loc[best_index, "mean_test_f1"]),
                "best_parameters": json.dumps(_json_safe(search.best_params_), sort_keys=True),
            }
        )

    pd.concat(all_search_results, ignore_index=True).to_csv(REPORT_DIR / "hyperparameter_search_results.csv", index=False)
    selection = pd.DataFrame(search_summaries).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    best_name = str(selection.iloc[0]["model"])
    best_model = searches[best_name].best_estimator_

    print(f"Generating out-of-fold predictions for {best_name}...", flush=True)
    oof_probability = cross_val_predict(
        clone(best_model),
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold, threshold_curve = best_f1_threshold(y_train, oof_probability)
    threshold_curve.to_csv(REPORT_DIR / "threshold_search.csv", index=False)
    _save_threshold_curve(threshold_curve, threshold)

    development_rows: list[dict[str, Any]] = []
    development_probabilities: dict[str, np.ndarray] = {}
    for name, search in searches.items():
        probability = search.best_estimator_.predict_proba(x_dev)[:, 1]
        development_probabilities[name] = probability
        row: dict[str, Any] = {"model": name}
        row.update(metrics_at_threshold(y_dev, probability, threshold if name == best_name else 0.5))
        development_rows.append(row)
    development = pd.DataFrame(development_rows)
    selection = selection.merge(
        development.rename(columns={column: f"dev_{column}" for column in development.columns if column != "model"}),
        on="model",
        how="left",
    )
    selection.to_csv(REPORT_DIR / "model_selection_summary.csv", index=False)
    selection[["model", "dev_threshold", "dev_accuracy", "dev_precision", "dev_recall", "dev_f1", "dev_roc_auc", "dev_pr_auc"]].to_csv(
        REPORT_DIR / "metrics.csv", index=False
    )
    _save_model_comparison(selection)

    dev_probability = development_probabilities[best_name]
    dev_prediction = (dev_probability >= threshold).astype(int)
    final_metrics = metrics_at_threshold(y_dev, dev_probability, threshold)
    confidence_intervals = bootstrap_intervals(y_dev, dev_probability, threshold)
    confidence_intervals.to_csv(REPORT_DIR / "metric_confidence_intervals.csv", index=False)
    _save_evaluation_plots(y_dev, dev_probability, dev_prediction, best_name)

    print("Generating SHAP explanations...", flush=True)
    shap_importance = generate_shap_artifacts(best_model, x_dev, feature_names)
    top_features = shap_importance.head(10)["feature"].tolist()
    error_summary = save_error_analysis(dev, dev_probability, dev_prediction, top_features)

    test_probability = best_model.predict_proba(x_test)[:, 1]
    pd.DataFrame(
        {
            "user_id": test["user_id"],
            "churn_probability": test_probability,
            "predicted_label": (test_probability >= threshold).astype(int),
        }
    ).to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    joblib.dump(best_model, MODEL_DIR / "best_churn_model.joblib")
    metadata = {
        "model": best_name,
        "selection_rule": "highest mean 5-fold training ROC-AUC",
        "features": feature_names,
        "best_parameters": searches[best_name].best_params_,
        "decision_threshold": threshold,
        "threshold_rule": "highest training out-of-fold churn F1; recall breaks ties",
        "cross_validation": {"folds": CV_FOLDS, "shuffle": True},
        "training_oof_metrics": metrics_at_threshold(y_train, oof_probability, threshold),
        "development_metrics": final_metrics,
        "random_seed": SEED,
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2), encoding="utf-8"
    )

    ci_lookup = confidence_intervals.set_index("metric")
    top_lines = "\n".join(
        f"- `{row.feature}` (mean |SHAP| = {row.mean_absolute_shap:.4f})"
        for row in shap_importance.head(10).itertuples()
    )
    error_counts = error_summary["players"].to_dict()
    summary = f"""# Tuned player-churn model results

## Protocol

Four model families were tuned using **{CV_FOLDS}-fold stratified cross-validation on the training split only**, with **{SEARCH_ITERATIONS} randomized configurations per family**. The winning family and hyperparameters were selected by mean cross-validated ROC-AUC. Its decision threshold was selected from training out-of-fold predictions by churn-class F1. The official development split was then evaluated once and was not used for model or threshold selection.

## Selected model

- Model: **{best_name}**
- Cross-validated training ROC-AUC: **{selection.iloc[0]['cv_roc_auc']:.4f} +/- {selection.iloc[0]['cv_roc_auc_std']:.4f}**
- Training-only decision threshold: **{threshold:.3f}**
- Development ROC-AUC: **{final_metrics['roc_auc']:.4f}** (bootstrap 95% CI {ci_lookup.loc['roc_auc', 'lower_95']:.4f}-{ci_lookup.loc['roc_auc', 'upper_95']:.4f})
- Development PR-AUC: **{final_metrics['pr_auc']:.4f}** (95% CI {ci_lookup.loc['pr_auc', 'lower_95']:.4f}-{ci_lookup.loc['pr_auc', 'upper_95']:.4f})
- Development precision: **{final_metrics['precision']:.4f}**
- Development recall: **{final_metrics['recall']:.4f}**
- Development F1: **{final_metrics['f1']:.4f}** (95% CI {ci_lookup.loc['f1', 'lower_95']:.4f}-{ci_lookup.loc['f1', 'upper_95']:.4f})
- Development accuracy: **{final_metrics['accuracy']:.4f}**

## Most influential predictors

SHAP values describe model associations, not causal effects:

{top_lines}

## Error analysis

- True positives: **{error_counts.get('true_positive', 0):,}**
- True negatives: **{error_counts.get('true_negative', 0):,}**
- False positives: **{error_counts.get('false_positive', 0):,}**
- False negatives: **{error_counts.get('false_negative', 0):,}**

False positives represent players who look behaviorally at risk but remain active; false negatives are the most important missed-retention opportunities. The accompanying error table supports comparison of their behavioral profiles.

## Limitations

The data covers one game and a short observation period, labels inactivity rather than permanent abandonment, and does not support causal conclusions. External validation on another game or later cohort is still required before deployment. The unlabeled test predictions are produced for submission or later evaluation, not reported as performance evidence.
"""
    (REPORT_DIR / "project_summary.md").write_text(summary, encoding="utf-8")

    print("\nModel selection summary:")
    print(selection[["model", "cv_roc_auc", "dev_roc_auc", "dev_pr_auc", "dev_f1"]].to_string(index=False))
    print(f"\nSelected {best_name}; training-only threshold={threshold:.3f}.")


if __name__ == "__main__":
    main()
