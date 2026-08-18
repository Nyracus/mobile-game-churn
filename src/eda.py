"""Generate reproducible exploratory analysis tables, figures, and a short report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
EXCLUDED = {"user_id", "split", "label"}


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(
        pd.read_csv(PROCESSED_DIR / f"{name}_features.csv")
        for name in ("train", "dev", "test")
    )  # type: ignore[return-value]


def _save_class_distribution(train: pd.DataFrame) -> None:
    counts = train["label"].value_counts().sort_index()
    labels = ["Retained", "Churned"]
    plt.figure(figsize=(6.5, 4.5))
    bars = plt.bar(labels, counts.values, color=["#4c78a8", "#e45756"])
    for bar, count in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, count + 70, f"{count:,}", ha="center")
    plt.title(f"Training labels (churn rate: {train['label'].mean():.1%})")
    plt.ylabel("Players")
    plt.ylim(0, counts.max() * 1.13)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_class_distribution.png", dpi=180)
    plt.close()


def _save_behavior_comparison(comparison: pd.DataFrame) -> None:
    top = comparison.reindex(comparison["standardized_mean_difference"].abs().nlargest(12).index)
    top = top.sort_values("standardized_mean_difference")
    colors = np.where(top["standardized_mean_difference"] > 0, "#e45756", "#4c78a8")
    plt.figure(figsize=(9, 6.5))
    plt.barh(top.index, top["standardized_mean_difference"], color=colors)
    plt.axvline(0, color="#555555", linewidth=1)
    plt.title("Largest standardized behavior differences")
    plt.xlabel("Churned mean minus retained mean (pooled SD units)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_behavior_differences.png", dpi=180)
    plt.close()


def _save_feature_distributions(train: pd.DataFrame) -> None:
    selected = [
        "hours_since_last_activity",
        "total_attempts",
        "max_level",
        "active_days",
        "success_rate",
        "retry_rate",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for feature, ax in zip(selected, axes.flat):
        cap = train[feature].quantile(0.99)
        plot_data = train.loc[train[feature] <= cap, [feature, "label"]].copy()
        plot_data["Status"] = plot_data["label"].map({0: "Retained", 1: "Churned"})
        sns.histplot(
            data=plot_data,
            x=feature,
            hue="Status",
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            ax=ax,
        )
        ax.set_title(feature.replace("_", " ").title())
        ax.set_ylabel("Density")
    fig.suptitle("Player behavior distributions by churn label (trimmed at 99th percentile)", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_feature_distributions.png", dpi=180, bbox_inches="tight")
    plt.close()


def _save_correlation_heatmap(train: pd.DataFrame, features: list[str]) -> None:
    label_correlations = train[features + ["label"]].corr(numeric_only=True)["label"].drop("label")
    selected = label_correlations.abs().nlargest(15).index.tolist()
    correlation = train[selected].corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(correlation, cmap="vlag", center=0, vmin=-1, vmax=1, square=True)
    plt.title("Correlation among the 15 features most associated with churn")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_correlation_heatmap.png", dpi=180)
    plt.close()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    train, dev, test = _load()
    features = [column for column in train.columns if column not in EXCLUDED]

    numeric_summary = train[features].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    numeric_summary["missing"] = train[features].isna().sum()
    numeric_summary.to_csv(REPORT_DIR / "eda_numeric_summary.csv")

    retained = train.loc[train["label"] == 0, features]
    churned = train.loc[train["label"] == 1, features]
    pooled_sd = train[features].std().replace(0, np.nan)
    comparison = pd.DataFrame(
        {
            "retained_mean": retained.mean(),
            "churned_mean": churned.mean(),
            "retained_median": retained.median(),
            "churned_median": churned.median(),
        }
    )
    comparison["standardized_mean_difference"] = (
        comparison["churned_mean"] - comparison["retained_mean"]
    ) / pooled_sd
    comparison["absolute_standardized_difference"] = comparison[
        "standardized_mean_difference"
    ].abs()
    comparison = comparison.sort_values("absolute_standardized_difference", ascending=False)
    comparison.to_csv(REPORT_DIR / "behavior_by_churn.csv")

    split_summary = pd.DataFrame(
        [
            {"split": "train", "players": len(train), "labeled": True, "churn_rate": train["label"].mean()},
            {"split": "dev", "players": len(dev), "labeled": True, "churn_rate": dev["label"].mean()},
            {"split": "test", "players": len(test), "labeled": False, "churn_rate": np.nan},
        ]
    )
    split_summary.to_csv(REPORT_DIR / "split_summary.csv", index=False)

    _save_class_distribution(train)
    _save_behavior_comparison(comparison)
    _save_feature_distributions(train)
    _save_correlation_heatmap(train, features)

    top_lines = "\n".join(
        f"- `{feature}`: {row.standardized_mean_difference:+.2f} pooled SD"
        for feature, row in comparison.head(8).iterrows()
    )
    report = f"""# Exploratory data analysis

## Dataset quality

- Players: **{len(train) + len(dev) + len(test):,}** ({len(train):,} train, {len(dev):,} development, {len(test):,} test)
- Engineered predictors: **{len(features)}**
- Training churn rate: **{train['label'].mean():.1%}**
- Development churn rate: **{dev['label'].mean():.1%}**
- Missing engineered training values: **{int(train[features].isna().sum().sum()):,}** (handled inside model pipelines)
- Duplicate training player IDs: **{int(train['user_id'].duplicated().sum()):,}**

The moderate class imbalance makes accuracy insufficient on its own. Model assessment therefore emphasizes ROC-AUC, PR-AUC, churn recall, and churn F1.

## Largest univariate behavior differences

Values below are churned-player means minus retained-player means in pooled standard-deviation units. They are descriptive associations, not causal effects.

{top_lines}

The strongest raw pattern is recency: churned users generally stopped playing earlier within the observation window. Churned players also attempted fewer levels and generated less progression activity. These variables may be useful for prediction, but interventions inferred from them require experimental validation.

## Generated artifacts

- `reports/behavior_by_churn.csv`: per-feature group comparison
- `reports/eda_numeric_summary.csv`: distribution and missingness summary
- `reports/split_summary.csv`: official split sizes and label rates
- `reports/figures/eda_*.png`: class, distribution, effect-size, and correlation plots
"""
    (REPORT_DIR / "eda_report.md").write_text(report, encoding="utf-8")
    print(f"EDA complete for {len(train):,} training players and {len(features)} features.")


if __name__ == "__main__":
    main()

