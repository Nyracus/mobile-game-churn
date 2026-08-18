# Exploratory data analysis

## Dataset quality

- Players: **13,589** (8,158 train, 2,658 development, 2,773 test)
- Engineered predictors: **41**
- Training churn rate: **33.5%**
- Development churn rate: **33.9%**
- Missing engineered training values: **0** (handled inside model pipelines)
- Duplicate training player IDs: **0**

The moderate class imbalance makes accuracy insufficient on its own. Model assessment therefore emphasizes ROC-AUC, PR-AUC, churn recall, and churn F1.

## Largest univariate behavior differences

Values below are churned-player means minus retained-player means in pooled standard-deviation units. They are descriptive associations, not causal effects.

- `hours_since_last_activity`: +0.97 pooled SD
- `active_days`: -0.96 pooled SD
- `observation_span_hours`: -0.94 pooled SD
- `session_count`: -0.82 pooled SD
- `mean_reststep`: +0.78 pooled SD
- `unique_levels`: -0.76 pooled SD
- `mean_level_passrate`: +0.76 pooled SD
- `weekend_attempt_rate`: +0.72 pooled SD

The strongest raw pattern is recency: churned users generally stopped playing earlier within the observation window. Churned players also attempted fewer levels and generated less progression activity. These variables may be useful for prediction, but interventions inferred from them require experimental validation.

## Generated artifacts

- `reports/behavior_by_churn.csv`: per-feature group comparison
- `reports/eda_numeric_summary.csv`: distribution and missingness summary
- `reports/split_summary.csv`: official split sizes and label rates
- `reports/figures/eda_*.png`: class, distribution, effect-size, and correlation plots
