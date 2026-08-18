# Tuned player-churn model results

## Protocol

Four model families were tuned using **5-fold stratified cross-validation on the training split only**, with **15 randomized configurations per family**. The winning family and hyperparameters were selected by mean cross-validated ROC-AUC. Its decision threshold was selected from training out-of-fold predictions by churn-class F1. The official development split was then evaluated once and was not used for model or threshold selection.

## Selected model

- Model: **random_forest**
- Cross-validated training ROC-AUC: **0.8067 +/- 0.0064**
- Training-only decision threshold: **0.320**
- Development ROC-AUC: **0.8024** (bootstrap 95% CI 0.7860-0.8196)
- Development PR-AUC: **0.6359** (95% CI 0.6022-0.6693)
- Development precision: **0.5789**
- Development recall: **0.7814**
- Development F1: **0.6651** (95% CI 0.6422-0.6868)
- Development accuracy: **0.7333**

## Most influential predictors

SHAP values describe model associations, not causal effects:

- `hours_since_last_activity` (mean |SHAP| = 0.0514)
- `weekend_attempt_rate` (mean |SHAP| = 0.0331)
- `observation_span_hours` (mean |SHAP| = 0.0315)
- `active_days` (mean |SHAP| = 0.0188)
- `total_duration` (mean |SHAP| = 0.0138)
- `session_count` (mean |SHAP| = 0.0118)
- `last25_mean_level` (mean |SHAP| = 0.0117)
- `reststep_std` (mean |SHAP| = 0.0102)
- `mean_level_retrytimes` (mean |SHAP| = 0.0094)
- `mean_level` (mean |SHAP| = 0.0092)

## Error analysis

- True positives: **704**
- True negatives: **1,245**
- False positives: **512**
- False negatives: **197**

False positives represent players who look behaviorally at risk but remain active; false negatives are the most important missed-retention opportunities. The accompanying error table supports comparison of their behavioral profiles.

## Limitations

The data covers one game and a short observation period, labels inactivity rather than permanent abandonment, and does not support causal conclusions. External validation on another game or later cohort is still required before deployment. The unlabeled test predictions are produced for submission or later evaluation, not reported as performance evidence.
