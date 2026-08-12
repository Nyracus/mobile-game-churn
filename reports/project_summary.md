# Initial model results

- Best model by development ROC-AUC: **logistic_regression**
- Tuned decision threshold: **0.455**
- ROC-AUC: **0.8058**
- PR-AUC: **0.6531**
- Precision: **0.5805**
- Recall: **0.7880**
- F1-score: **0.6685**

Top model features: duration_per_active_day, attempts_per_active_day, levels_per_active_day, hours_since_last_activity, retry_attempts, total_duration, session_count, unique_levels, total_attempts, mean_level_retrytimes.

These are baseline development-set results. The next research stage should add cross-validation, formal hyperparameter optimization, SHAP explanations, and error analysis before making final claims.
