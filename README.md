# Early Prediction of Player Churn in Mobile Games

This project predicts whether a mobile-game player will become inactive in the following week using early gameplay logs. It turns 2.19 million level-attempt events into one feature row per player, compares multiple classifiers, and explains the strongest model's churn signals.

## Final result

The training set has a 33.5% churn rate, so this is a **mild-imbalance (Group 2)** classification problem. Four model families were tuned with five-fold stratified cross-validation. Random Forest was selected by training CV AUROC and evaluated once on the official development split:

- Development AUROC: **0.8024**
- Precision: **0.5789**
- Recall: **0.7814**
- F1-score: **0.6651**
- AUPRC: **0.6359**

The decision threshold (0.320) was selected using training out-of-fold predictions, not development labels.

## Dataset

- Source: [Prediction of User Loss in Mobile Games](https://www.kaggle.com/datasets/manchvictor/prediction-of-user-loss-in-mobile-games)
- `train.csv`: 8,158 labeled players
- `dev.csv`: 2,658 labeled validation players
- `test.csv`: 2,773 unlabeled players
- `level_seq.csv`: 2,194,351 gameplay events
- `level_meta.csv`: aggregate information for 1,509 levels

The source files have a `.csv` extension but use tab delimiters. The scripts handle this automatically.

## Project structure

```text
mobile-game-churn/
|-- data/
|   |-- raw/data/             # original Kaggle files
|   `-- processed/            # generated player-level tables
|-- models/                   # fitted model and feature list
|-- notebooks/                # readable exploratory/modeling notebook
|-- deliverables/
|   |-- mobile_game_churn_project_report.docx
|   `-- mobile_game_churn_project_presentation.pptx
|-- reports/
|   |-- figures/              # generated plots
|   |-- metrics.csv           # holdout model comparison
|   |-- model_selection_summary.csv
|   |-- shap_importance.csv
|   `-- project_summary.md    # generated result summary
|-- src/
|   |-- download_data.py
|   |-- build_features.py
|   |-- eda.py
|   |-- train_models.py
|   |-- workflow_figure.py
|   |-- build_report.py
|   `-- run_pipeline.py
`-- requirements.txt
```

## Setup and run (PowerShell)

The workspace already contains a ready-to-use `.venv`. To rerun the complete pipeline (the raw dataset is downloaded automatically if missing):

```powershell
cd E:\Work\BRACU\CSE437\Project\mobile-game-churn
.\.venv\Scripts\python.exe src\run_pipeline.py
```

If the environment is removed, recreate it with an installed Python 3.12+ interpreter and install the requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\run_pipeline.py
```

Run the stages independently if needed:

```powershell
.\.venv\Scripts\python.exe src\build_features.py
.\.venv\Scripts\python.exe src\eda.py
.\.venv\Scripts\python.exe src\train_models.py
.\.venv\Scripts\python.exe src\workflow_figure.py
```

The DOCX report can be rebuilt with the bundled workspace Python or an environment containing `python-docx`:

```powershell
.\.venv\Scripts\python.exe src\build_report.py
```

## Modeling protocol

1. Aggregate attempt, progress, success, retry, help-use, duration, difficulty, and time-based features per player.
2. Preserve the dataset's official train/development/test split.
3. Compare Logistic Regression, Random Forest, Histogram Gradient Boosting, and XGBoost.
4. Tune 15 configurations per family with five-fold stratified training CV.
5. Select the family and hyperparameters by mean training CV AUROC.
6. Select the churn-F1 threshold from training out-of-fold probabilities.
7. Evaluate the fixed model and threshold once on the development split.
8. Produce bootstrap intervals, SHAP explanations, error analysis, and unlabeled test predictions.

## Leakage precautions

- `user_id` is retained only for joining and is excluded from model inputs.
- Labels are never used during feature generation.
- Model, hyperparameter, and threshold selection use only the official training split.
- The labeled development set is an untouched final holdout.
- The untouched test set has no public labels, so it is used only for final predictions.

## Reproducibility

All random models use seed `437`. Generated files can be recreated by rerunning `src/run_pipeline.py`.
