# Early Prediction of Player Churn in Mobile Games

This project predicts whether a mobile-game player will become inactive in the following week using early gameplay logs. It turns 2.19 million level-attempt events into one feature row per player, compares multiple classifiers, and explains the strongest model's churn signals.

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
|-- notebooks/                # optional exploratory notebooks
|-- reports/
|   |-- figures/              # generated plots
|   |-- metrics.csv           # validation model comparison
|   `-- project_summary.md    # generated result summary
|-- src/
|   |-- download_data.py
|   |-- build_features.py
|   |-- train_models.py
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
.\.venv\Scripts\python.exe src\train_models.py
```

## Current modeling plan

1. Aggregate attempt, progress, success, retry, help-use, duration, difficulty, and time-based features per player.
2. Preserve the dataset's official train/development/test split.
3. Compare Logistic Regression, Random Forest, Histogram Gradient Boosting, and XGBoost.
4. Select the best model by validation ROC-AUC; also report precision, recall, F1, PR-AUC, and accuracy.
5. Tune the decision threshold on the development set for churn F1.
6. Produce a test prediction file and model-level feature importance.

## Leakage precautions

- `user_id` is retained only for joining and is excluded from model inputs.
- Labels are never used during feature generation.
- Model selection and threshold selection use only the official development set.
- The untouched test set has no public labels, so it is used only for final predictions.

## Reproducibility

All random models use seed `437`. Generated files can be recreated by rerunning `src/run_pipeline.py`.
