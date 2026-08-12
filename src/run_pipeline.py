"""Run feature generation followed by baseline model training."""

from build_features import main as build_features
from download_data import ensure_dataset
from train_models import main as train_models


if __name__ == "__main__":
    ensure_dataset()
    build_features()
    train_models()
