"""Build one leakage-safe feature row per player from raw gameplay events."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def read_source(name: str, **kwargs: object) -> pd.DataFrame:
    """Read a tab-delimited source file and fail clearly if it is missing."""
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download and extract the Kaggle dataset into {RAW_DIR}."
        )
    return pd.read_csv(path, sep="\t", **kwargs)


def build_event_features() -> pd.DataFrame:
    level_meta = read_source("level_meta.csv")
    events = read_source("level_seq.csv", parse_dates=["time"])

    events = events.merge(level_meta, on="level_id", how="left", validate="many_to_one")
    events = events.sort_values(["user_id", "time", "level_id"], kind="stable")

    events["activity_date"] = events["time"].dt.floor("D")
    events["is_weekend"] = (events["time"].dt.dayofweek >= 5).astype("int8")
    events["is_night"] = events["time"].dt.hour.isin([0, 1, 2, 3, 4, 5]).astype("int8")
    events["failed"] = 1 - events["f_success"]
    events["hard_level"] = (events["f_avg_passrate"] < 0.50).astype("int8")
    events["duration_ratio"] = events["f_duration"] / events["f_avg_duration"].replace(0, np.nan)

    gap_minutes = events.groupby("user_id", sort=False)["time"].diff().dt.total_seconds() / 60
    events["new_session"] = (gap_minutes.isna() | (gap_minutes > 30)).astype("int8")

    grouped = events.groupby("user_id", sort=False)
    features = grouped.agg(
        total_attempts=("level_id", "size"),
        unique_levels=("level_id", "nunique"),
        min_level=("level_id", "min"),
        max_level=("level_id", "max"),
        mean_level=("level_id", "mean"),
        level_std=("level_id", "std"),
        successes=("f_success", "sum"),
        success_rate=("f_success", "mean"),
        failures=("failed", "sum"),
        help_uses=("f_help", "sum"),
        help_rate=("f_help", "mean"),
        total_duration=("f_duration", "sum"),
        mean_duration=("f_duration", "mean"),
        median_duration=("f_duration", "median"),
        duration_std=("f_duration", "std"),
        max_duration=("f_duration", "max"),
        mean_reststep=("f_reststep", "mean"),
        reststep_std=("f_reststep", "std"),
        min_reststep=("f_reststep", "min"),
        active_days=("activity_date", "nunique"),
        session_count=("new_session", "sum"),
        weekend_attempt_rate=("is_weekend", "mean"),
        night_attempt_rate=("is_night", "mean"),
        hard_level_attempt_rate=("hard_level", "mean"),
        mean_level_passrate=("f_avg_passrate", "mean"),
        mean_level_retrytimes=("f_avg_retrytimes", "mean"),
        mean_duration_ratio=("duration_ratio", "mean"),
        first_activity=("time", "min"),
        last_activity=("time", "max"),
    )

    features["retry_attempts"] = features["total_attempts"] - features["unique_levels"]
    features["retry_rate"] = features["retry_attempts"] / features["total_attempts"]
    features["attempts_per_level"] = features["total_attempts"] / features["unique_levels"]
    features["attempts_per_active_day"] = features["total_attempts"] / features["active_days"]
    features["duration_per_active_day"] = features["total_duration"] / features["active_days"]
    features["levels_per_active_day"] = features["unique_levels"] / features["active_days"]
    features["observation_span_hours"] = (
        features["last_activity"] - features["first_activity"]
    ).dt.total_seconds() / 3600
    dataset_end = events["time"].max()
    features["hours_since_last_activity"] = (
        dataset_end - features["last_activity"]
    ).dt.total_seconds() / 3600

    last_25 = grouped.tail(25).groupby("user_id", sort=False).agg(
        last25_success_rate=("f_success", "mean"),
        last25_mean_duration=("f_duration", "mean"),
        last25_help_rate=("f_help", "mean"),
        last25_mean_level=("level_id", "mean"),
    )
    features = features.join(last_25)
    features["success_rate_change"] = features["last25_success_rate"] - features["success_rate"]
    features["duration_change_ratio"] = (
        features["last25_mean_duration"] / features["mean_duration"].replace(0, np.nan)
    )

    features = features.drop(columns=["first_activity", "last_activity"])
    features = features.replace([np.inf, -np.inf], np.nan).reset_index()
    return features


def attach_split(
    player_features: pd.DataFrame, filename: str, split_name: str
) -> pd.DataFrame:
    members = read_source(filename)
    merged = members.merge(player_features, on="user_id", how="left", validate="one_to_one")
    if merged.drop(columns=["label"], errors="ignore").isna().all(axis=1).any():
        raise ValueError(f"At least one {split_name} player has no gameplay events.")
    merged.insert(1, "split", split_name)
    return merged


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    player_features = build_event_features()

    train = attach_split(player_features, "train.csv", "train")
    dev = attach_split(player_features, "dev.csv", "dev")
    test = attach_split(player_features, "test.csv", "test")

    player_features.to_csv(PROCESSED_DIR / "all_player_features.csv", index=False)
    train.to_csv(PROCESSED_DIR / "train_features.csv", index=False)
    dev.to_csv(PROCESSED_DIR / "dev_features.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test_features.csv", index=False)

    print(f"Built {player_features.shape[1] - 1} features for {len(player_features):,} players.")
    print(f"Train={len(train):,}, dev={len(dev):,}, test={len(test):,}")


if __name__ == "__main__":
    main()

