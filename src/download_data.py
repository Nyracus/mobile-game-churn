"""Download and safely extract the public Kaggle dataset."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXPECTED_FILE = RAW_DIR / "data" / "level_seq.csv"
ARCHIVE_PATH = DATA_DIR / "mobile-game-churn.zip"
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "manchvictor/prediction-of-user-loss-in-mobile-games"
)


def _safe_extract(archive: ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination != target and destination not in target.parents:
            raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
    archive.extractall(destination)


def ensure_dataset() -> None:
    if EXPECTED_FILE.exists():
        print(f"Dataset already available at {RAW_DIR}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading the public mobile-game churn dataset from Kaggle...")
    urlretrieve(DATASET_URL, ARCHIVE_PATH)
    with ZipFile(ARCHIVE_PATH) as archive:
        _safe_extract(archive, RAW_DIR)

    if not EXPECTED_FILE.exists():
        raise FileNotFoundError("The archive was downloaded but did not contain the expected files.")
    print(f"Dataset extracted to {RAW_DIR}")


if __name__ == "__main__":
    ensure_dataset()

