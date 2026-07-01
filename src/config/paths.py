"""Centralized filesystem paths for the Trendyol Datathon project.

All project modules should import paths from this file instead of defining
hardcoded filesystem locations.
"""

from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
SUBMISSIONS_DIR: Path = PROJECT_ROOT / "submissions"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
TESTS_DIR: Path = PROJECT_ROOT / "tests"

ITEMS_FILE: Path = RAW_DATA_DIR / "items.csv"
TERMS_FILE: Path = RAW_DATA_DIR / "terms.csv"
TRAINING_PAIRS_FILE: Path = RAW_DATA_DIR / "training_pairs.csv"
SUBMISSION_PAIRS_FILE: Path = RAW_DATA_DIR / "submission_pairs.csv"
SAMPLE_SUBMISSION_FILE: Path = RAW_DATA_DIR / "sample_submission.csv"
