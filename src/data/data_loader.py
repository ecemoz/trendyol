"""Data loading utilities for raw Trendyol Datathon datasets."""

from pathlib import Path
from time import perf_counter

import pandas as pd

from src.config.paths import (
    ITEMS_FILE,
    SAMPLE_SUBMISSION_FILE,
    SUBMISSION_PAIRS_FILE,
    TERMS_FILE,
    TRAINING_PAIRS_FILE,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def _load_csv(file_path: Path, dataset_name: str) -> pd.DataFrame:
    """Load a CSV file and log basic read metadata.

    Args:
        file_path: Path of the CSV file to read.
        dataset_name: Human-readable dataset name for logs and errors.

    Returns:
        Loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the expected CSV file does not exist.
        RuntimeError: If pandas cannot read the CSV file.
    """
    if not file_path.exists():
        message = f"{dataset_name} file not found at: {file_path}"
        logger.error(message)
        raise FileNotFoundError(message)

    logger.info("Reading %s from %s", dataset_name, file_path)
    start_time = perf_counter()

    try:
        dataframe = pd.read_csv(file_path)
    except Exception as exc:
        message = f"Failed to read {dataset_name} from {file_path}"
        logger.exception(message)
        raise RuntimeError(message) from exc

    elapsed_seconds = perf_counter() - start_time
    row_count, column_count = dataframe.shape

    logger.info(
        "Loaded %s with %s rows and %s columns in %.3f seconds",
        dataset_name,
        row_count,
        column_count,
        elapsed_seconds,
    )

    return dataframe


def load_items() -> pd.DataFrame:
    """Load the raw items dataset.

    Returns:
        Items dataset as a pandas DataFrame.
    """
    return _load_csv(ITEMS_FILE, "items")


def load_terms() -> pd.DataFrame:
    """Load the raw search terms dataset.

    Returns:
        Terms dataset as a pandas DataFrame.
    """
    return _load_csv(TERMS_FILE, "terms")


def load_training_pairs() -> pd.DataFrame:
    """Load the raw training query-product pairs dataset.

    Returns:
        Training pairs dataset as a pandas DataFrame.
    """
    return _load_csv(TRAINING_PAIRS_FILE, "training_pairs")


def load_submission_pairs() -> pd.DataFrame:
    """Load the raw submission query-product pairs dataset.

    Returns:
        Submission pairs dataset as a pandas DataFrame.
    """
    return _load_csv(SUBMISSION_PAIRS_FILE, "submission_pairs")


def load_sample_submission() -> pd.DataFrame:
    """Load the raw sample submission dataset.

    Returns:
        Sample submission dataset as a pandas DataFrame.
    """
    return _load_csv(SAMPLE_SUBMISSION_FILE, "sample_submission")
