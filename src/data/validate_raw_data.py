"""Validate that all raw datasets can be loaded and structurally reported."""

from collections.abc import Callable

import pandas as pd

from src.data.data_loader import (
    load_items,
    load_sample_submission,
    load_submission_pairs,
    load_terms,
    load_training_pairs,
)
from src.data.data_reporter import DatasetReport, report_dataset
from src.utils.logger import get_logger


logger = get_logger(__name__)

DatasetLoader = Callable[[], pd.DataFrame]


def validate_raw_datasets() -> list[DatasetReport]:
    """Load and report all raw datasets required for Sprint 1 validation.

    Returns:
        A list of generated dataset reports.
    """
    dataset_loaders: tuple[tuple[str, DatasetLoader], ...] = (
        ("items", load_items),
        ("terms", load_terms),
        ("training_pairs", load_training_pairs),
        ("submission_pairs", load_submission_pairs),
        ("sample_submission", load_sample_submission),
    )

    logger.info("Starting raw data validation")
    reports: list[DatasetReport] = []

    for dataset_name, loader in dataset_loaders:
        logger.info("Validating dataset: %s", dataset_name)
        dataframe = loader()
        report = report_dataset(dataframe, dataset_name)
        reports.append(report)

    logger.info("Raw data validation completed successfully")
    return reports


def main() -> None:
    """Run raw data validation as a command-line entrypoint."""
    validate_raw_datasets()


if __name__ == "__main__":
    main()
