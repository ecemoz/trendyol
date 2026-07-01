"""Dataset reporting utilities for raw data validation."""

from dataclasses import dataclass

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)
DEFAULT_PREVIEW_ROWS: int = 5
BYTES_PER_MEGABYTE: int = 1024 * 1024


@dataclass(frozen=True)
class DatasetReport:
    """Container for non-transformative dataset validation metadata."""

    name: str
    shape: tuple[int, int]
    columns: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    memory_usage_mb: float
    duplicate_rows: int
    preview: pd.DataFrame


def build_dataset_report(
    dataframe: pd.DataFrame,
    dataset_name: str,
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
) -> DatasetReport:
    """Build a structural validation report for a DataFrame.

    Args:
        dataframe: Dataset to inspect without modifying it.
        dataset_name: Human-readable dataset name.
        preview_rows: Number of first rows to include in the preview.

    Returns:
        Dataset report containing shape, schema, missingness, memory,
        duplicate count, and a small preview.
    """
    logger.info("Building validation report for %s", dataset_name)

    memory_usage_mb = (
        dataframe.memory_usage(deep=True).sum() / BYTES_PER_MEGABYTE
    )

    report = DatasetReport(
        name=dataset_name,
        shape=dataframe.shape,
        columns=list(dataframe.columns),
        dtypes={column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        missing_values=dataframe.isna().sum().astype(int).to_dict(),
        memory_usage_mb=memory_usage_mb,
        duplicate_rows=int(dataframe.duplicated().sum()),
        preview=dataframe.head(preview_rows),
    )

    logger.info(
        "Report completed for %s | shape=%s | duplicate_rows=%s | memory=%.3f MB",
        report.name,
        report.shape,
        report.duplicate_rows,
        report.memory_usage_mb,
    )

    return report


def print_dataset_report(report: DatasetReport) -> None:
    """Print a dataset report in a readable console format.

    Args:
        report: Dataset report to print.
    """
    separator = "=" * 80
    subsection_separator = "-" * 80

    print(separator)
    print(f"Dataset: {report.name}")
    print(separator)

    print(f"Shape: {report.shape}")
    print(f"Memory Usage: {report.memory_usage_mb:.3f} MB")
    print(f"Duplicate Rows: {report.duplicate_rows}")

    print(subsection_separator)
    print("Columns:")
    for column in report.columns:
        print(f"- {column}")

    print(subsection_separator)
    print("Data Types:")
    for column, dtype in report.dtypes.items():
        print(f"- {column}: {dtype}")

    print(subsection_separator)
    print("Missing Values:")
    for column, missing_count in report.missing_values.items():
        print(f"- {column}: {missing_count}")

    print(subsection_separator)
    print("First 5 Rows:")
    print(report.preview)
    print(separator)


def report_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
) -> DatasetReport:
    """Build and print a structural validation report for a DataFrame.

    Args:
        dataframe: Dataset to inspect without modifying it.
        dataset_name: Human-readable dataset name.
        preview_rows: Number of first rows to include in the preview.

    Returns:
        The generated dataset report.
    """
    report = build_dataset_report(
        dataframe=dataframe,
        dataset_name=dataset_name,
        preview_rows=preview_rows,
    )
    print_dataset_report(report)
    return report
