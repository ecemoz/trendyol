"""Data quality analysis across all raw datasets."""

import re

import pandas as pd

from src.analysis.common import AnalysisResult, is_unknown_like, percentage
from src.data.data_loader import (
    load_items,
    load_sample_submission,
    load_submission_pairs,
    load_terms,
    load_training_pairs,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

LONG_TEXT_QUANTILE: float = 0.99
LONG_TEXT_MIN_LENGTH: int = 120
UNEXPECTED_CHARACTER_PATTERN: re.Pattern[str] = re.compile(
    r"[^0-9A-Za-zÇĞİÖŞÜçğıöşü\s\.,:;!\?\-/&\(\)\[\]'\"%+#_*]",
    flags=re.UNICODE,
)
ENCODING_PROBLEM_PATTERN: re.Pattern[str] = re.compile(
    r"[ÃÂÄÅ�ðþÐÞ]",
    flags=re.UNICODE,
)


def _text_columns(dataframe: pd.DataFrame) -> list[str]:
    """Find object-like columns in a DataFrame.

    Args:
        dataframe: Dataset to inspect.

    Returns:
        Column names with object/string-like dtype.
    """
    return [
        column
        for column in dataframe.columns
        if pd.api.types.is_object_dtype(dataframe[column])
        or pd.api.types.is_string_dtype(dataframe[column])
    ]


def _empty_string_count(series: pd.Series) -> int:
    """Count empty or whitespace-only strings in a series.

    Args:
        series: Series to inspect.

    Returns:
        Empty string count.
    """
    return int(series.fillna("").astype(str).str.strip().eq("").sum())


def _unknown_count(series: pd.Series) -> int:
    """Count unknown-like values in a series.

    Args:
        series: Series to inspect.

    Returns:
        Unknown-like value count.
    """
    return int(series.map(is_unknown_like).sum())


def _unexpected_character_count(series: pd.Series) -> int:
    """Count rows containing unexpected characters.

    Args:
        series: Text series to inspect.

    Returns:
        Number of rows containing unexpected characters.
    """
    text_series = series.fillna("").astype(str)
    return int(text_series.str.contains(UNEXPECTED_CHARACTER_PATTERN, regex=True).sum())


def _encoding_problem_count(series: pd.Series) -> int:
    """Count rows with common mojibake or replacement-character patterns.

    Args:
        series: Text series to inspect.

    Returns:
        Number of rows with possible encoding problems.
    """
    text_series = series.fillna("").astype(str)
    return int(text_series.str.contains(ENCODING_PROBLEM_PATTERN, regex=True).sum())


def _long_text_count(series: pd.Series) -> tuple[int, int]:
    """Count unusually long text rows using a conservative dynamic threshold.

    Args:
        series: Text series to inspect.

    Returns:
        Tuple of long text count and threshold used.
    """
    text_lengths = series.fillna("").astype(str).str.len()
    if text_lengths.empty:
        return 0, LONG_TEXT_MIN_LENGTH

    dynamic_threshold = int(text_lengths.quantile(LONG_TEXT_QUANTILE))
    threshold = max(dynamic_threshold, LONG_TEXT_MIN_LENGTH)
    return int((text_lengths > threshold).sum()), threshold


def _analyze_dataset_quality(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze data quality for one dataset.

    Args:
        dataframe: Dataset to inspect.
        dataset_name: Dataset name.

    Returns:
        Dataset-level and column-level quality tables.
    """
    total_rows = int(dataframe.shape[0])
    duplicate_rows = int(dataframe.duplicated().sum())

    dataset_summary = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "row_count": total_rows,
                "column_count": int(dataframe.shape[1]),
                "duplicate_rows": duplicate_rows,
                "duplicate_percentage": percentage(duplicate_rows, total_rows),
            }
        ]
    )

    column_rows: list[dict[str, object]] = []
    text_columns = set(_text_columns(dataframe))

    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())
        empty_count = (
            _empty_string_count(dataframe[column]) if column in text_columns else 0
        )
        unknown_count = _unknown_count(dataframe[column]) if column in text_columns else 0
        unexpected_count = (
            _unexpected_character_count(dataframe[column])
            if column in text_columns
            else 0
        )
        encoding_count = (
            _encoding_problem_count(dataframe[column]) if column in text_columns else 0
        )
        long_text_count, long_text_threshold = (
            _long_text_count(dataframe[column])
            if column in text_columns
            else (0, 0)
        )

        column_rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "dtype": str(dataframe[column].dtype),
                "missing_count": missing_count,
                "missing_percentage": percentage(missing_count, total_rows),
                "empty_string_count": empty_count,
                "empty_string_percentage": percentage(empty_count, total_rows),
                "unknown_count": unknown_count,
                "unknown_percentage": percentage(unknown_count, total_rows),
                "long_text_count": long_text_count,
                "long_text_threshold": long_text_threshold,
                "unexpected_character_count": unexpected_count,
                "unexpected_character_percentage": percentage(
                    unexpected_count,
                    total_rows,
                ),
                "encoding_problem_count": encoding_count,
                "encoding_problem_percentage": percentage(encoding_count, total_rows),
            }
        )

    return dataset_summary, pd.DataFrame(column_rows)


def _build_quality_comments(
    dataset_summary: pd.DataFrame,
    column_summary: pd.DataFrame,
) -> list[str]:
    """Create short comments for the data quality report.

    Args:
        dataset_summary: Dataset-level quality summary.
        column_summary: Column-level quality summary.

    Returns:
        Human-readable interpretation comments.
    """
    total_duplicate_rows = int(dataset_summary["duplicate_rows"].sum())
    highest_missing_row = column_summary.sort_values(
        "missing_percentage",
        ascending=False,
    ).iloc[0]
    highest_unknown_row = column_summary.sort_values(
        "unknown_percentage",
        ascending=False,
    ).iloc[0]
    total_encoding_problem_rows = int(column_summary["encoding_problem_count"].sum())

    return [
        (
            f"Tüm datasetlerde toplam {total_duplicate_rows} duplicate row "
            "tespit edildi; duplicate oranları veri ayrımı ve değerlendirme "
            "öncesinde takip edilmelidir."
        ),
        (
            f"En yüksek missing oranı {highest_missing_row['dataset']}."
            f"{highest_missing_row['column']} kolonunda "
            f"%{highest_missing_row['missing_percentage']:.2f}."
        ),
        (
            f"En yüksek unknown/boş değer riski {highest_unknown_row['dataset']}."
            f"{highest_unknown_row['column']} kolonunda "
            f"%{highest_unknown_row['unknown_percentage']:.2f}."
        ),
        (
            f"Olası encoding problemi sayısı {total_encoding_problem_rows}; "
            "bu değer yüksekse metin temizliği öncesi örnek kayıtlar manuel "
            "incelenmelidir."
        ),
    ]


def analyze_data_quality(
    datasets: dict[str, pd.DataFrame],
) -> AnalysisResult:
    """Analyze data quality across multiple datasets.

    Args:
        datasets: Mapping from dataset name to DataFrame.

    Returns:
        Analysis result containing quality metrics, tables, and comments.
    """
    logger.info("Starting data quality analysis")

    dataset_tables: list[pd.DataFrame] = []
    column_tables: list[pd.DataFrame] = []

    for dataset_name, dataframe in datasets.items():
        logger.info("Analyzing data quality for %s", dataset_name)
        dataset_summary, column_summary = _analyze_dataset_quality(
            dataframe=dataframe,
            dataset_name=dataset_name,
        )
        dataset_tables.append(dataset_summary)
        column_tables.append(column_summary)

    dataset_summary_table = pd.concat(dataset_tables, ignore_index=True)
    column_summary_table = pd.concat(column_tables, ignore_index=True)

    metrics: dict[str, object] = {
        "dataset_count": len(datasets),
        "total_row_count": int(dataset_summary_table["row_count"].sum()),
        "total_duplicate_rows": int(dataset_summary_table["duplicate_rows"].sum()),
        "total_missing_values": int(column_summary_table["missing_count"].sum()),
        "total_empty_strings": int(column_summary_table["empty_string_count"].sum()),
        "total_unknown_values": int(column_summary_table["unknown_count"].sum()),
        "total_unexpected_character_rows": int(
            column_summary_table["unexpected_character_count"].sum()
        ),
        "total_encoding_problem_rows": int(
            column_summary_table["encoding_problem_count"].sum()
        ),
    }

    tables = {
        "dataset_quality_summary": dataset_summary_table,
        "column_quality_summary": column_summary_table,
    }

    result = AnalysisResult(
        title="Veri Kalitesi",
        metrics=metrics,
        comments=_build_quality_comments(
            dataset_summary=dataset_summary_table,
            column_summary=column_summary_table,
        ),
        tables=tables,
    )
    logger.info("Data quality analysis completed")
    return result


def run_quality_analysis() -> AnalysisResult:
    """Load all raw datasets and run data quality analysis.

    Returns:
        Data quality analysis result.
    """
    datasets = {
        "terms": load_terms(),
        "items": load_items(),
        "training_pairs": load_training_pairs(),
        "submission_pairs": load_submission_pairs(),
        "sample_submission": load_sample_submission(),
    }
    return analyze_data_quality(datasets)


def main() -> None:
    """Run data quality analysis as a command-line entrypoint."""
    result = run_quality_analysis()
    print(result)


if __name__ == "__main__":
    main()
