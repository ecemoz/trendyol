"""Shared helpers for Sprint 2 exploratory data analysis modules."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


UNKNOWN_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "unknown",
        "unk",
        "none",
        "null",
        "nan",
        "na",
        "n/a",
        "belirsiz",
        "bilinmiyor",
        "yok",
        "diğer",
        "diger",
    }
)
TURKISH_CHARACTERS: frozenset[str] = frozenset("çğıöşüÇĞİÖŞÜ")


@dataclass(frozen=True)
class AnalysisResult:
    """Container for metrics, narrative comments, and optional detail tables.

    Attributes:
        title: Human-readable analysis section title.
        metrics: Scalar metrics that summarize the analysis.
        comments: Short interpretations written after the metrics.
        tables: Optional tabular outputs such as top-N distributions.
    """

    title: str
    metrics: dict[str, Any]
    comments: list[str] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Calculate a ratio while protecting against division by zero.

    Args:
        numerator: Value in the numerator.
        denominator: Value in the denominator.

    Returns:
        Ratio as a float. Returns 0.0 when denominator is zero.
    """
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def percentage(numerator: int | float, denominator: int | float) -> float:
    """Calculate a percentage while protecting against division by zero.

    Args:
        numerator: Value in the numerator.
        denominator: Value in the denominator.

    Returns:
        Percentage value in the 0-100 range.
    """
    return safe_ratio(numerator, denominator) * 100.0


def normalize_text(value: object) -> str:
    """Normalize a value into a stripped lowercase text representation.

    Args:
        value: Any raw cell value.

    Returns:
        Normalized text. Missing values become an empty string.
    """
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def is_unknown_like(value: object) -> bool:
    """Check whether a value represents an unknown or empty category.

    Args:
        value: Any raw cell value.

    Returns:
        True when the value is missing, empty, or one of the known placeholders.
    """
    return normalize_text(value) in UNKNOWN_TOKENS


def count_words(text: object) -> int:
    """Count whitespace-separated words in a text value.

    Args:
        text: Raw text value.

    Returns:
        Number of non-empty whitespace-separated tokens.
    """
    normalized_text = normalize_text(text)
    if not normalized_text:
        return 0
    return len(normalized_text.split())


def contains_turkish_character(text: object) -> bool:
    """Check whether a text contains at least one Turkish-specific character.

    Args:
        text: Raw text value.

    Returns:
        True when the text contains Turkish-specific characters.
    """
    return any(character in TURKISH_CHARACTERS for character in str(text))


def build_value_counts_table(
    series: pd.Series,
    value_column: str,
    count_column: str = "count",
    top_n: int | None = None,
    include_percentage: bool = True,
) -> pd.DataFrame:
    """Build a reusable value-counts table.

    Args:
        series: Series to summarize.
        value_column: Name of the output column containing unique values.
        count_column: Name of the output column containing counts.
        top_n: Optional number of rows to keep.
        include_percentage: Whether to include a percentage column.

    Returns:
        DataFrame with values, counts, and optional percentages.
    """
    counts = series.value_counts(dropna=False)
    if top_n is not None:
        counts = counts.head(top_n)

    table = counts.rename_axis(value_column).reset_index(name=count_column)
    if include_percentage:
        total_count = int(series.shape[0])
        table["percentage"] = table[count_column].apply(
            lambda value: percentage(value, total_count)
        )

    return table
