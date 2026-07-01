"""Query-level exploratory data analysis for the terms dataset."""

from collections import Counter
import re

import pandas as pd

from src.analysis.common import (
    AnalysisResult,
    contains_turkish_character,
    count_words,
    normalize_text,
    percentage,
)
from src.data.data_loader import load_terms
from src.utils.logger import get_logger


logger = get_logger(__name__)

QUERY_COLUMN: str = "query"
TERM_ID_COLUMN: str = "term_id"
TOP_WORD_LIMIT: int = 30
LENGTH_BIN_COUNT: int = 20
WORD_PATTERN: re.Pattern[str] = re.compile(
    r"[0-9a-zA-ZçğıöşüÇĞİÖŞÜ]+",
    flags=re.UNICODE,
)


def _validate_terms_columns(terms: pd.DataFrame) -> None:
    """Validate that the terms dataset has the columns required for analysis.

    Args:
        terms: Terms dataset.

    Raises:
        ValueError: If a required column is missing.
    """
    required_columns = {TERM_ID_COLUMN, QUERY_COLUMN}
    missing_columns = required_columns.difference(terms.columns)

    if missing_columns:
        message = f"Missing required terms columns: {sorted(missing_columns)}"
        logger.error(message)
        raise ValueError(message)


def _tokenize_query(query: object) -> list[str]:
    """Tokenize a query for frequency analysis.

    Args:
        query: Raw query value.

    Returns:
        Lowercase alphanumeric tokens, including Turkish characters.
    """
    normalized_query = normalize_text(query)
    return WORD_PATTERN.findall(normalized_query)


def _build_length_histogram(query_lengths: pd.Series) -> pd.DataFrame:
    """Build a character-length histogram table.

    Args:
        query_lengths: Character length per query.

    Returns:
        Histogram table with bin intervals and counts.
    """
    if query_lengths.empty:
        return pd.DataFrame(columns=["length_bin", "count"])

    binned_lengths = pd.cut(
        query_lengths,
        bins=min(LENGTH_BIN_COUNT, max(int(query_lengths.nunique()), 1)),
        include_lowest=True,
        duplicates="drop",
    )
    return (
        binned_lengths.value_counts(sort=False)
        .rename_axis("length_bin")
        .reset_index(name="count")
    )


def _build_word_count_histogram(word_counts: pd.Series) -> pd.DataFrame:
    """Build a word-count histogram table.

    Args:
        word_counts: Word count per query.

    Returns:
        Histogram table with word counts and row counts.
    """
    return (
        word_counts.value_counts()
        .sort_index()
        .rename_axis("word_count")
        .reset_index(name="count")
    )


def _build_top_words_table(queries: pd.Series) -> pd.DataFrame:
    """Build a table of the most frequent query words.

    Args:
        queries: Query text series.

    Returns:
        Top word-frequency table.
    """
    word_counter: Counter[str] = Counter()
    for query in queries:
        word_counter.update(_tokenize_query(query))

    return pd.DataFrame(
        word_counter.most_common(TOP_WORD_LIMIT),
        columns=["word", "count"],
    )


def _build_query_comments(metrics: dict[str, object]) -> list[str]:
    """Create short comments for the query analysis report.

    Args:
        metrics: Query analysis metrics.

    Returns:
        Human-readable interpretation comments.
    """
    unique_ratio = metrics["unique_query_ratio"]
    average_word_count = metrics["average_word_count"]
    single_word_ratio = metrics["single_word_query_ratio"]
    numeric_ratio = metrics["numeric_query_ratio"]
    turkish_character_ratio = metrics["turkish_character_query_ratio"]

    comments = [
        (
            f"Query benzersizlik oranı %{unique_ratio:.2f}; bu değer kullanıcı "
            "arama alanının tekrar edip etmediğini gösterir."
        ),
        (
            f"Ortalama kelime sayısı {average_word_count:.2f}; query'ler kısa "
            "ise lexical eşleşme sinyalleri daha kritik hale gelir."
        ),
        (
            f"Tek kelimelik query oranı %{single_word_ratio:.2f}; bu oran "
            "yüksekse marka, kategori ve attribute bilgisi ayrıştırıcı olabilir."
        ),
        (
            f"Sayı içeren query oranı %{numeric_ratio:.2f}; beden, model, seri "
            "ve ölçü benzeri niyetlerin varlığına işaret edebilir."
        ),
        (
            f"Türkçe karakter içeren query oranı %{turkish_character_ratio:.2f}; "
            "metin normalizasyonu kararlarında Türkçe karakter davranışı "
            "özellikle korunmalıdır."
        ),
    ]
    return comments


def analyze_queries(terms: pd.DataFrame) -> AnalysisResult:
    """Analyze the terms dataset without changing the source data.

    Args:
        terms: Terms dataset with ``term_id`` and ``query`` columns.

    Returns:
        Analysis result containing metrics, tables, and comments.
    """
    logger.info("Starting query analysis")
    _validate_terms_columns(terms)

    queries = terms[QUERY_COLUMN].fillna("").astype(str)
    normalized_queries = queries.map(normalize_text)
    query_lengths = normalized_queries.str.len()
    word_counts = normalized_queries.map(count_words)

    total_query_count = int(terms.shape[0])
    unique_query_count = int(normalized_queries.nunique(dropna=True))
    shortest_query_index = query_lengths.idxmin() if total_query_count else None
    longest_query_index = query_lengths.idxmax() if total_query_count else None

    single_word_query_count = int((word_counts == 1).sum())
    multi_word_query_count = int((word_counts > 1).sum())
    turkish_character_query_count = int(
        queries.map(contains_turkish_character).sum()
    )
    numeric_query_count = int(queries.str.contains(r"\d", regex=True).sum())

    metrics: dict[str, object] = {
        "total_query_count": total_query_count,
        "unique_query_count": unique_query_count,
        "unique_query_ratio": percentage(unique_query_count, total_query_count),
        "average_character_length": float(query_lengths.mean())
        if total_query_count
        else 0.0,
        "average_word_count": float(word_counts.mean()) if total_query_count else 0.0,
        "shortest_query": queries.loc[shortest_query_index]
        if shortest_query_index is not None
        else "",
        "shortest_query_length": int(query_lengths.min()) if total_query_count else 0,
        "longest_query": queries.loc[longest_query_index]
        if longest_query_index is not None
        else "",
        "longest_query_length": int(query_lengths.max()) if total_query_count else 0,
        "turkish_character_query_count": turkish_character_query_count,
        "turkish_character_query_ratio": percentage(
            turkish_character_query_count,
            total_query_count,
        ),
        "numeric_query_count": numeric_query_count,
        "numeric_query_ratio": percentage(numeric_query_count, total_query_count),
        "single_word_query_count": single_word_query_count,
        "single_word_query_ratio": percentage(
            single_word_query_count,
            total_query_count,
        ),
        "multi_word_query_count": multi_word_query_count,
        "multi_word_query_ratio": percentage(
            multi_word_query_count,
            total_query_count,
        ),
    }

    tables = {
        "query_length_histogram": _build_length_histogram(query_lengths),
        "query_word_count_histogram": _build_word_count_histogram(word_counts),
        "top_words": _build_top_words_table(queries),
    }

    result = AnalysisResult(
        title="Query Analizi",
        metrics=metrics,
        comments=_build_query_comments(metrics),
        tables=tables,
    )
    logger.info("Query analysis completed")
    return result


def run_query_analysis() -> AnalysisResult:
    """Load terms data and run query analysis.

    Returns:
        Query analysis result.
    """
    terms = load_terms()
    return analyze_queries(terms)


def main() -> None:
    """Run query analysis as a command-line entrypoint."""
    result = run_query_analysis()
    print(result)


if __name__ == "__main__":
    main()
