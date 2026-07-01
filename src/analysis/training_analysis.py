"""Training-pair exploratory analysis for query-item matches."""

import pandas as pd

from src.analysis.common import AnalysisResult, percentage
from src.data.data_loader import load_training_pairs
from src.utils.logger import get_logger


logger = get_logger(__name__)

ID_COLUMN: str = "id"
TERM_ID_COLUMN: str = "term_id"
ITEM_ID_COLUMN: str = "item_id"
LABEL_COLUMN: str = "label"
TOP_ENTITY_LIMIT: int = 20


def _validate_training_columns(training_pairs: pd.DataFrame) -> None:
    """Validate that the training pairs dataset has required columns.

    Args:
        training_pairs: Training pairs dataset.

    Raises:
        ValueError: If a required column is missing.
    """
    required_columns = {ID_COLUMN, TERM_ID_COLUMN, ITEM_ID_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(training_pairs.columns)

    if missing_columns:
        message = f"Missing required training pair columns: {sorted(missing_columns)}"
        logger.error(message)
        raise ValueError(message)


def _build_group_size_distribution(
    group_sizes: pd.Series,
    group_size_column: str,
) -> pd.DataFrame:
    """Build a distribution table from entity group sizes.

    Args:
        group_sizes: Number of linked entities per group.
        group_size_column: Name of the output group-size column.

    Returns:
        Distribution table with group size, count, and percentage.
    """
    total_group_count = int(group_sizes.shape[0])
    table = (
        group_sizes.value_counts()
        .sort_index()
        .rename_axis(group_size_column)
        .reset_index(name="entity_count")
    )
    table["percentage"] = table["entity_count"].apply(
        lambda value: percentage(value, total_group_count)
    )
    return table


def _build_top_entities(
    group_sizes: pd.Series,
    entity_column: str,
    linked_count_column: str,
) -> pd.DataFrame:
    """Build a top-N table for popular queries or products.

    Args:
        group_sizes: Number of linked entities per query or product.
        entity_column: Name for the entity id column.
        linked_count_column: Name for the count column.

    Returns:
        Top entity table.
    """
    return (
        group_sizes.sort_values(ascending=False)
        .head(TOP_ENTITY_LIMIT)
        .rename_axis(entity_column)
        .reset_index(name=linked_count_column)
    )


def _build_label_distribution(training_pairs: pd.DataFrame) -> pd.DataFrame:
    """Build the label distribution table.

    Args:
        training_pairs: Training pairs dataset.

    Returns:
        Label distribution table.
    """
    total_pair_count = int(training_pairs.shape[0])
    table = (
        training_pairs[LABEL_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("label")
        .reset_index(name="count")
    )
    table["percentage"] = table["count"].apply(
        lambda value: percentage(value, total_pair_count)
    )
    return table


def _build_training_comments(metrics: dict[str, object]) -> list[str]:
    """Create short comments for the training-pair analysis report.

    Args:
        metrics: Training-pair analysis metrics.

    Returns:
        Human-readable interpretation comments.
    """
    return [
        (
            f"Training setinde {metrics['total_pair_count']} pair var; bu sayı "
            "pozitif eşleşme grafiğinin toplam gözlem hacmini gösterir."
        ),
        (
            f"Query başına ortalama ürün sayısı "
            f"{metrics['average_items_per_query']:.2f}; query'lerin katalogda "
            "ne kadar geniş ürün alanına bağlandığını anlamamızı sağlar."
        ),
        (
            f"Ürün başına ortalama query sayısı "
            f"{metrics['average_queries_per_item']:.2f}; popüler ürünlerin "
            "birden fazla arama niyetiyle eşleşip eşleşmediğini gösterir."
        ),
        (
            f"Pozitif label oranı %{metrics['positive_label_ratio']:.2f}; "
            "etiket dağılımı modelleme öncesi değerlendirme stratejisini "
            "doğrudan etkiler."
        ),
    ]


def analyze_training_pairs(training_pairs: pd.DataFrame) -> AnalysisResult:
    """Analyze the training query-item pair dataset.

    Args:
        training_pairs: Training pairs dataset.

    Returns:
        Analysis result containing pair metrics, tables, and comments.
    """
    logger.info("Starting training pair analysis")
    _validate_training_columns(training_pairs)

    total_pair_count = int(training_pairs.shape[0])
    unique_query_count = int(training_pairs[TERM_ID_COLUMN].nunique())
    unique_item_count = int(training_pairs[ITEM_ID_COLUMN].nunique())

    items_per_query = training_pairs.groupby(TERM_ID_COLUMN)[ITEM_ID_COLUMN].nunique()
    queries_per_item = training_pairs.groupby(ITEM_ID_COLUMN)[TERM_ID_COLUMN].nunique()
    label_distribution = _build_label_distribution(training_pairs)

    positive_label_count = int((training_pairs[LABEL_COLUMN] == 1).sum())
    negative_label_count = int((training_pairs[LABEL_COLUMN] == 0).sum())

    metrics: dict[str, object] = {
        "total_pair_count": total_pair_count,
        "unique_query_count": unique_query_count,
        "unique_item_count": unique_item_count,
        "average_items_per_query": float(items_per_query.mean())
        if unique_query_count
        else 0.0,
        "maximum_items_per_query": int(items_per_query.max())
        if unique_query_count
        else 0,
        "minimum_items_per_query": int(items_per_query.min())
        if unique_query_count
        else 0,
        "average_queries_per_item": float(queries_per_item.mean())
        if unique_item_count
        else 0.0,
        "maximum_queries_per_item": int(queries_per_item.max())
        if unique_item_count
        else 0,
        "minimum_queries_per_item": int(queries_per_item.min())
        if unique_item_count
        else 0,
        "positive_label_count": positive_label_count,
        "negative_label_count": negative_label_count,
        "positive_label_ratio": percentage(positive_label_count, total_pair_count),
        "negative_label_ratio": percentage(negative_label_count, total_pair_count),
    }

    tables = {
        "items_per_query_distribution": _build_group_size_distribution(
            items_per_query,
            group_size_column="item_count",
        ),
        "queries_per_item_distribution": _build_group_size_distribution(
            queries_per_item,
            group_size_column="query_count",
        ),
        "top_20_queries": _build_top_entities(
            items_per_query,
            entity_column="term_id",
            linked_count_column="matched_item_count",
        ),
        "top_20_items": _build_top_entities(
            queries_per_item,
            entity_column="item_id",
            linked_count_column="matched_query_count",
        ),
        "label_distribution": label_distribution,
    }

    result = AnalysisResult(
        title="Training Pair Analizi",
        metrics=metrics,
        comments=_build_training_comments(metrics),
        tables=tables,
    )
    logger.info("Training pair analysis completed")
    return result


def run_training_analysis() -> AnalysisResult:
    """Load training pairs data and run pair analysis.

    Returns:
        Training-pair analysis result.
    """
    training_pairs = load_training_pairs()
    return analyze_training_pairs(training_pairs)


def main() -> None:
    """Run training pair analysis as a command-line entrypoint."""
    result = run_training_analysis()
    print(result)


if __name__ == "__main__":
    main()
