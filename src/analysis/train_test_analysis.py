"""Train-test overlap analysis for query-item pair datasets."""

import pandas as pd

from src.analysis.common import AnalysisResult, percentage
from src.data.data_loader import load_submission_pairs, load_training_pairs
from src.utils.logger import get_logger


logger = get_logger(__name__)

TERM_ID_COLUMN: str = "term_id"
ITEM_ID_COLUMN: str = "item_id"


def _validate_pair_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Validate pair dataset columns needed for overlap analysis.

    Args:
        dataframe: Pair dataset.
        dataset_name: Human-readable dataset name for errors.

    Raises:
        ValueError: If a required column is missing.
    """
    required_columns = {TERM_ID_COLUMN, ITEM_ID_COLUMN}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        message = (
            f"Missing required {dataset_name} columns: {sorted(missing_columns)}"
        )
        logger.error(message)
        raise ValueError(message)


def _build_overlap_table(
    entity_name: str,
    train_ids: set[str],
    test_ids: set[str],
) -> pd.DataFrame:
    """Build a one-row overlap summary for query or item ids.

    Args:
        entity_name: Entity label such as ``query`` or ``item``.
        train_ids: Unique ids observed in training pairs.
        test_ids: Unique ids observed in submission pairs.

    Returns:
        DataFrame with overlap and cold-start counts.
    """
    overlap_count = len(test_ids.intersection(train_ids))
    new_count = len(test_ids.difference(train_ids))
    train_only_count = len(train_ids.difference(test_ids))

    return pd.DataFrame(
        [
            {
                "entity": entity_name,
                "train_unique_count": len(train_ids),
                "test_unique_count": len(test_ids),
                "test_seen_in_train_count": overlap_count,
                "test_seen_in_train_percentage": percentage(
                    overlap_count,
                    len(test_ids),
                ),
                "test_new_count": new_count,
                "test_new_percentage": percentage(new_count, len(test_ids)),
                "train_only_count": train_only_count,
            }
        ]
    )


def _build_train_test_comments(metrics: dict[str, object]) -> list[str]:
    """Create short comments for the train-test analysis report.

    Args:
        metrics: Train-test overlap metrics.

    Returns:
        Human-readable interpretation comments.
    """
    return [
        (
            f"Test query'lerinin %{metrics['test_query_seen_in_train_ratio']:.2f} "
            "kadarı train'de görülmüş; bu oran query cold-start riskini gösterir."
        ),
        (
            f"Test item'larının %{metrics['test_item_seen_in_train_ratio']:.2f} "
            "kadarı train'de görülmüş; ürün cold-start durumu model seçimini "
            "etkileyebilir."
        ),
        (
            f"Tam pair overlap oranı %{metrics['exact_pair_overlap_ratio']:.2f}; "
            "train'deki aynı query-item eşleşmelerinin test adaylarında ne kadar "
            "tekrarlandığını gösterir."
        ),
        (
            "Train/test overlap bilgisi ileride ezberleme riski, cold-start "
            "dayanıklılığı ve aday sıralama stratejisi için temel karar girdisidir."
        ),
    ]


def analyze_train_test_overlap(
    training_pairs: pd.DataFrame,
    submission_pairs: pd.DataFrame,
) -> AnalysisResult:
    """Analyze overlap between training and submission pair datasets.

    Args:
        training_pairs: Training query-item pair dataset.
        submission_pairs: Submission candidate pair dataset.

    Returns:
        Analysis result containing overlap metrics, tables, and comments.
    """
    logger.info("Starting train-test overlap analysis")
    _validate_pair_columns(training_pairs, "training_pairs")
    _validate_pair_columns(submission_pairs, "submission_pairs")

    train_query_ids = set(training_pairs[TERM_ID_COLUMN].dropna().astype(str))
    test_query_ids = set(submission_pairs[TERM_ID_COLUMN].dropna().astype(str))
    train_item_ids = set(training_pairs[ITEM_ID_COLUMN].dropna().astype(str))
    test_item_ids = set(submission_pairs[ITEM_ID_COLUMN].dropna().astype(str))

    train_pairs = set(
        zip(
            training_pairs[TERM_ID_COLUMN].astype(str),
            training_pairs[ITEM_ID_COLUMN].astype(str),
            strict=False,
        )
    )
    test_pairs = set(
        zip(
            submission_pairs[TERM_ID_COLUMN].astype(str),
            submission_pairs[ITEM_ID_COLUMN].astype(str),
            strict=False,
        )
    )

    test_query_seen_count = len(test_query_ids.intersection(train_query_ids))
    test_item_seen_count = len(test_item_ids.intersection(train_item_ids))
    exact_pair_overlap_count = len(test_pairs.intersection(train_pairs))

    metrics: dict[str, object] = {
        "train_unique_query_count": len(train_query_ids),
        "test_unique_query_count": len(test_query_ids),
        "test_query_seen_in_train_count": test_query_seen_count,
        "test_query_seen_in_train_ratio": percentage(
            test_query_seen_count,
            len(test_query_ids),
        ),
        "new_test_query_count": len(test_query_ids.difference(train_query_ids)),
        "train_unique_item_count": len(train_item_ids),
        "test_unique_item_count": len(test_item_ids),
        "test_item_seen_in_train_count": test_item_seen_count,
        "test_item_seen_in_train_ratio": percentage(
            test_item_seen_count,
            len(test_item_ids),
        ),
        "new_test_item_count": len(test_item_ids.difference(train_item_ids)),
        "train_unique_pair_count": len(train_pairs),
        "test_unique_pair_count": len(test_pairs),
        "exact_pair_overlap_count": exact_pair_overlap_count,
        "exact_pair_overlap_ratio": percentage(
            exact_pair_overlap_count,
            len(test_pairs),
        ),
    }

    tables = {
        "query_overlap": _build_overlap_table(
            entity_name="query",
            train_ids=train_query_ids,
            test_ids=test_query_ids,
        ),
        "item_overlap": _build_overlap_table(
            entity_name="item",
            train_ids=train_item_ids,
            test_ids=test_item_ids,
        ),
    }

    result = AnalysisResult(
        title="Train/Test Analizi",
        metrics=metrics,
        comments=_build_train_test_comments(metrics),
        tables=tables,
    )
    logger.info("Train-test overlap analysis completed")
    return result


def run_train_test_analysis() -> AnalysisResult:
    """Load pair datasets and run train-test overlap analysis.

    Returns:
        Train-test overlap analysis result.
    """
    training_pairs = load_training_pairs()
    submission_pairs = load_submission_pairs()
    return analyze_train_test_overlap(training_pairs, submission_pairs)


def main() -> None:
    """Run train-test overlap analysis as a command-line entrypoint."""
    result = run_train_test_analysis()
    print(result)


if __name__ == "__main__":
    main()
