"""Validation and false-negative protection for generated negative samples."""

from dataclasses import dataclass

import pandas as pd

from src.analysis.common import normalize_text
from src.config.negative_sampling import (
    DROP_HIGH_RISK_FALSE_NEGATIVES,
    MAX_LEXICAL_SIMILARITY_WARNING_THRESHOLD,
)
from src.negative_sampling.sampler import (
    ITEM_ID_COLUMN,
    NORMALIZED_QUERY_COLUMN,
    NORMALIZED_TITLE_COLUMN,
    TERM_ID_COLUMN,
    TITLE_COLUMN,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

LEXICAL_SIMILARITY_COLUMN: str = "lexical_similarity"
HIGH_RISK_FALSE_NEGATIVE_COLUMN: str = "high_risk_false_negative"
VALIDATION_REASON_COLUMN: str = "validation_reason"
POSITIVE_ITEM_ID_COLUMN: str = "positive_item_id"


@dataclass(frozen=True)
class ValidationReport:
    """Summary of negative sample validation.

    Attributes:
        input_count: Number of negative rows received by the validator.
        output_count: Number of negative rows returned by the validator.
        known_positive_pair_count: Count removed because pair is known positive.
        same_item_count: Count removed because negative item equals positive item.
        exact_query_title_match_count: Count removed because query equals title.
        high_risk_similarity_count: Count flagged as high-risk by lexical similarity.
        dropped_high_risk_count: Count dropped due to high-risk similarity policy.
    """

    input_count: int
    output_count: int
    known_positive_pair_count: int
    same_item_count: int
    exact_query_title_match_count: int
    high_risk_similarity_count: int
    dropped_high_risk_count: int


def lexical_jaccard_similarity(left_text: object, right_text: object) -> float:
    """Calculate token-level Jaccard similarity between two texts.

    Args:
        left_text: First raw text.
        right_text: Second raw text.

    Returns:
        Jaccard similarity in the 0-1 range.
    """
    left_tokens = set(normalize_text(left_text).split())
    right_tokens = set(normalize_text(right_text).split())

    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0

    return len(left_tokens.intersection(right_tokens)) / len(
        left_tokens.union(right_tokens)
    )


def _positive_pair_mask(
    negatives: pd.DataFrame,
    positive_pair_set: set[tuple[str, str]],
) -> pd.Series:
    """Find rows whose query-item pair is known positive.

    Args:
        negatives: Generated negative samples.
        positive_pair_set: Known positive ``(term_id, item_id)`` pairs.

    Returns:
        Boolean mask for known positive pairs.
    """
    candidate_pairs = zip(
        negatives[TERM_ID_COLUMN].astype(str),
        negatives[ITEM_ID_COLUMN].astype(str),
        strict=False,
    )
    return pd.Series(
        [pair in positive_pair_set for pair in candidate_pairs],
        index=negatives.index,
    )


def _same_item_mask(negatives: pd.DataFrame) -> pd.Series:
    """Find rows where the sampled negative item equals the positive anchor item.

    Args:
        negatives: Generated negative samples.

    Returns:
        Boolean mask for same-item rows.
    """
    if POSITIVE_ITEM_ID_COLUMN not in negatives.columns:
        return pd.Series(False, index=negatives.index)

    return (
        negatives[ITEM_ID_COLUMN].astype(str)
        == negatives[POSITIVE_ITEM_ID_COLUMN].astype(str)
    )


def _exact_query_title_match_mask(negatives: pd.DataFrame) -> pd.Series:
    """Find rows where query text and sampled item title are exactly equal.

    Args:
        negatives: Generated negative samples.

    Returns:
        Boolean mask for exact query-title matches.
    """
    query_text = negatives[NORMALIZED_QUERY_COLUMN].fillna("").astype(str)
    if NORMALIZED_TITLE_COLUMN in negatives.columns:
        title_text = negatives[NORMALIZED_TITLE_COLUMN].fillna("").astype(str)
    else:
        title_text = negatives[TITLE_COLUMN].map(normalize_text)

    return query_text.eq(title_text)


def _append_validation_reason(
    existing_reason: pd.Series,
    mask: pd.Series,
    reason: str,
) -> pd.Series:
    """Append a reason label to rows matching a validation mask.

    Args:
        existing_reason: Current validation reason text per row.
        mask: Rows to update.
        reason: Reason label to append.

    Returns:
        Updated validation reason series.
    """
    updated_reason = existing_reason.copy()
    current_values = updated_reason.loc[mask].fillna("").astype(str)
    updated_reason.loc[mask] = current_values.apply(
        lambda value: reason if not value else f"{value};{reason}"
    )
    return updated_reason


def validate_negative_samples(
    negatives: pd.DataFrame,
    positive_pair_set: set[tuple[str, str]],
    similarity_threshold: float = MAX_LEXICAL_SIMILARITY_WARNING_THRESHOLD,
    drop_high_risk: bool = DROP_HIGH_RISK_FALSE_NEGATIVES,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate negative samples and remove likely false negatives.

    Args:
        negatives: Generated negative samples.
        positive_pair_set: Known positive ``(term_id, item_id)`` pairs.
        similarity_threshold: Lexical similarity threshold for high-risk rows.
        drop_high_risk: Whether high-risk lexical rows should be removed.

    Returns:
        Tuple of validated negative samples and validation report.
    """
    logger.info("Starting negative sample validation")
    input_count = int(negatives.shape[0])
    if negatives.empty:
        report = ValidationReport(
            input_count=0,
            output_count=0,
            known_positive_pair_count=0,
            same_item_count=0,
            exact_query_title_match_count=0,
            high_risk_similarity_count=0,
            dropped_high_risk_count=0,
        )
        return negatives.copy(), report

    validated = negatives.copy()
    validated[VALIDATION_REASON_COLUMN] = ""

    known_positive_mask = _positive_pair_mask(validated, positive_pair_set)
    same_item_mask = _same_item_mask(validated)
    exact_query_title_mask = _exact_query_title_match_mask(validated)

    validated[VALIDATION_REASON_COLUMN] = _append_validation_reason(
        validated[VALIDATION_REASON_COLUMN],
        known_positive_mask,
        "known_positive_pair",
    )
    validated[VALIDATION_REASON_COLUMN] = _append_validation_reason(
        validated[VALIDATION_REASON_COLUMN],
        same_item_mask,
        "same_item",
    )
    validated[VALIDATION_REASON_COLUMN] = _append_validation_reason(
        validated[VALIDATION_REASON_COLUMN],
        exact_query_title_mask,
        "exact_query_title_match",
    )

    lexical_similarity = [
        lexical_jaccard_similarity(query, title)
        for query, title in zip(
            validated[NORMALIZED_QUERY_COLUMN],
            validated[NORMALIZED_TITLE_COLUMN],
            strict=False,
        )
    ]
    validated[LEXICAL_SIMILARITY_COLUMN] = lexical_similarity
    high_risk_mask = validated[LEXICAL_SIMILARITY_COLUMN] >= similarity_threshold
    validated[HIGH_RISK_FALSE_NEGATIVE_COLUMN] = high_risk_mask
    validated[VALIDATION_REASON_COLUMN] = _append_validation_reason(
        validated[VALIDATION_REASON_COLUMN],
        high_risk_mask,
        "high_lexical_similarity",
    )

    hard_drop_mask = known_positive_mask | same_item_mask | exact_query_title_mask
    if drop_high_risk:
        drop_mask = hard_drop_mask | high_risk_mask
    else:
        drop_mask = hard_drop_mask

    output = validated.loc[~drop_mask].reset_index(drop=True)
    report = ValidationReport(
        input_count=input_count,
        output_count=int(output.shape[0]),
        known_positive_pair_count=int(known_positive_mask.sum()),
        same_item_count=int(same_item_mask.sum()),
        exact_query_title_match_count=int(exact_query_title_mask.sum()),
        high_risk_similarity_count=int(high_risk_mask.sum()),
        dropped_high_risk_count=int(high_risk_mask.sum()) if drop_high_risk else 0,
    )

    logger.info(
        "Negative validation completed | input=%s | output=%s | known_positive=%s | "
        "same_item=%s | exact_query_title=%s | high_risk=%s",
        report.input_count,
        report.output_count,
        report.known_positive_pair_count,
        report.same_item_count,
        report.exact_query_title_match_count,
        report.high_risk_similarity_count,
    )
    return output, report
