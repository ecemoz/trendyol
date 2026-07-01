"""Shared primitives for negative sampling strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from src.analysis.common import normalize_text
from src.config.negative_sampling import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RANDOM_SEED,
    SAMPLE_TYPE_POSITIVE,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

ID_COLUMN: str = "id"
TERM_ID_COLUMN: str = "term_id"
ITEM_ID_COLUMN: str = "item_id"
QUERY_COLUMN: str = "query"
TITLE_COLUMN: str = "title"
CATEGORY_COLUMN: str = "category"
BRAND_COLUMN: str = "brand"
ATTRIBUTES_COLUMN: str = "attributes"
LABEL_COLUMN: str = "label"
SAMPLE_TYPE_COLUMN: str = "sample_type"
MAIN_CATEGORY_COLUMN: str = "main_category"
SUB_CATEGORY_COLUMN: str = "sub_category"
NORMALIZED_TITLE_COLUMN: str = "normalized_title"
NORMALIZED_QUERY_COLUMN: str = "normalized_query"

OUTPUT_COLUMNS: list[str] = [
    QUERY_COLUMN,
    ITEM_ID_COLUMN,
    TITLE_COLUMN,
    CATEGORY_COLUMN,
    BRAND_COLUMN,
    ATTRIBUTES_COLUMN,
    LABEL_COLUMN,
    SAMPLE_TYPE_COLUMN,
]

ITEM_FEATURE_COLUMNS: list[str] = [
    ITEM_ID_COLUMN,
    TITLE_COLUMN,
    CATEGORY_COLUMN,
    BRAND_COLUMN,
    ATTRIBUTES_COLUMN,
    MAIN_CATEGORY_COLUMN,
    SUB_CATEGORY_COLUMN,
    NORMALIZED_TITLE_COLUMN,
]


@dataclass(frozen=True)
class SamplingResult:
    """Container for samples and sampling metadata.

    Attributes:
        samples: Generated sample rows.
        requested_count: Number of negatives requested from the sampler.
        generated_count: Number of rows generated before downstream validation.
        elapsed_seconds: Runtime of the sampler.
        sample_type: Negative sample type.
    """

    samples: pd.DataFrame
    requested_count: int
    generated_count: int
    elapsed_seconds: float
    sample_type: str


def split_category(category: object) -> list[str]:
    """Split a category path into normalized hierarchy levels.

    Args:
        category: Raw category path.

    Returns:
        Ordered category hierarchy levels.
    """
    normalized_category = normalize_text(category)
    if not normalized_category:
        return []

    return [level.strip() for level in normalized_category.split("/") if level.strip()]


def main_category(category: object) -> str:
    """Extract the main category from a category path.

    Args:
        category: Raw category path.

    Returns:
        Main category, or an empty string when unavailable.
    """
    category_levels = split_category(category)
    return category_levels[0] if category_levels else ""


def sub_category(category: object) -> str:
    """Extract the first subcategory from a category path.

    Args:
        category: Raw category path.

    Returns:
        First subcategory, or an empty string when unavailable.
    """
    category_levels = split_category(category)
    return category_levels[1] if len(category_levels) > 1 else ""


def prepare_items_for_sampling(items: pd.DataFrame) -> pd.DataFrame:
    """Prepare item metadata used by all negative samplers.

    Args:
        items: Raw items dataset.

    Returns:
        Item metadata with category levels and normalized title.

    Raises:
        ValueError: If required item columns are missing.
    """
    required_columns = {
        ITEM_ID_COLUMN,
        TITLE_COLUMN,
        CATEGORY_COLUMN,
        BRAND_COLUMN,
        ATTRIBUTES_COLUMN,
    }
    missing_columns = required_columns.difference(items.columns)
    if missing_columns:
        raise ValueError(f"Missing required item columns: {sorted(missing_columns)}")

    logger.info("Preparing item metadata for negative sampling")
    prepared_items = items[
        [ITEM_ID_COLUMN, TITLE_COLUMN, CATEGORY_COLUMN, BRAND_COLUMN, ATTRIBUTES_COLUMN]
    ].copy()
    prepared_items[TITLE_COLUMN] = prepared_items[TITLE_COLUMN].fillna("").astype(str)
    prepared_items[CATEGORY_COLUMN] = (
        prepared_items[CATEGORY_COLUMN].fillna("").astype(str)
    )
    prepared_items[BRAND_COLUMN] = prepared_items[BRAND_COLUMN].fillna("").astype(str)
    prepared_items[ATTRIBUTES_COLUMN] = (
        prepared_items[ATTRIBUTES_COLUMN].fillna("").astype(str)
    )
    prepared_items[MAIN_CATEGORY_COLUMN] = prepared_items[CATEGORY_COLUMN].map(
        main_category
    )
    prepared_items[SUB_CATEGORY_COLUMN] = prepared_items[CATEGORY_COLUMN].map(
        sub_category
    )
    prepared_items[NORMALIZED_TITLE_COLUMN] = prepared_items[TITLE_COLUMN].map(
        normalize_text
    )
    return prepared_items


def prepare_positive_samples(
    training_pairs: pd.DataFrame,
    terms: pd.DataFrame,
    prepared_items: pd.DataFrame,
) -> pd.DataFrame:
    """Build positive training samples by joining pairs, terms, and items.

    Args:
        training_pairs: Raw positive training pairs.
        terms: Raw terms dataset.
        prepared_items: Prepared item metadata.

    Returns:
        Positive samples with output and helper columns.

    Raises:
        ValueError: If required columns are missing.
    """
    logger.info("Preparing positive samples")
    required_pair_columns = {TERM_ID_COLUMN, ITEM_ID_COLUMN, LABEL_COLUMN}
    required_term_columns = {TERM_ID_COLUMN, QUERY_COLUMN}

    missing_pair_columns = required_pair_columns.difference(training_pairs.columns)
    missing_term_columns = required_term_columns.difference(terms.columns)
    if missing_pair_columns:
        raise ValueError(
            f"Missing required training pair columns: {sorted(missing_pair_columns)}"
        )
    if missing_term_columns:
        raise ValueError(f"Missing required terms columns: {sorted(missing_term_columns)}")

    positives = training_pairs[[TERM_ID_COLUMN, ITEM_ID_COLUMN, LABEL_COLUMN]].merge(
        terms[[TERM_ID_COLUMN, QUERY_COLUMN]],
        on=TERM_ID_COLUMN,
        how="left",
    )
    positives = positives.merge(
        prepared_items[ITEM_FEATURE_COLUMNS],
        on=ITEM_ID_COLUMN,
        how="left",
    )
    positives[QUERY_COLUMN] = positives[QUERY_COLUMN].fillna("").astype(str)
    positives[NORMALIZED_QUERY_COLUMN] = positives[QUERY_COLUMN].map(normalize_text)
    positives[LABEL_COLUMN] = POSITIVE_LABEL
    positives[SAMPLE_TYPE_COLUMN] = SAMPLE_TYPE_POSITIVE
    return positives


def build_positive_pair_set(positive_samples: pd.DataFrame) -> set[tuple[str, str]]:
    """Build a set of known positive ``(term_id, item_id)`` pairs.

    Args:
        positive_samples: Positive sample dataframe.

    Returns:
        Set of positive pair identifiers.
    """
    return set(
        zip(
            positive_samples[TERM_ID_COLUMN].astype(str),
            positive_samples[ITEM_ID_COLUMN].astype(str),
            strict=False,
        )
    )


def to_training_output(samples: pd.DataFrame) -> pd.DataFrame:
    """Select the final output columns for model-ready training data.

    Args:
        samples: Positive and negative sample dataframe.

    Returns:
        DataFrame with the required output columns.
    """
    return samples[OUTPUT_COLUMNS].copy()


class BaseNegativeSampler(ABC):
    """Base class for negative sampling strategies."""

    def __init__(
        self,
        items: pd.DataFrame,
        sample_type: str,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        """Initialize a negative sampler.

        Args:
            items: Prepared item metadata.
            sample_type: Name of the negative sample type.
            random_seed: Random seed for reproducible sampling.
        """
        self.items = items
        self.sample_type = sample_type
        self.rng = np.random.default_rng(random_seed)

    @abstractmethod
    def sample(
        self,
        positive_samples: pd.DataFrame,
        target_count: int,
    ) -> SamplingResult:
        """Generate negative samples.

        Args:
            positive_samples: Positive samples used as anchors.
            target_count: Number of negatives requested.

        Returns:
            Sampling result with generated sample rows and metadata.
        """

    def _empty_result(
        self,
        requested_count: int,
        elapsed_seconds: float,
    ) -> SamplingResult:
        """Build an empty sampling result.

        Args:
            requested_count: Number of negatives requested.
            elapsed_seconds: Runtime of the sampler.

        Returns:
            Empty sampling result.
        """
        return SamplingResult(
            samples=pd.DataFrame(),
            requested_count=requested_count,
            generated_count=0,
            elapsed_seconds=elapsed_seconds,
            sample_type=self.sample_type,
        )

    def _build_negative_rows(
        self,
        anchors: pd.DataFrame,
        sampled_items: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine positive anchors with sampled negative item metadata.

        Args:
            anchors: Positive rows used as query anchors.
            sampled_items: Negative item metadata aligned with anchors.

        Returns:
            Negative sample rows with helper columns.
        """
        negatives = pd.DataFrame(
            {
                TERM_ID_COLUMN: anchors[TERM_ID_COLUMN].to_numpy(),
                QUERY_COLUMN: anchors[QUERY_COLUMN].to_numpy(),
                NORMALIZED_QUERY_COLUMN: anchors[NORMALIZED_QUERY_COLUMN].to_numpy(),
                ITEM_ID_COLUMN: sampled_items[ITEM_ID_COLUMN].to_numpy(),
                TITLE_COLUMN: sampled_items[TITLE_COLUMN].to_numpy(),
                CATEGORY_COLUMN: sampled_items[CATEGORY_COLUMN].to_numpy(),
                BRAND_COLUMN: sampled_items[BRAND_COLUMN].to_numpy(),
                ATTRIBUTES_COLUMN: sampled_items[ATTRIBUTES_COLUMN].to_numpy(),
                MAIN_CATEGORY_COLUMN: sampled_items[MAIN_CATEGORY_COLUMN].to_numpy(),
                SUB_CATEGORY_COLUMN: sampled_items[SUB_CATEGORY_COLUMN].to_numpy(),
                NORMALIZED_TITLE_COLUMN: sampled_items[
                    NORMALIZED_TITLE_COLUMN
                ].to_numpy(),
                LABEL_COLUMN: NEGATIVE_LABEL,
                SAMPLE_TYPE_COLUMN: self.sample_type,
                "positive_item_id": anchors[ITEM_ID_COLUMN].to_numpy(),
                "positive_category": anchors[CATEGORY_COLUMN].to_numpy(),
                "positive_main_category": anchors[MAIN_CATEGORY_COLUMN].to_numpy(),
                "positive_sub_category": anchors[SUB_CATEGORY_COLUMN].to_numpy(),
            }
        )
        return negatives

    def _sample_anchor_rows(
        self,
        positive_samples: pd.DataFrame,
        target_count: int,
    ) -> pd.DataFrame:
        """Sample positive anchors with replacement.

        Args:
            positive_samples: Positive samples used as anchors.
            target_count: Number of anchors to sample.

        Returns:
            Anchor rows aligned to requested target count.
        """
        sampled_indices = self.rng.choice(
            positive_samples.index.to_numpy(),
            size=target_count,
            replace=True,
        )
        return positive_samples.loc[sampled_indices].reset_index(drop=True)

    def _finish(
        self,
        samples: pd.DataFrame,
        requested_count: int,
        start_time: float,
    ) -> SamplingResult:
        """Build a populated sampling result and log summary information.

        Args:
            samples: Generated negative rows.
            requested_count: Number of negatives requested.
            start_time: Sampler start timestamp.

        Returns:
            Sampling result.
        """
        elapsed_seconds = perf_counter() - start_time
        generated_count = int(samples.shape[0])
        logger.info(
            "%s sampling completed | requested=%s | generated=%s | elapsed=%.3fs",
            self.sample_type,
            requested_count,
            generated_count,
            elapsed_seconds,
        )
        return SamplingResult(
            samples=samples,
            requested_count=requested_count,
            generated_count=generated_count,
            elapsed_seconds=elapsed_seconds,
            sample_type=self.sample_type,
        )
