"""Easy negative sampler using products from different main categories."""

from time import perf_counter

import pandas as pd

from src.config.negative_sampling import RANDOM_SEED, SAMPLE_TYPE_EASY
from src.negative_sampling.sampler import (
    BaseNegativeSampler,
    MAIN_CATEGORY_COLUMN,
    SamplingResult,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class EasyNegativeSampler(BaseNegativeSampler):
    """Sample easy negatives from a different main category than the anchor."""

    def __init__(
        self,
        items: pd.DataFrame,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        """Initialize the easy negative sampler.

        Args:
            items: Prepared item metadata.
            random_seed: Random seed for reproducible sampling.
        """
        super().__init__(
            items=items,
            sample_type=SAMPLE_TYPE_EASY,
            random_seed=random_seed,
        )
        self.category_to_indices = {
            category: group.index.to_numpy()
            for category, group in self.items.groupby(MAIN_CATEGORY_COLUMN)
        }
        self.all_indices = self.items.index.to_numpy()
        self.categories = list(self.category_to_indices.keys())

    def _sample_item_index_for_anchor(self, anchor_main_category: str) -> int:
        """Sample one item index outside the anchor main category.

        Args:
            anchor_main_category: Main category of the positive anchor item.

        Returns:
            Sampled item index from a different main category.
        """
        candidate_categories = [
            category
            for category in self.categories
            if category != anchor_main_category
            and len(self.category_to_indices.get(category, [])) > 0
        ]
        if not candidate_categories:
            return int(self.rng.choice(self.all_indices))

        sampled_category = str(self.rng.choice(candidate_categories))
        candidate_indices = self.category_to_indices[sampled_category]
        return int(self.rng.choice(candidate_indices))

    def sample(
        self,
        positive_samples: pd.DataFrame,
        target_count: int,
    ) -> SamplingResult:
        """Generate easy negative samples.

        Args:
            positive_samples: Positive samples used as query anchors.
            target_count: Number of negatives requested.

        Returns:
            Sampling result with easy negative rows.
        """
        logger.info("Starting easy negative sampling | target=%s", target_count)
        start_time = perf_counter()

        if target_count <= 0 or positive_samples.empty or self.items.empty:
            return self._empty_result(
                requested_count=target_count,
                elapsed_seconds=perf_counter() - start_time,
            )

        anchors = self._sample_anchor_rows(
            positive_samples=positive_samples,
            target_count=target_count,
        )
        sampled_indices = [
            self._sample_item_index_for_anchor(anchor_main_category)
            for anchor_main_category in anchors[MAIN_CATEGORY_COLUMN].astype(str)
        ]
        sampled_items = self.items.loc[sampled_indices].reset_index(drop=True)
        samples = self._build_negative_rows(
            anchors=anchors,
            sampled_items=sampled_items,
        )
        return self._finish(
            samples=samples,
            requested_count=target_count,
            start_time=start_time,
        )
