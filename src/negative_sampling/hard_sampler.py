"""Hard negative sampler using same category and lexical title similarity."""

from time import perf_counter

import pandas as pd

from src.config.negative_sampling import (
    HARD_CANDIDATE_POOL_SIZE,
    RANDOM_SEED,
    SAMPLE_TYPE_HARD,
)
from src.negative_sampling.sampler import (
    BaseNegativeSampler,
    CATEGORY_COLUMN,
    ITEM_ID_COLUMN,
    NORMALIZED_QUERY_COLUMN,
    NORMALIZED_TITLE_COLUMN,
    SamplingResult,
)
from src.negative_sampling.validator import lexical_jaccard_similarity
from src.utils.logger import get_logger


logger = get_logger(__name__)


class HardNegativeSampler(BaseNegativeSampler):
    """Sample hard negatives from the same category with similar titles."""

    def __init__(
        self,
        items: pd.DataFrame,
        random_seed: int = RANDOM_SEED,
        candidate_pool_size: int = HARD_CANDIDATE_POOL_SIZE,
    ) -> None:
        """Initialize the hard negative sampler.

        Args:
            items: Prepared item metadata.
            random_seed: Random seed for reproducible sampling.
            candidate_pool_size: Number of category-level candidates to score.
        """
        super().__init__(
            items=items,
            sample_type=SAMPLE_TYPE_HARD,
            random_seed=random_seed,
        )
        self.candidate_pool_size = candidate_pool_size
        self.category_to_indices = {
            category: group.index.to_numpy()
            for category, group in self.items.groupby(CATEGORY_COLUMN)
        }
        self.all_indices = self.items.index.to_numpy()

    def _sample_candidate_indices(
        self,
        category: str,
        positive_item_id: str,
    ) -> list[int]:
        """Sample candidate item indices from the same full category.

        Args:
            category: Full category path of the positive anchor item.
            positive_item_id: Positive anchor item id to exclude.

        Returns:
            Candidate item indices.
        """
        category_indices = self.category_to_indices.get(category)
        if category_indices is None or len(category_indices) == 0:
            category_indices = self.all_indices

        pool_size = min(self.candidate_pool_size, len(category_indices))
        sampled_indices = self.rng.choice(
            category_indices,
            size=pool_size,
            replace=False,
        )
        candidates = self.items.loc[sampled_indices]
        candidates = candidates[
            candidates[ITEM_ID_COLUMN].astype(str) != str(positive_item_id)
        ]

        if candidates.empty:
            fallback_index = int(self.rng.choice(self.all_indices))
            return [fallback_index]

        return [int(index) for index in candidates.index]

    def _select_best_candidate_index(
        self,
        query_text: str,
        candidate_indices: list[int],
    ) -> int:
        """Select the candidate with the highest query-title lexical similarity.

        Args:
            query_text: Normalized query text.
            candidate_indices: Candidate item indices.

        Returns:
            Best candidate item index.
        """
        candidates = self.items.loc[candidate_indices]
        scores = [
            lexical_jaccard_similarity(query_text, title)
            for title in candidates[NORMALIZED_TITLE_COLUMN]
        ]
        best_position = int(pd.Series(scores).idxmax())
        return int(candidates.index[best_position])

    def sample(
        self,
        positive_samples: pd.DataFrame,
        target_count: int,
    ) -> SamplingResult:
        """Generate hard negative samples.

        Args:
            positive_samples: Positive samples used as query anchors.
            target_count: Number of negatives requested.

        Returns:
            Sampling result with hard negative rows.
        """
        logger.info("Starting hard negative sampling | target=%s", target_count)
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
            self._select_best_candidate_index(
                query_text=query_text,
                candidate_indices=self._sample_candidate_indices(
                    category=category,
                    positive_item_id=positive_item_id,
                ),
            )
            for query_text, category, positive_item_id in zip(
                anchors[NORMALIZED_QUERY_COLUMN].astype(str),
                anchors[CATEGORY_COLUMN].astype(str),
                anchors[ITEM_ID_COLUMN].astype(str),
                strict=False,
            )
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
