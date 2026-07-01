"""Medium negative sampler using same main category and different subcategory."""

from time import perf_counter

import pandas as pd

from src.config.negative_sampling import RANDOM_SEED, SAMPLE_TYPE_MEDIUM
from src.negative_sampling.sampler import (
    BaseNegativeSampler,
    MAIN_CATEGORY_COLUMN,
    SUB_CATEGORY_COLUMN,
    SamplingResult,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class MediumNegativeSampler(BaseNegativeSampler):
    """Sample medium negatives from same main category but different subcategory."""

    def __init__(
        self,
        items: pd.DataFrame,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        """Initialize the medium negative sampler.

        Args:
            items: Prepared item metadata.
            random_seed: Random seed for reproducible sampling.
        """
        super().__init__(
            items=items,
            sample_type=SAMPLE_TYPE_MEDIUM,
            random_seed=random_seed,
        )
        self.group_to_indices = {
            (main_category, sub_category): group.index.to_numpy()
            for (main_category, sub_category), group in self.items.groupby(
                [MAIN_CATEGORY_COLUMN, SUB_CATEGORY_COLUMN]
            )
        }
        self.main_category_to_subcategories: dict[str, list[str]] = {}
        for main_category, sub_category in self.group_to_indices:
            self.main_category_to_subcategories.setdefault(main_category, []).append(
                sub_category
            )
        self.all_indices = self.items.index.to_numpy()

    def _sample_item_index_for_anchor(
        self,
        anchor_main_category: str,
        anchor_sub_category: str,
    ) -> int:
        """Sample one item index from same main category and different subcategory.

        Args:
            anchor_main_category: Main category of the positive anchor item.
            anchor_sub_category: First subcategory of the positive anchor item.

        Returns:
            Sampled item index.
        """
        candidate_subcategories = [
            sub_category
            for sub_category in self.main_category_to_subcategories.get(
                anchor_main_category,
                [],
            )
            if sub_category != anchor_sub_category
            and len(
                self.group_to_indices.get((anchor_main_category, sub_category), [])
            )
            > 0
        ]

        if not candidate_subcategories:
            return int(self.rng.choice(self.all_indices))

        sampled_subcategory = str(self.rng.choice(candidate_subcategories))
        candidate_indices = self.group_to_indices[
            (anchor_main_category, sampled_subcategory)
        ]
        return int(self.rng.choice(candidate_indices))

    def sample(
        self,
        positive_samples: pd.DataFrame,
        target_count: int,
    ) -> SamplingResult:
        """Generate medium negative samples.

        Args:
            positive_samples: Positive samples used as query anchors.
            target_count: Number of negatives requested.

        Returns:
            Sampling result with medium negative rows.
        """
        logger.info("Starting medium negative sampling | target=%s", target_count)
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
            self._sample_item_index_for_anchor(
                anchor_main_category=anchor_main_category,
                anchor_sub_category=anchor_sub_category,
            )
            for anchor_main_category, anchor_sub_category in zip(
                anchors[MAIN_CATEGORY_COLUMN].astype(str),
                anchors[SUB_CATEGORY_COLUMN].astype(str),
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
