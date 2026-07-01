"""Category-based feature extraction utilities."""

from dataclasses import dataclass

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)

DEFAULT_CATEGORY_COLUMN: str = "category"
CATEGORY_SEPARATOR: str = "/"
MAX_CATEGORY_LEVEL: int = 6


@dataclass(frozen=True)
class CategoryFeatureExtractor:
    """Extract reusable category-only features from category paths.

    Attributes:
        category_column: Name of the column containing raw category paths.
        output_prefix: Prefix used for generated category level columns.
    """

    category_column: str = DEFAULT_CATEGORY_COLUMN
    output_prefix: str = "category"

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract category-based features from a DataFrame.

        Args:
            dataframe: Input DataFrame containing the configured category column.

        Returns:
            DataFrame containing only generated category feature columns.

        Raises:
            ValueError: If the configured category column is missing.
        """
        if self.category_column not in dataframe.columns:
            message = f"Missing category column: {self.category_column}"
            logger.error(message)
            raise ValueError(message)

        logger.info(
            "Extracting category features from column: %s",
            self.category_column,
        )
        categories = dataframe[self.category_column].fillna("").astype(str)
        category_levels = categories.map(self._split_category)

        features = pd.DataFrame(index=dataframe.index)
        features[f"{self.output_prefix}_depth"] = category_levels.map(len)
        features["main_category"] = category_levels.map(
            lambda levels: levels[0] if levels else ""
        )
        features["leaf_category"] = category_levels.map(
            lambda levels: levels[-1] if levels else ""
        )

        for level_index in range(MAX_CATEGORY_LEVEL):
            feature_name = f"{self.output_prefix}_level_{level_index + 1}"
            features[feature_name] = category_levels.map(
                lambda levels, index=level_index: levels[index]
                if index < len(levels)
                else ""
            )

        features[f"{self.output_prefix}_has_multiple_levels"] = (
            features[f"{self.output_prefix}_depth"] > 1
        ).astype(int)

        logger.info("Category feature extraction completed")
        return features

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract category features with a scikit-learn-like interface.

        Args:
            dataframe: Input DataFrame containing the configured category column.

        Returns:
            DataFrame containing only generated category feature columns.
        """
        return self.transform(dataframe)

    @staticmethod
    def _split_category(category: str) -> list[str]:
        """Split a category path into clean hierarchy levels.

        Args:
            category: Raw category path.

        Returns:
            Ordered category levels. Empty levels are removed.
        """
        stripped_category = category.strip().lower()
        if not stripped_category:
            return []

        return [
            level.strip()
            for level in stripped_category.split(CATEGORY_SEPARATOR)
            if level.strip()
        ]
