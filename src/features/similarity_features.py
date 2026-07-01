"""Lexical similarity feature extraction utilities."""

import re
from dataclasses import dataclass

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)

DEFAULT_QUERY_COLUMN: str = "query"
DEFAULT_TITLE_COLUMN: str = "title"
DEFAULT_BRAND_COLUMN: str = "brand"
DEFAULT_CATEGORY_COLUMN: str = "category"
DEFAULT_ATTRIBUTES_COLUMN: str = "attributes"
TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"[0-9A-Za-zçğıöşüÇĞİÖŞÜ]+",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class SimilarityFeatureExtractor:
    """Extract simple lexical similarity features between query and item fields.

    Attributes:
        query_column: Name of the query text column.
        title_column: Name of the product title column.
        brand_column: Name of the product brand column.
        category_column: Name of the product category column.
        attributes_column: Name of the product attributes column.
    """

    query_column: str = DEFAULT_QUERY_COLUMN
    title_column: str = DEFAULT_TITLE_COLUMN
    brand_column: str = DEFAULT_BRAND_COLUMN
    category_column: str = DEFAULT_CATEGORY_COLUMN
    attributes_column: str = DEFAULT_ATTRIBUTES_COLUMN

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract lexical similarity features from a DataFrame.

        Args:
            dataframe: Input DataFrame containing query and item text columns.

        Returns:
            DataFrame containing only generated similarity feature columns.

        Raises:
            ValueError: If any required input column is missing.
        """
        self._validate_columns(dataframe)
        logger.info("Extracting lexical similarity features")

        query_text = dataframe[self.query_column].map(self._normalize_text)
        title_text = dataframe[self.title_column].map(self._normalize_text)
        brand_text = dataframe[self.brand_column].map(self._normalize_text)
        category_text = dataframe[self.category_column].map(self._normalize_text)
        attribute_text = dataframe[self.attributes_column].map(self._normalize_text)

        query_tokens = query_text.map(self._tokenize_to_set)
        title_tokens = title_text.map(self._tokenize_to_set)
        category_tokens = category_text.map(self._tokenize_to_set)
        attribute_tokens = attribute_text.map(self._tokenize_to_set)

        features = pd.DataFrame(index=dataframe.index)
        features["query_title_word_overlap_count"] = [
            self._overlap_count(query, title)
            for query, title in zip(query_tokens, title_tokens, strict=False)
        ]
        features["query_title_word_overlap_ratio"] = [
            self._overlap_ratio(query, title)
            for query, title in zip(query_tokens, title_tokens, strict=False)
        ]
        features["query_title_jaccard_similarity"] = [
            self._jaccard_similarity(query, title)
            for query, title in zip(query_tokens, title_tokens, strict=False)
        ]
        features["query_brand_exact_match"] = [
            int(bool(query) and query == brand)
            for query, brand in zip(query_text, brand_text, strict=False)
        ]
        features["query_contains_brand"] = [
            int(bool(brand) and brand in query)
            for query, brand in zip(query_text, brand_text, strict=False)
        ]
        features["query_category_word_overlap_count"] = [
            self._overlap_count(query, category)
            for query, category in zip(query_tokens, category_tokens, strict=False)
        ]
        features["query_category_word_overlap_ratio"] = [
            self._overlap_ratio(query, category)
            for query, category in zip(query_tokens, category_tokens, strict=False)
        ]
        features["query_attribute_word_overlap_count"] = [
            self._overlap_count(query, attributes)
            for query, attributes in zip(query_tokens, attribute_tokens, strict=False)
        ]
        features["query_attribute_word_overlap_ratio"] = [
            self._overlap_ratio(query, attributes)
            for query, attributes in zip(query_tokens, attribute_tokens, strict=False)
        ]

        logger.info("Lexical similarity feature extraction completed")
        return features

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract lexical similarity features with a scikit-learn-like interface.

        Args:
            dataframe: Input DataFrame containing query and item text columns.

        Returns:
            DataFrame containing only generated similarity feature columns.
        """
        return self.transform(dataframe)

    def _validate_columns(self, dataframe: pd.DataFrame) -> None:
        """Validate that all required columns exist.

        Args:
            dataframe: Input DataFrame.

        Raises:
            ValueError: If any required column is missing.
        """
        required_columns = {
            self.query_column,
            self.title_column,
            self.brand_column,
            self.category_column,
            self.attributes_column,
        }
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            message = f"Missing required similarity columns: {sorted(missing_columns)}"
            logger.error(message)
            raise ValueError(message)

    @staticmethod
    def _normalize_text(value: object) -> str:
        """Normalize text with trim and lowercase.

        Args:
            value: Raw text value.

        Returns:
            Normalized text. Missing values become an empty string.
        """
        if pd.isna(value):
            return ""
        return str(value).strip().lower()

    @staticmethod
    def _tokenize_to_set(text: str) -> set[str]:
        """Tokenize normalized text into a unique token set.

        Args:
            text: Normalized text.

        Returns:
            Set of lexical tokens.
        """
        return set(TOKEN_PATTERN.findall(text))

    @staticmethod
    def _overlap_count(left_tokens: set[str], right_tokens: set[str]) -> int:
        """Count overlapping tokens between two token sets.

        Args:
            left_tokens: First token set.
            right_tokens: Second token set.

        Returns:
            Number of shared tokens.
        """
        return len(left_tokens.intersection(right_tokens))

    @classmethod
    def _overlap_ratio(cls, query_tokens: set[str], target_tokens: set[str]) -> float:
        """Calculate query-token coverage by a target token set.

        Args:
            query_tokens: Query token set.
            target_tokens: Target field token set.

        Returns:
            Overlap count divided by query token count. Returns 0.0 for empty query.
        """
        if not query_tokens:
            return 0.0
        return cls._overlap_count(query_tokens, target_tokens) / len(query_tokens)

    @classmethod
    def _jaccard_similarity(
        cls,
        left_tokens: set[str],
        right_tokens: set[str],
    ) -> float:
        """Calculate token-level Jaccard similarity.

        Args:
            left_tokens: First token set.
            right_tokens: Second token set.

        Returns:
            Jaccard similarity in the 0-1 range.
        """
        if not left_tokens and not right_tokens:
            return 0.0
        union_count = len(left_tokens.union(right_tokens))
        if union_count == 0:
            return 0.0
        return cls._overlap_count(left_tokens, right_tokens) / union_count
