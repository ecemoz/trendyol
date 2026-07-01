"""Query-based feature extraction utilities."""

import re
from dataclasses import dataclass

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)

DEFAULT_QUERY_COLUMN: str = "query"
TURKISH_CHARACTER_PATTERN: re.Pattern[str] = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
NUMBER_PATTERN: re.Pattern[str] = re.compile(r"\d")
TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"[0-9A-Za-zçğıöşüÇĞİÖŞÜ]+",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class QueryFeatureExtractor:
    """Extract reusable query-only features from search text.

    Attributes:
        query_column: Name of the column containing raw query text.
        output_prefix: Prefix used for generated feature column names.
    """

    query_column: str = DEFAULT_QUERY_COLUMN
    output_prefix: str = "query"

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract query-based features from a DataFrame.

        Args:
            dataframe: Input DataFrame containing the configured query column.

        Returns:
            DataFrame containing only generated query feature columns.

        Raises:
            ValueError: If the configured query column is missing.
        """
        if self.query_column not in dataframe.columns:
            message = f"Missing query column: {self.query_column}"
            logger.error(message)
            raise ValueError(message)

        logger.info("Extracting query features from column: %s", self.query_column)
        queries = dataframe[self.query_column].fillna("").astype(str)
        stripped_queries = queries.str.strip()
        tokens = stripped_queries.map(self._tokenize)
        token_lengths = tokens.map(lambda values: [len(value) for value in values])

        features = pd.DataFrame(index=dataframe.index)
        features[f"{self.output_prefix}_char_count"] = stripped_queries.str.len()
        features[f"{self.output_prefix}_word_count"] = stripped_queries.map(
            self._count_words
        )
        features[f"{self.output_prefix}_is_single_word"] = (
            features[f"{self.output_prefix}_word_count"] == 1
        ).astype(int)
        features[f"{self.output_prefix}_contains_number"] = stripped_queries.str.contains(
            NUMBER_PATTERN,
            regex=True,
        ).astype(int)
        features[f"{self.output_prefix}_contains_turkish_char"] = (
            stripped_queries.str.contains(TURKISH_CHARACTER_PATTERN, regex=True)
            .fillna(False)
            .astype(int)
        )
        features[f"{self.output_prefix}_is_lowercase"] = stripped_queries.map(
            self._is_lowercase
        ).astype(int)
        features[f"{self.output_prefix}_token_count"] = tokens.map(len)
        features[f"{self.output_prefix}_unique_token_count"] = tokens.map(
            lambda values: len(set(values))
        )
        features[f"{self.output_prefix}_avg_token_length"] = token_lengths.map(
            self._average_length
        )

        logger.info("Query feature extraction completed")
        return features

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract query features with a scikit-learn-like interface.

        Args:
            dataframe: Input DataFrame containing the configured query column.

        Returns:
            DataFrame containing only generated query feature columns.
        """
        return self.transform(dataframe)

    @staticmethod
    def _tokenize(query: str) -> list[str]:
        """Tokenize query text into lowercase alphanumeric tokens.

        Args:
            query: Raw query text.

        Returns:
            List of normalized tokens.
        """
        return [token.lower() for token in TOKEN_PATTERN.findall(query)]

    @staticmethod
    def _count_words(query: str) -> int:
        """Count whitespace-separated words in query text.

        Args:
            query: Raw query text.

        Returns:
            Number of non-empty whitespace-separated words.
        """
        if not query:
            return 0
        return len(query.split())

    @staticmethod
    def _is_lowercase(query: str) -> bool:
        """Check whether query text is already lowercase.

        Args:
            query: Raw query text.

        Returns:
            True when the text equals its lowercase representation.
        """
        return query == query.lower()

    @staticmethod
    def _average_length(lengths: list[int]) -> float:
        """Calculate the average token length.

        Args:
            lengths: Token length values.

        Returns:
            Average length, or 0.0 when no tokens exist.
        """
        if not lengths:
            return 0.0
        return float(sum(lengths) / len(lengths))
