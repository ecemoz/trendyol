"""Title-based feature extraction utilities."""

import re
from dataclasses import dataclass

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)

DEFAULT_TITLE_COLUMN: str = "title"
TURKISH_CHARACTER_PATTERN: re.Pattern[str] = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
NUMBER_PATTERN: re.Pattern[str] = re.compile(r"\d")
TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"[0-9A-Za-zçğıöşüÇĞİÖŞÜ]+",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class TitleFeatureExtractor:
    """Extract reusable title-only features from product titles.

    Attributes:
        title_column: Name of the column containing raw product title text.
        output_prefix: Prefix used for generated feature column names.
    """

    title_column: str = DEFAULT_TITLE_COLUMN
    output_prefix: str = "title"

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract title-based features from a DataFrame.

        Args:
            dataframe: Input DataFrame containing the configured title column.

        Returns:
            DataFrame containing only generated title feature columns.

        Raises:
            ValueError: If the configured title column is missing.
        """
        if self.title_column not in dataframe.columns:
            message = f"Missing title column: {self.title_column}"
            logger.error(message)
            raise ValueError(message)

        logger.info("Extracting title features from column: %s", self.title_column)
        titles = dataframe[self.title_column].fillna("").astype(str)
        stripped_titles = titles.str.strip()
        tokens = stripped_titles.map(self._tokenize)
        token_lengths = tokens.map(lambda values: [len(value) for value in values])

        features = pd.DataFrame(index=dataframe.index)
        features[f"{self.output_prefix}_char_count"] = stripped_titles.str.len()
        features[f"{self.output_prefix}_word_count"] = stripped_titles.map(
            self._count_words
        )
        features[f"{self.output_prefix}_token_count"] = tokens.map(len)
        features[f"{self.output_prefix}_unique_token_count"] = tokens.map(
            lambda values: len(set(values))
        )
        features[f"{self.output_prefix}_avg_token_length"] = token_lengths.map(
            self._average_length
        )
        features[f"{self.output_prefix}_contains_number"] = stripped_titles.str.contains(
            NUMBER_PATTERN,
            regex=True,
        ).astype(int)
        features[f"{self.output_prefix}_contains_turkish_char"] = (
            stripped_titles.str.contains(TURKISH_CHARACTER_PATTERN, regex=True)
            .fillna(False)
            .astype(int)
        )
        features[f"{self.output_prefix}_is_lowercase"] = stripped_titles.map(
            self._is_lowercase
        ).astype(int)

        logger.info("Title feature extraction completed")
        return features

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract title features with a scikit-learn-like interface.

        Args:
            dataframe: Input DataFrame containing the configured title column.

        Returns:
            DataFrame containing only generated title feature columns.
        """
        return self.transform(dataframe)

    @staticmethod
    def _tokenize(title: str) -> list[str]:
        """Tokenize title text into lowercase alphanumeric tokens.

        Args:
            title: Raw title text.

        Returns:
            List of normalized tokens.
        """
        return [token.lower() for token in TOKEN_PATTERN.findall(title)]

    @staticmethod
    def _count_words(title: str) -> int:
        """Count whitespace-separated words in title text.

        Args:
            title: Raw title text.

        Returns:
            Number of non-empty whitespace-separated words.
        """
        if not title:
            return 0
        return len(title.split())

    @staticmethod
    def _is_lowercase(title: str) -> bool:
        """Check whether title text is already lowercase.

        Args:
            title: Raw title text.

        Returns:
            True when the text equals its lowercase representation.
        """
        return title == title.lower()

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
