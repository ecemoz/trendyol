"""Brand-based feature extraction utilities."""

import re
from dataclasses import dataclass

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)

DEFAULT_BRAND_COLUMN: str = "brand"
UNKNOWN_BRAND_VALUES: frozenset[str] = frozenset(
    {
        "unknown",
        "unk",
        "none",
        "null",
        "nan",
        "na",
        "n/a",
        "belirsiz",
        "bilinmiyor",
    }
)
TURKISH_CHARACTER_PATTERN: re.Pattern[str] = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
NUMBER_PATTERN: re.Pattern[str] = re.compile(r"\d")
TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"[0-9A-Za-zçğıöşüÇĞİÖŞÜ]+",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class BrandFeatureExtractor:
    """Extract reusable brand-only features from brand text.

    Attributes:
        brand_column: Name of the column containing raw brand text.
        output_prefix: Prefix used for generated feature column names.
    """

    brand_column: str = DEFAULT_BRAND_COLUMN
    output_prefix: str = "brand"

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract brand-based features from a DataFrame.

        Args:
            dataframe: Input DataFrame containing the configured brand column.

        Returns:
            DataFrame containing only generated brand feature columns.

        Raises:
            ValueError: If the configured brand column is missing.
        """
        if self.brand_column not in dataframe.columns:
            message = f"Missing brand column: {self.brand_column}"
            logger.error(message)
            raise ValueError(message)

        logger.info("Extracting brand features from column: %s", self.brand_column)
        brands = dataframe[self.brand_column].map(self._normalize_brand)
        tokens = brands.map(self._tokenize)

        features = pd.DataFrame(index=dataframe.index)
        features[f"{self.output_prefix}_normalized"] = brands
        features[f"{self.output_prefix}_char_count"] = brands.str.len()
        features[f"{self.output_prefix}_word_count"] = brands.map(self._count_words)
        features[f"{self.output_prefix}_token_count"] = tokens.map(len)
        features[f"{self.output_prefix}_contains_number"] = brands.str.contains(
            NUMBER_PATTERN,
            regex=True,
        ).astype(int)
        features[f"{self.output_prefix}_contains_turkish_char"] = (
            brands.str.contains(TURKISH_CHARACTER_PATTERN, regex=True)
            .fillna(False)
            .astype(int)
        )
        features[f"{self.output_prefix}_is_unknown"] = brands.map(
            self._is_unknown
        ).astype(int)
        features[f"{self.output_prefix}_is_empty"] = brands.eq("").astype(int)

        logger.info("Brand feature extraction completed")
        return features

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract brand features with a scikit-learn-like interface.

        Args:
            dataframe: Input DataFrame containing the configured brand column.

        Returns:
            DataFrame containing only generated brand feature columns.
        """
        return self.transform(dataframe)

    @staticmethod
    def _normalize_brand(brand: object) -> str:
        """Normalize brand text with trim and lowercase.

        Args:
            brand: Raw brand value.

        Returns:
            Normalized brand text. Missing values become an empty string.
        """
        if pd.isna(brand):
            return ""
        return str(brand).strip().lower()

    @staticmethod
    def _tokenize(brand: str) -> list[str]:
        """Tokenize brand text into lowercase alphanumeric tokens.

        Args:
            brand: Normalized brand text.

        Returns:
            List of tokens.
        """
        return TOKEN_PATTERN.findall(brand)

    @staticmethod
    def _count_words(brand: str) -> int:
        """Count whitespace-separated words in brand text.

        Args:
            brand: Normalized brand text.

        Returns:
            Number of non-empty whitespace-separated words.
        """
        if not brand:
            return 0
        return len(brand.split())

    @staticmethod
    def _is_unknown(brand: str) -> bool:
        """Check whether a normalized brand is an unknown placeholder.

        Args:
            brand: Normalized brand text.

        Returns:
            True when brand is a known unknown placeholder.
        """
        return brand in UNKNOWN_BRAND_VALUES
