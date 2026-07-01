"""TF-IDF similarity feature extraction utilities."""

from dataclasses import dataclass, field
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_QUERY_COLUMN: str = "query"
DEFAULT_TITLE_COLUMN: str = "title"
DEFAULT_CATEGORY_COLUMN: str = "category"
DEFAULT_ATTRIBUTES_COLUMN: str = "attributes"
TOKEN_PATTERN: str = r"[0-9A-Za-zçğıöşüÇĞİÖŞÜ]+"


@dataclass
class TfidfSimilarityFeatureExtractor:
    """Extract TF-IDF cosine similarity features between query and other item fields.

    Attributes:
        query_column: Name of the query text column.
        title_column: Name of the product title column.
        category_column: Name of the product category column.
        attributes_column: Name of the product attributes column.
        token_pattern: Regular expression pattern for tokenization.
    """

    query_column: str = DEFAULT_QUERY_COLUMN
    title_column: str = DEFAULT_TITLE_COLUMN
    category_column: str = DEFAULT_CATEGORY_COLUMN
    attributes_column: str = DEFAULT_ATTRIBUTES_COLUMN
    token_pattern: str = TOKEN_PATTERN

    _vectorizers: dict[str, TfidfVectorizer] = field(
        default_factory=dict, init=False
    )

    def fit(self, dataframe: pd.DataFrame) -> "TfidfSimilarityFeatureExtractor":
        """Fit joint TF-IDF vectorizers for each query-target column pair.

        Args:
            dataframe: Input DataFrame containing the query and target columns.

        Returns:
            The fitted extractor instance.
        """
        self._validate_columns(dataframe)
        logger.info("Fitting TF-IDF similarity vectorizers...")

        # Get normalized texts
        query_text = dataframe[self.query_column].map(self._normalize_text)

        # 1. Query <-> Title
        title_text = dataframe[self.title_column].map(self._normalize_text)
        corpus_title = pd.concat([query_text, title_text], ignore_index=True)
        vec_title = TfidfVectorizer(token_pattern=self.token_pattern, norm="l2")
        vec_title.fit(corpus_title)
        self._vectorizers["title"] = vec_title

        # 2. Query <-> Category
        category_text = dataframe[self.category_column].map(self._normalize_text)
        corpus_category = pd.concat([query_text, category_text], ignore_index=True)
        vec_category = TfidfVectorizer(token_pattern=self.token_pattern, norm="l2")
        vec_category.fit(corpus_category)
        self._vectorizers["category"] = vec_category

        # 3. Query <-> Attributes
        attribute_text = dataframe[self.attributes_column].map(self._normalize_text)
        corpus_attribute = pd.concat([query_text, attribute_text], ignore_index=True)
        vec_attribute = TfidfVectorizer(token_pattern=self.token_pattern, norm="l2")
        vec_attribute.fit(corpus_attribute)
        self._vectorizers["attributes"] = vec_attribute

        logger.info("TF-IDF similarity vectorizers fitted successfully.")
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract TF-IDF similarity features from a DataFrame.

        Args:
            dataframe: Input DataFrame.

        Returns:
            DataFrame containing only generated TF-IDF similarity feature columns.

        Raises:
            ValueError: If the extractor is not fitted or any required column is missing.
        """
        self._validate_columns(dataframe)
        if not self._vectorizers:
            message = "Extractor is not fitted. Call fit() before transform()."
            logger.error(message)
            raise ValueError(message)

        logger.info("Extracting TF-IDF similarity features...")
        features = pd.DataFrame(index=dataframe.index)

        # Normalize texts
        query_text = dataframe[self.query_column].map(self._normalize_text)

        # Compute query <-> title similarity
        title_text = dataframe[self.title_column].map(self._normalize_text)
        features["tfidf_query_title_cosine_similarity"] = (
            self._calculate_cosine_similarity(
                self._vectorizers["title"], query_text, title_text
            )
        )

        # Compute query <-> category similarity
        category_text = dataframe[self.category_column].map(self._normalize_text)
        features["tfidf_query_category_similarity"] = (
            self._calculate_cosine_similarity(
                self._vectorizers["category"], query_text, category_text
            )
        )

        # Compute query <-> attribute similarity
        attribute_text = dataframe[self.attributes_column].map(self._normalize_text)
        features["tfidf_query_attribute_similarity"] = (
            self._calculate_cosine_similarity(
                self._vectorizers["attributes"], query_text, attribute_text
            )
        )

        logger.info("TF-IDF similarity features extracted successfully.")
        return features

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fit joint TF-IDF vectorizers and extract features.

        Args:
            dataframe: Input DataFrame.

        Returns:
            DataFrame containing only generated TF-IDF similarity feature columns.
        """
        return self.fit(dataframe).transform(dataframe)

    def _validate_columns(self, dataframe: pd.DataFrame) -> None:
        """Validate that all required columns exist in the DataFrame.

        Args:
            dataframe: Input DataFrame.

        Raises:
            ValueError: If any required column is missing.
        """
        required_columns = {
            self.query_column,
            self.title_column,
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
    def _calculate_cosine_similarity(
        vectorizer: TfidfVectorizer, col1_series: pd.Series, col2_series: pd.Series
    ) -> np.ndarray:
        """Calculate row-wise cosine similarity between two text columns.

        Args:
            vectorizer: Fitted TfidfVectorizer.
            col1_series: First column text.
            col2_series: Second column text.

        Returns:
            1D NumPy array containing the cosine similarity values.
        """
        # Transform the series to sparse matrices
        mat1 = vectorizer.transform(col1_series)
        mat2 = vectorizer.transform(col2_series)

        # Row-wise dot product of L2-normalized sparse matrices
        # Since vectorizer outputs L2 normalized rows, the row-wise dot product
        # is mathematically equal to cosine similarity.
        similarity = np.asarray(mat1.multiply(mat2).sum(axis=1)).reshape(-1)
        return similarity
