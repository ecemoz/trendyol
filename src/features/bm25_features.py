"""BM25 similarity feature extraction utilities."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_QUERY_COLUMN: str = "query"
DEFAULT_TITLE_COLUMN: str = "title"
DEFAULT_CATEGORY_COLUMN: str = "category"
DEFAULT_ATTRIBUTES_COLUMN: str = "attributes"
TOKEN_PATTERN: str = r"[0-9A-Za-zçğıöşüÇĞİÖŞÜ]+"


@dataclass
class Bm25SimilarityFeatureExtractor:
    """Extract Okapi BM25 similarity features between query and other item fields.

    Attributes:
        query_column: Name of the query text column.
        title_column: Name of the product title column.
        category_column: Name of the product category column.
        attributes_column: Name of the product attributes column.
        token_pattern: Regular expression pattern for tokenization.
        k1: BM25 k1 parameter (controls term frequency saturation).
        b: BM25 b parameter (controls document length normalization).
    """

    query_column: str = DEFAULT_QUERY_COLUMN
    title_column: str = DEFAULT_TITLE_COLUMN
    category_column: str = DEFAULT_CATEGORY_COLUMN
    attributes_column: str = DEFAULT_ATTRIBUTES_COLUMN
    token_pattern: str = TOKEN_PATTERN
    k1: float = 1.5
    b: float = 0.75

    _vectorizers: dict[str, CountVectorizer] = field(
        default_factory=dict, init=False
    )
    _avgdls: dict[str, float] = field(default_factory=dict, init=False)
    _idfs: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def fit(self, dataframe: pd.DataFrame) -> "Bm25SimilarityFeatureExtractor":
        """Fit document-specific vectorizers and compute IDF and average document lengths.

        Args:
            dataframe: Input DataFrame.

        Returns:
            The fitted extractor instance.
        """
        self._validate_columns(dataframe)
        logger.info("Fitting BM25 similarity features...")

        # We fit separate vectorizer and compute statistics for each target column
        for col in [self.title_column, self.category_column, self.attributes_column]:
            docs = dataframe[col].map(self._normalize_text)
            vectorizer = CountVectorizer(token_pattern=self.token_pattern)
            TF = vectorizer.fit_transform(docs)

            # Document lengths and average document length
            doc_lens = np.asarray(TF.sum(axis=1)).reshape(-1)
            avgdl = doc_lens.mean()
            self._avgdls[col] = float(avgdl) if avgdl > 0.0 else 1.0

            # Document frequencies and IDF
            df = np.asarray((TF > 0).sum(axis=0)).reshape(-1)
            N = len(docs)
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
            # Clip negative IDF values to 0.0 (standard Okapi BM25 practice)
            idf = np.maximum(idf, 0.0)

            self._vectorizers[col] = vectorizer
            self._idfs[col] = idf

        logger.info("BM25 similarity features fitted successfully.")
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract BM25 features from a DataFrame.

        Args:
            dataframe: Input DataFrame.

        Returns:
            DataFrame containing only generated BM25 similarity features.

        Raises:
            ValueError: If the extractor has not been fitted yet.
        """
        self._validate_columns(dataframe)
        if not self._vectorizers:
            message = "Extractor is not fitted. Call fit() before transform()."
            logger.error(message)
            raise ValueError(message)

        logger.info("Extracting BM25 similarity features...")
        features = pd.DataFrame(index=dataframe.index)

        query_text = dataframe[self.query_column].map(self._normalize_text)

        # 1. Query <-> Title
        title_text = dataframe[self.title_column].map(self._normalize_text)
        features["bm25_query_title_score"] = self._calculate_bm25_scores(
            self.title_column, query_text, title_text
        )

        # 2. Query <-> Category
        category_text = dataframe[self.category_column].map(self._normalize_text)
        features["bm25_query_category_score"] = self._calculate_bm25_scores(
            self.category_column, query_text, category_text
        )

        # 3. Query <-> Attributes
        attribute_text = dataframe[self.attributes_column].map(self._normalize_text)
        features["bm25_query_attribute_score"] = self._calculate_bm25_scores(
            self.attributes_column, query_text, attribute_text
        )

        logger.info("BM25 similarity features extracted successfully.")
        return features

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fit model statistics and extract BM25 features.

        Args:
            dataframe: Input DataFrame.

        Returns:
            DataFrame containing only generated BM25 similarity features.
        """
        return self.fit(dataframe).transform(dataframe)

    def _validate_columns(self, dataframe: pd.DataFrame) -> None:
        """Validate that required columns exist in the DataFrame."""
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
        """Normalize text with trim and lowercase."""
        if pd.isna(value):
            return ""
        return str(value).strip().lower()

    def _calculate_bm25_scores(
        self, col_name: str, query_series: pd.Series, doc_series: pd.Series
    ) -> np.ndarray:
        """Calculate row-wise BM25 similarity score using sparse matrix operations.

        Args:
            col_name: The target column name (used to lookup fitted vectorizer & parameters).
            query_series: Normalized queries.
            doc_series: Normalized target documents.

        Returns:
            1D NumPy array containing the BM25 scores.
        """
        vectorizer = self._vectorizers[col_name]
        idf = self._idfs[col_name]
        avgdl = self._avgdls[col_name]

        # Transform inputs to sparse matrices
        TF = vectorizer.transform(doc_series)
        QF = vectorizer.transform(query_series)

        # Get document lengths
        doc_lens = np.asarray(TF.sum(axis=1)).reshape(-1)

        # Find query term presence (binary matrix)
        QF_binary = QF.sign()

        # Keep only document term frequencies of terms present in the query
        TF_active = TF.multiply(QF_binary)

        # Convert to COO format to get row and column indices of non-zero elements
        coo = TF_active.tocoo()
        r = coo.row
        c = coo.col
        tf_val = coo.data

        # Calculate scores for each active query-document term pair
        doc_len_r = doc_lens[r]
        idf_c = idf[c]

        numerator = tf_val * (self.k1 + 1.0)
        denominator = tf_val + self.k1 * (1.0 - self.b + self.b * doc_len_r / avgdl)
        term_scores = idf_c * (numerator / denominator)

        # Aggregate term scores row-wise (row index corresponds to document index)
        N = len(query_series)
        bm25_scores = np.zeros(N)
        np.add.at(bm25_scores, r, term_scores)

        return bm25_scores
