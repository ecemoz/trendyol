"""Unit tests for the BM25 similarity feature extractor."""

import unittest
import numpy as np
import pandas as pd

from src.features.bm25_features import Bm25SimilarityFeatureExtractor


class TestBm25SimilarityFeatureExtractor(unittest.TestCase):
    """Unit tests for Bm25SimilarityFeatureExtractor."""

    def test_transform_without_fit(self) -> None:
        """Verify that transforming without fitting raises a ValueError."""
        df = pd.DataFrame(
            {
                "query": ["kırmızı elbise"],
                "title": ["kırmızı elbise"],
                "category": ["giyim"],
                "attributes": ["renk kırmızı"],
            }
        )
        extractor = Bm25SimilarityFeatureExtractor()
        with self.assertRaisesRegex(ValueError, "Extractor is not fitted"):
            extractor.transform(df)

    def test_missing_columns(self) -> None:
        """Verify that ValueError is raised if input columns are missing."""
        df = pd.DataFrame({"query": ["test"]})
        extractor = Bm25SimilarityFeatureExtractor()
        with self.assertRaisesRegex(ValueError, "Missing required similarity columns"):
            extractor.fit(df)

    def test_correctness(self) -> None:
        """Verify correctness of calculated BM25 relevance scores on sample data."""
        df = pd.DataFrame(
            {
                "query": [
                    "kırmızı elbise",  # Both words present in doc
                    "mavi gömlek",     # Disjoint query vs title
                    "deri ceket",      # Partial match (only "deri" present in title)
                    "",                # Empty query
                    "spor ayakkabı",   # NaN attributes handling
                ],
                "title": [
                    "kırmızı elbise",
                    "yeşil pantolon",
                    "deri mont",
                    "şık gözlük",
                    "klasik ayakkabı",
                ],
                "category": [
                    "kadın giyim",
                    "erkek giyim",
                    "giyim mont",
                    "aksesuar",
                    "ayakkabı spor",
                ],
                "attributes": [
                    "renk kırmızı kumaş pamuk",
                    "renk yeşil model dar",
                    "deri siyah fermuarlı",
                    np.nan,
                    "spor bağcıklı",
                ],
            }
        )

        extractor = Bm25SimilarityFeatureExtractor()
        features = extractor.fit_transform(df)

        # 1. Shape check
        self.assertEqual(features.shape, (5, 3))
        self.assertEqual(
            list(features.columns),
            [
                "bm25_query_title_score",
                "bm25_query_category_score",
                "bm25_query_attribute_score",
            ],
        )

        # 2. Perfect match should have a positive BM25 score
        self.assertGreater(features.loc[0, "bm25_query_title_score"], 0.0)

        # 3. No overlap should have a BM25 score of exactly 0.0
        self.assertEqual(features.loc[1, "bm25_query_title_score"], 0.0)

        # 4. Partial match should have score greater than 0.0 but less than perfect match
        val_partial = features.loc[2, "bm25_query_title_score"]
        val_perfect = features.loc[0, "bm25_query_title_score"]
        self.assertGreater(val_partial, 0.0)
        self.assertGreater(val_perfect, val_partial)

        # 5. Empty inputs / NaN inputs should yield exactly 0.0
        self.assertEqual(features.loc[3, "bm25_query_title_score"], 0.0)
        self.assertEqual(features.loc[3, "bm25_query_attribute_score"], 0.0)

        # 6. Categories and attributes check
        # "spor ayakkabı" vs "ayakkabı spor" (perfect overlap) should have a high score
        self.assertGreater(features.loc[4, "bm25_query_category_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
