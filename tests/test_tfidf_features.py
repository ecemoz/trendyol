"""Unit tests for the TF-IDF similarity feature extractor."""

import unittest
import numpy as np
import pandas as pd

from src.features.tfidf_features import TfidfSimilarityFeatureExtractor


class TestTfidfSimilarityFeatureExtractor(unittest.TestCase):
    """Unit tests for TfidfSimilarityFeatureExtractor."""

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
        extractor = TfidfSimilarityFeatureExtractor()
        with self.assertRaisesRegex(ValueError, "Extractor is not fitted"):
            extractor.transform(df)

    def test_missing_columns(self) -> None:
        """Verify that ValueError is raised if input columns are missing."""
        df = pd.DataFrame({"query": ["test"]})
        extractor = TfidfSimilarityFeatureExtractor()
        with self.assertRaisesRegex(ValueError, "Missing required similarity columns"):
            extractor.fit(df)

    def test_correctness(self) -> None:
        """Verify correctness of calculated TF-IDF similarities on sample data."""
        df = pd.DataFrame(
            {
                "query": [
                    "kırmızı elbise",  # Perfect match with title
                    "mavi gömlek",     # No overlap with title (yeşil pantolon)
                    "deri ceket",      # Partial overlap with title (deri mont)
                    "",                # Empty query
                    "spor ayakkabı",   # NaN / missing attributes test
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

        extractor = TfidfSimilarityFeatureExtractor()
        features = extractor.fit_transform(df)

        # 1. Shape check
        self.assertEqual(features.shape, (5, 3))
        self.assertEqual(
            list(features.columns),
            [
                "tfidf_query_title_cosine_similarity",
                "tfidf_query_category_similarity",
                "tfidf_query_attribute_similarity",
            ],
        )

        # 2. Perfect match similarity should be close to 1.0 (with floating-point tolerance)
        np.testing.assert_allclose(
            features.loc[0, "tfidf_query_title_cosine_similarity"],
            1.0,
            atol=1e-5,
        )

        # 3. No overlap similarity should be exactly 0.0
        self.assertEqual(features.loc[1, "tfidf_query_title_cosine_similarity"], 0.0)

        # 4. Partial overlap should be strictly between 0.0 and 1.0
        val_partial = features.loc[2, "tfidf_query_title_cosine_similarity"]
        self.assertTrue(0.0 < val_partial < 1.0)

        # 5. Empty/missing inputs should yield 0.0
        self.assertEqual(features.loc[3, "tfidf_query_title_cosine_similarity"], 0.0)
        self.assertEqual(features.loc[3, "tfidf_query_attribute_similarity"], 0.0)

        # 6. Check categories and attributes
        # "kırmızı elbise" vs "renk kırmızı kumaş pamuk" should have positive similarity due to "kırmızı"
        self.assertGreater(features.loc[0, "tfidf_query_attribute_similarity"], 0.0)
        # "spor ayakkabı" vs "ayakkabı spor" should have similarity close to 1.0
        np.testing.assert_allclose(
            features.loc[4, "tfidf_query_category_similarity"],
            1.0,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
