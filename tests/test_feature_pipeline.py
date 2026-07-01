"""Integration test for FeaturePipeline integration with TF-IDF features."""

import unittest
import pandas as pd
from src.features.feature_pipeline import FeaturePipeline


class TestFeaturePipelineIntegration(unittest.TestCase):
    """Verify FeaturePipeline integrates TF-IDF features correctly."""

    def test_pipeline_execution(self) -> None:
        """Run FeaturePipeline on a small dataset and check columns."""
        df = pd.DataFrame(
            {
                "query": ["kırmızı elbise", "mavi gömlek"],
                "title": ["kırmızı elbise", "yeşil pantolon"],
                "category": ["kadın giyim", "erkek giyim"],
                "brand": ["trendyol", "mavi"],
                "attributes": ["renk kırmızı", "renk yeşil"],
                "label": [1, 0],
                "sample_type": ["easy", "hard"],
            }
        )

        pipeline = FeaturePipeline()
        features = pipeline.transform(df)

        # Output shape validation
        self.assertEqual(features.shape[0], 2)
        
        # Verify new TF-IDF similarity columns are present
        new_cols = [
            "tfidf_query_title_cosine_similarity",
            "tfidf_query_category_similarity",
            "tfidf_query_attribute_similarity",
        ]
        for col in new_cols:
            self.assertIn(col, features.columns)
            
        # Verify the preserved label and sample_type columns are present
        self.assertIn("label", features.columns)
        self.assertIn("sample_type", features.columns)
        
        # Verify values are populated correctly
        self.assertGreater(features.loc[0, "tfidf_query_title_cosine_similarity"], 0.9)
        self.assertEqual(features.loc[1, "tfidf_query_title_cosine_similarity"], 0.0)


if __name__ == "__main__":
    unittest.main()
