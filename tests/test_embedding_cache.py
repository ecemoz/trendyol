"""Unit test for embedding storage and cache manager."""

import unittest
from pathlib import Path
import numpy as np

from src.config.paths import PROJECT_ROOT
from src.embeddings.embedding_cache import EmbeddingCache


class TestEmbeddingCache(unittest.TestCase):
    """Verify loading, saving, and cache checking functionality."""

    def setUp(self) -> None:
        """Set up a temporary cache directory under data/interim/."""
        self.test_cache_dir = PROJECT_ROOT / "data" / "interim" / "test_embedding_cache"
        self.cache = EmbeddingCache(cache_dir=self.test_cache_dir)
        # Ensure clean state
        self.cache.clear_cache()

    def tearDown(self) -> None:
        """Clean up temporary files."""
        self.cache.clear_cache()

    def test_cache_workflow(self) -> None:
        """Test end-to-end caching flow for queries and items."""
        # Initial state checks
        self.assertFalse(self.cache.query_cache_exists())
        self.assertFalse(self.cache.item_cache_exists())
        self.assertFalse(self.cache.cache_exists())

        # Create dummy data
        queries = ["kırmızı elbise", "mavi pantolon"]
        query_embs = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            dtype=np.float32,
        )

        items = ["trendyol kırmızı elbise", "mavi kot pantolon"]
        item_embs = np.array(
            [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]],
            dtype=np.float32,
        )

        # Save queries
        self.cache.save_query_embeddings(queries, query_embs)
        self.assertTrue(self.cache.query_cache_exists())
        self.assertFalse(self.cache.cache_exists())  # item cache still missing

        # Save items
        self.cache.save_item_embeddings(items, item_embs)
        self.assertTrue(self.cache.item_cache_exists())
        self.assertTrue(self.cache.cache_exists())  # both present now

        # Load back queries and assert correctness
        loaded_queries, loaded_query_embs = self.cache.load_query_embeddings()
        self.assertEqual(loaded_queries, queries)
        np.testing.assert_allclose(loaded_query_embs, query_embs, rtol=1e-5)

        # Load back items and assert correctness
        loaded_items, loaded_item_embs = self.cache.load_item_embeddings()
        self.assertEqual(loaded_items, items)
        np.testing.assert_allclose(loaded_item_embs, item_embs, rtol=1e-5)

        # Clear cache and verify deletion
        self.cache.clear_cache()
        self.assertFalse(self.cache.query_cache_exists())
        self.assertFalse(self.cache.item_cache_exists())
        self.assertFalse(self.cache.cache_exists())


if __name__ == "__main__":
    unittest.main()
