"""Unit test for E5 embedding generator."""

import unittest
import numpy as np
from src.embeddings.e5_embedding_generator import E5EmbeddingGenerator


class TestE5EmbeddingGenerator(unittest.TestCase):
    """Verify prefixing, embedding shapes, data types, and normalization."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load E5 generator once for all test methods."""
        # Using cpu to run quickly in tests
        cls.generator = E5EmbeddingGenerator(device="cpu")

    def test_query_encoding(self) -> None:
        """Verify query encoding shape, type, and L2 normalization."""
        queries = ["kırmızı elbise", "mavi pantolon"]
        embeddings = self.generator.encode_queries(queries)

        # Dimension checks: E5-base should output 768 dim
        self.assertEqual(embeddings.shape, (2, 768))
        self.assertEqual(embeddings.dtype, np.float32)

        # Norm checks: Row norms should be 1.0 because normalize_embeddings=True
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), rtol=1e-5)

    def test_passage_encoding(self) -> None:
        """Verify passage encoding shape, type, and L2 normalization."""
        passages = ["trendyol kırmızı elbise", "mavi kot pantolon"]
        embeddings = self.generator.encode_passages(passages)

        self.assertEqual(embeddings.shape, (2, 768))
        self.assertEqual(embeddings.dtype, np.float32)

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
