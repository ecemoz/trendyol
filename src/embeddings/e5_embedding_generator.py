"""E5 Embedding Generator module using intfloat/multilingual-e5-base."""

from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_NAME: str = "intfloat/multilingual-e5-base"
DEFAULT_BATCH_SIZE: int = 32


class E5EmbeddingGenerator:
    """Generates normalized dense embeddings using the Multilingual E5 model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the sentence-transformer model.

        Args:
            model_name: Hugging Face model identifier.
            device: Computing device ('cpu', 'cuda', 'mps'). Auto-selects if None.
        """
        self.model_name = model_name
        logger.info(
            "Loading SentenceTransformer model: %s on device: %s",
            model_name,
            device or "auto",
        )
        self.model = SentenceTransformer(model_name, device=device)
        self.device = self.model.device
        logger.info("Model loaded successfully on device: %s", self.device)

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> np.ndarray:
        """Encode search queries with E5 query prefix.

        Args:
            queries: List of query strings.
            batch_size: Batch size for encoding.

        Returns:
            Normalized 2D float32 numpy array of query embeddings.
        """
        prefixed_queries = [f"query: {query}" for query in queries]
        logger.info("Encoding %d queries with batch_size=%d", len(queries), batch_size)
        
        embeddings = self.model.encode(
            prefixed_queries,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_passages(
        self,
        passages: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> np.ndarray:
        """Encode catalog items/passages with E5 passage prefix.

        Args:
            passages: List of passage/text strings.
            batch_size: Batch size for encoding.

        Returns:
            Normalized 2D float32 numpy array of passage embeddings.
        """
        prefixed_passages = [f"passage: {passage}" for passage in passages]
        logger.info("Encoding %d passages with batch_size=%d", len(passages), batch_size)
        
        embeddings = self.model.encode(
            prefixed_passages,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
