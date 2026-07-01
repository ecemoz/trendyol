"""Embedding Cache system to manage query and item embedding storage."""

import shutil
from pathlib import Path
import numpy as np

from src.config.paths import PROJECT_ROOT
from src.embeddings.embedding_storage import (
    load_embeddings_from_parquet,
    save_embeddings_to_parquet,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_EMBEDDING_DIR: Path = PROJECT_ROOT / "data" / "embeddings"
QUERY_EMBEDDINGS_FILE: Path = DEFAULT_EMBEDDING_DIR / "query_embeddings.parquet"
ITEM_EMBEDDINGS_FILE: Path = DEFAULT_EMBEDDING_DIR / "item_embeddings.parquet"


class EmbeddingCache:
    """Manages read/write caching of query and item embeddings."""

    def __init__(self, cache_dir: Path = DEFAULT_EMBEDDING_DIR) -> None:
        """Initialize the EmbeddingCache with a specific directory.

        Args:
            cache_dir: Directory where embedding parquet files will be stored.
        """
        self.cache_dir = cache_dir
        self.query_file = cache_dir / "query_embeddings.parquet"
        self.item_file = cache_dir / "item_embeddings.parquet"

    def cache_exists(self) -> bool:
        """Check if both query and item embedding cache files exist.

        Returns:
            True if both cache files exist, False otherwise.
        """
        return self.query_file.exists() and self.item_file.exists()

    def query_cache_exists(self) -> bool:
        """Check if the query embedding cache file exists.

        Returns:
            True if the query cache file exists, False otherwise.
        """
        return self.query_file.exists()

    def item_cache_exists(self) -> bool:
        """Check if the item embedding cache file exists.

        Returns:
            True if the item cache file exists, False otherwise.
        """
        return self.item_file.exists()

    def load_query_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Load cached query embeddings from disk.

        Returns:
            Tuple of unique query strings and their corresponding embeddings.

        Raises:
            FileNotFoundError: If the cache file does not exist.
        """
        if not self.query_cache_exists():
            message = f"Query embedding cache not found at {self.query_file}"
            logger.error(message)
            raise FileNotFoundError(message)
        return load_embeddings_from_parquet(self.query_file)

    def load_item_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Load cached item embeddings from disk.

        Returns:
            Tuple of unique item strings and their corresponding embeddings.

        Raises:
            FileNotFoundError: If the cache file does not exist.
        """
        if not self.item_cache_exists():
            message = f"Item embedding cache not found at {self.item_file}"
            logger.error(message)
            raise FileNotFoundError(message)
        return load_embeddings_from_parquet(self.item_file)

    def save_query_embeddings(self, texts: list[str], embeddings: np.ndarray) -> None:
        """Save query embeddings to cache.

        Args:
            texts: List of query strings.
            embeddings: 2D numpy array of query embeddings.
        """
        save_embeddings_to_parquet(texts, embeddings, self.query_file)

    def save_item_embeddings(self, texts: list[str], embeddings: np.ndarray) -> None:
        """Save item embeddings to cache.

        Args:
            texts: List of item text representation strings.
            embeddings: 2D numpy array of item embeddings.
        """
        save_embeddings_to_parquet(texts, embeddings, self.item_file)

    def clear_cache(self) -> None:
        """Remove all cache files and the cache directory."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info("Cleared embedding cache directory: %s", self.cache_dir)
        else:
            logger.info("Cache directory %s does not exist, nothing to clear", self.cache_dir)
