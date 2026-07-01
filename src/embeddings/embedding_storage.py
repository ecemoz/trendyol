"""Storage utility for loading and saving sentence embeddings to disk."""

from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_embeddings_to_parquet(
    texts: list[str],
    embeddings: np.ndarray,
    output_path: Path,
) -> None:
    """Save text and corresponding embedding vectors to a parquet file.

    Args:
        texts: List of input strings.
        embeddings: 2D numpy array of shape (num_texts, embedding_dim).
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Store embedding arrays as lists of floats in a single column
    df = pd.DataFrame({
        "text": texts,
        "embedding": list(embeddings)
    })
    
    df.to_parquet(output_path, index=False)
    logger.info("Saved %d embeddings to %s", len(texts), output_path)


def load_embeddings_from_parquet(
    file_path: Path,
) -> tuple[list[str], np.ndarray]:
    """Load text and corresponding embedding vectors from a parquet file.

    Args:
        file_path: Source file path.

    Returns:
        Tuple containing list of texts and 2D numpy array of embeddings.
    """
    df = pd.read_parquet(file_path)
    texts = df["text"].tolist()
    embeddings = np.vstack(df["embedding"].values).astype(np.float32)
    logger.info(
        "Loaded %d embeddings of dimension %d from %s",
        len(texts),
        embeddings.shape[1],
        file_path,
    )
    return texts, embeddings
