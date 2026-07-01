"""Script to generate and cache E5 embeddings for queries and items."""

import argparse
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

from src.data.data_loader import load_items, load_terms
from src.embeddings.embedding_cache import EmbeddingCache
from src.embeddings.e5_embedding_generator import E5EmbeddingGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_passage_column(df: pd.DataFrame) -> pd.Series:
    """Build passage text representation from item fields in a vectorized way.

    Args:
        df: Items DataFrame containing title, category, brand, and attributes.

    Returns:
        Series containing space-normalized combined passage string.
    """
    # Combine title + category + brand + attributes
    passage = (
        df["title"].fillna("").astype(str).str.strip()
        + " "
        + df["category"].fillna("").astype(str).str.strip()
        + " "
        + df["brand"].fillna("").astype(str).str.strip()
        + " "
        + df["attributes"].fillna("").astype(str).str.strip()
    ).str.strip()
    # Normalize spaces
    return passage.str.replace(r"\s+", " ", regex=True)


def generate_e5_embeddings_cache(
    limit: Optional[int] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> None:
    """Load terms and items, generate E5 embeddings, and save to cache.

    Args:
        limit: Optional maximum number of unique items/queries to process (for testing).
        batch_size: Batch size used for sentence transformer encoding.
        device: CPU/GPU device identifier.
    """
    logger.info("Starting E5 embedding cache generation pipeline")
    start_time = time.time()

    cache = EmbeddingCache()

    # Determine if cache already exists
    if limit is None and cache.cache_exists():
        logger.info("Embedding cache files already exist. Skipping regeneration.")
        return

    # Initialize model generator
    generator = E5EmbeddingGenerator(device=device)

    # 1. Process Queries
    if limit is not None or not cache.query_cache_exists():
        logger.info("Processing queries for embedding cache...")
        df_terms = load_terms()
        unique_queries = df_terms["query"].dropna().unique().tolist()
        
        if limit is not None:
            unique_queries = unique_queries[:limit]

        logger.info("Found %d unique queries to encode", len(unique_queries))
        query_embeddings = generator.encode_queries(
            unique_queries,
            batch_size=batch_size,
        )
        
        # Save queries to cache
        cache.save_query_embeddings(unique_queries, query_embeddings)
    else:
        logger.info("Query embedding cache already exists. Skipping query encoding.")

    # 2. Process Items
    if limit is not None or not cache.item_cache_exists():
        logger.info("Processing items for embedding cache...")
        df_items = load_items()
        
        # Build passage column
        df_items["passage"] = build_passage_column(df_items)
        unique_passages = df_items["passage"].dropna().unique().tolist()
        
        # Filter out empty strings
        unique_passages = [p for p in unique_passages if p]
        
        if limit is not None:
            unique_passages = unique_passages[:limit]

        logger.info("Found %d unique item passages to encode", len(unique_passages))
        item_embeddings = generator.encode_passages(
            unique_passages,
            batch_size=batch_size,
        )
        
        # Save items to cache
        cache.save_item_embeddings(unique_passages, item_embeddings)
    else:
        logger.info("Item embedding cache already exists. Skipping item encoding.")

    elapsed_time = time.time() - start_time
    logger.info(
        "Embedding cache generation pipeline completed successfully in %.2f seconds",
        elapsed_time,
    )


def main() -> None:
    """Parse command line arguments and execute cache generation."""
    parser = argparse.ArgumentParser(description="Generate and cache E5 embeddings.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of unique items and queries to process (for testing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for SentenceTransformer encoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computing device (e.g. cpu, cuda, mps).",
    )
    args = parser.parse_args()

    generate_e5_embeddings_cache(
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
