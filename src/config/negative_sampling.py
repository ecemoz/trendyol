"""Configuration values for the Sprint 3 negative sampling pipeline."""

from pathlib import Path

from src.config.paths import PROCESSED_DATA_DIR, PROJECT_ROOT


RANDOM_SEED: int = 42

NEGATIVES_PER_POSITIVE: int = 1

NEGATIVE_SAMPLING_RATIOS: dict[str, float] = {
    "easy": 0.30,
    "medium": 0.30,
    "hard": 0.40,
}

BATCH_SIZE: int = 10_000
CHUNK_SIZE: int = 100_000

HARD_CANDIDATE_POOL_SIZE: int = 50
MAX_SAMPLING_ATTEMPTS_PER_ROW: int = 10

MAX_LEXICAL_SIMILARITY_WARNING_THRESHOLD: float = 0.85
DROP_HIGH_RISK_FALSE_NEGATIVES: bool = True

NEGATIVE_SAMPLING_OUTPUT_FILE: Path = (
    PROCESSED_DATA_DIR / "training_dataset_with_negatives.csv"
)

NEGATIVE_SAMPLING_REPORTS_DIR: Path = PROJECT_ROOT / "reports"
NEGATIVE_SAMPLING_FIGURES_DIR: Path = NEGATIVE_SAMPLING_REPORTS_DIR / "figures"
NEGATIVE_SAMPLING_REPORT_FILE: Path = (
    NEGATIVE_SAMPLING_REPORTS_DIR / "sprint_3_negative_sampling_report.md"
)

SAMPLE_TYPE_POSITIVE: str = "positive"
SAMPLE_TYPE_EASY: str = "easy"
SAMPLE_TYPE_MEDIUM: str = "medium"
SAMPLE_TYPE_HARD: str = "hard"

POSITIVE_LABEL: int = 1
NEGATIVE_LABEL: int = 0


def validate_negative_sampling_config() -> None:
    """Validate negative sampling configuration values.

    Raises:
        ValueError: If ratios or numeric configuration values are invalid.
    """
    ratio_sum = sum(NEGATIVE_SAMPLING_RATIOS.values())
    if round(ratio_sum, 10) != 1.0:
        raise ValueError(
            "NEGATIVE_SAMPLING_RATIOS must sum to 1.0. "
            f"Current sum: {ratio_sum}"
        )

    if NEGATIVES_PER_POSITIVE <= 0:
        raise ValueError("NEGATIVES_PER_POSITIVE must be greater than zero.")

    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be greater than zero.")

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    if HARD_CANDIDATE_POOL_SIZE <= 0:
        raise ValueError("HARD_CANDIDATE_POOL_SIZE must be greater than zero.")

    if MAX_SAMPLING_ATTEMPTS_PER_ROW <= 0:
        raise ValueError("MAX_SAMPLING_ATTEMPTS_PER_ROW must be greater than zero.")

    if not 0.0 <= MAX_LEXICAL_SIMILARITY_WARNING_THRESHOLD <= 1.0:
        raise ValueError(
            "MAX_LEXICAL_SIMILARITY_WARNING_THRESHOLD must be between 0 and 1."
        )
