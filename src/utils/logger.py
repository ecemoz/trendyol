"""Logging utilities for the Trendyol Datathon project."""

import logging
from pathlib import Path

from src.config.paths import LOGS_DIR


DEFAULT_LOG_FILE_NAME: str = "application.log"
DEFAULT_LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or retrieve a configured logger.

    Args:
        name: Logger name, usually ``__name__`` from the caller module.
        log_file: Optional log file path. Defaults to ``logs/application.log``.
        level: Logging level for the logger and its handlers.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    resolved_log_file = log_file or LOGS_DIR / DEFAULT_LOG_FILE_NAME

    formatter = logging.Formatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )

    file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
