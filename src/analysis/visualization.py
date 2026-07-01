"""Visualization utilities for Sprint 2 exploratory data analysis."""

import os
from pathlib import Path

MATPLOTLIB_CACHE_DIR = Path("/private/tmp/trendyol_matplotlib_cache")
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.common import AnalysisResult, count_words, normalize_text
from src.utils.logger import get_logger


logger = get_logger(__name__)

FIGURE_DPI: int = 140
DEFAULT_FIGURE_SIZE: tuple[float, float] = (10.0, 6.0)
BAR_FIGURE_SIZE: tuple[float, float] = (12.0, 7.0)
COLOR_PRIMARY: str = "#2f6f9f"
COLOR_SECONDARY: str = "#d9822b"
COLOR_ACCENT: str = "#4f8f6f"
GRID_ALPHA: float = 0.25


def _prepare_output_dir(output_dir: Path) -> None:
    """Create the figure output directory if needed.

    Args:
        output_dir: Directory where figures will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)


def _finalize_plot(
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path:
    """Apply common plot styling and save the active figure.

    Args:
        output_path: Target image path.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Saved figure path.
    """
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", output_path)
    return output_path


def plot_query_length_histogram(
    terms: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot query character length distribution.

    Args:
        terms: Terms dataset.
        output_dir: Figure output directory.

    Returns:
        Saved figure path.
    """
    _prepare_output_dir(output_dir)
    query_lengths = terms["query"].fillna("").astype(str).map(normalize_text).str.len()

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    plt.hist(query_lengths, bins=30, color=COLOR_PRIMARY, edgecolor="white")
    return _finalize_plot(
        output_path=output_dir / "query_length_histogram.png",
        title="Query Uzunluğu Dağılımı",
        xlabel="Karakter Sayısı",
        ylabel="Query Sayısı",
    )


def plot_query_word_count_histogram(
    terms: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot query word count distribution.

    Args:
        terms: Terms dataset.
        output_dir: Figure output directory.

    Returns:
        Saved figure path.
    """
    _prepare_output_dir(output_dir)
    word_counts = terms["query"].fillna("").astype(str).map(count_words)

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    plt.hist(
        word_counts,
        bins=range(0, int(word_counts.max()) + 2),
        color=COLOR_SECONDARY,
        edgecolor="white",
        align="left",
    )
    return _finalize_plot(
        output_path=output_dir / "query_word_count_histogram.png",
        title="Query Kelime Sayısı Dağılımı",
        xlabel="Kelime Sayısı",
        ylabel="Query Sayısı",
    )


def plot_title_length_histogram(
    items: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Plot product title character length distribution.

    Args:
        items: Items dataset.
        output_dir: Figure output directory.

    Returns:
        Saved figure path.
    """
    _prepare_output_dir(output_dir)
    title_lengths = items["title"].fillna("").astype(str).map(normalize_text).str.len()

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    plt.hist(title_lengths, bins=40, color=COLOR_ACCENT, edgecolor="white")
    return _finalize_plot(
        output_path=output_dir / "title_length_histogram.png",
        title="Ürün Title Uzunluğu Dağılımı",
        xlabel="Karakter Sayısı",
        ylabel="Ürün Sayısı",
    )


def plot_top_table_barh(
    table: pd.DataFrame,
    value_column: str,
    count_column: str,
    title: str,
    xlabel: str,
    output_path: Path,
) -> Path:
    """Plot a horizontal bar chart from a top-N table.

    Args:
        table: Table containing value and count columns.
        value_column: Column used as bar labels.
        count_column: Column used as bar values.
        title: Plot title.
        xlabel: X-axis label.
        output_path: Target image path.

    Returns:
        Saved figure path.
    """
    _prepare_output_dir(output_path.parent)
    plot_table = table[[value_column, count_column]].copy()
    plot_table = plot_table.sort_values(count_column, ascending=True)

    plt.figure(figsize=BAR_FIGURE_SIZE)
    plt.barh(
        plot_table[value_column].astype(str),
        plot_table[count_column],
        color=COLOR_PRIMARY,
    )
    return _finalize_plot(
        output_path=output_path,
        title=title,
        xlabel=xlabel,
        ylabel="",
    )


def plot_distribution_bar(
    table: pd.DataFrame,
    value_column: str,
    count_column: str,
    title: str,
    xlabel: str,
    output_path: Path,
) -> Path:
    """Plot a vertical bar chart from a distribution table.

    Args:
        table: Distribution table.
        value_column: Column used as category labels.
        count_column: Column used as bar values.
        title: Plot title.
        xlabel: X-axis label.
        output_path: Target image path.

    Returns:
        Saved figure path.
    """
    _prepare_output_dir(output_path.parent)
    plot_table = table[[value_column, count_column]].copy()

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    plt.bar(
        plot_table[value_column].astype(str),
        plot_table[count_column],
        color=COLOR_SECONDARY,
    )
    plt.xticks(rotation=35, ha="right")
    return _finalize_plot(
        output_path=output_path,
        title=title,
        xlabel=xlabel,
        ylabel="Kayıt Sayısı",
    )


def create_eda_visualizations(
    terms: pd.DataFrame,
    items: pd.DataFrame,
    item_result: AnalysisResult,
    category_result: AnalysisResult,
    attribute_result: AnalysisResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Create the standard Sprint 2 EDA visualization set.

    Args:
        terms: Terms dataset.
        items: Items dataset.
        item_result: Item analysis result.
        category_result: Category analysis result.
        attribute_result: Attribute analysis result.
        output_dir: Directory where figures will be saved.

    Returns:
        Mapping from figure key to saved path.
    """
    logger.info("Creating Sprint 2 EDA visualizations")
    _prepare_output_dir(output_dir)

    figure_paths = {
        "query_length": plot_query_length_histogram(terms, output_dir),
        "query_word_count": plot_query_word_count_histogram(terms, output_dir),
        "title_length": plot_title_length_histogram(items, output_dir),
        "top_20_brands": plot_top_table_barh(
            table=item_result.tables["top_20_brands"],
            value_column="brand",
            count_column="count",
            title="Top 20 Marka",
            xlabel="Ürün Sayısı",
            output_path=output_dir / "top_20_brands.png",
        ),
        "top_20_categories": plot_top_table_barh(
            table=category_result.tables["top_20_main_categories"],
            value_column="main_category",
            count_column="count",
            title="Top 20 Ana Kategori",
            xlabel="Ürün Sayısı",
            output_path=output_dir / "top_20_categories.png",
        ),
        "gender_distribution": plot_distribution_bar(
            table=item_result.tables["gender_distribution"],
            value_column="gender",
            count_column="count",
            title="Gender Dağılımı",
            xlabel="Gender",
            output_path=output_dir / "gender_distribution.png",
        ),
        "age_group_distribution": plot_distribution_bar(
            table=item_result.tables["age_group_distribution"],
            value_column="age_group",
            count_column="count",
            title="Age Group Dağılımı",
            xlabel="Age Group",
            output_path=output_dir / "age_group_distribution.png",
        ),
        "top_attribute_keys": plot_top_table_barh(
            table=attribute_result.tables["top_30_attribute_keys"],
            value_column="attribute_key",
            count_column="count",
            title="Top Attribute Anahtarları",
            xlabel="Kullanım Sayısı",
            output_path=output_dir / "top_attribute_keys.png",
        ),
    }

    logger.info("Sprint 2 EDA visualizations completed")
    return figure_paths
