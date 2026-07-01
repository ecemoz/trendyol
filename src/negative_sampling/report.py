"""Reporting utilities for the Sprint 3 negative sampling pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MATPLOTLIB_CACHE_DIR = Path("/private/tmp/trendyol_matplotlib_cache")
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config.negative_sampling import (
    NEGATIVE_SAMPLING_FIGURES_DIR,
    NEGATIVE_SAMPLING_REPORT_FILE,
    NEGATIVE_SAMPLING_RATIOS,
)
from src.negative_sampling.sampler import (
    CATEGORY_COLUMN,
    LABEL_COLUMN,
    SAMPLE_TYPE_COLUMN,
    SamplingResult,
)
from src.negative_sampling.validator import ValidationReport
from src.utils.logger import get_logger


logger = get_logger(__name__)

FIGURE_DPI: int = 140
DEFAULT_FIGURE_SIZE: tuple[float, float] = (10.0, 6.0)
BAR_FIGURE_SIZE: tuple[float, float] = (12.0, 7.0)
COLOR_PRIMARY: str = "#2f6f9f"
COLOR_SECONDARY: str = "#d9822b"
COLOR_ACCENT: str = "#4f8f6f"
GRID_ALPHA: float = 0.25
TABLE_PREVIEW_ROWS: int = 12


@dataclass(frozen=True)
class PipelinePerformance:
    """Runtime and size metadata for negative sampling execution.

    Attributes:
        elapsed_seconds: Total pipeline runtime.
        memory_usage_mb: Output dataset memory usage in megabytes.
        positive_count: Number of positive rows in the final dataset.
        negative_count: Number of negative rows in the final dataset.
        total_count: Total number of rows in the final dataset.
    """

    elapsed_seconds: float
    memory_usage_mb: float
    positive_count: int
    negative_count: int
    total_count: int


def _format_value(value: Any) -> str:
    """Format a value for markdown output.

    Args:
        value: Raw value.

    Returns:
        Readable string representation.
    """
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _escape_markdown_cell(value: Any) -> str:
    """Escape a table cell for markdown rendering.

    Args:
        value: Raw table cell value.

    Returns:
        Escaped cell text.
    """
    if isinstance(value, float):
        text = str(int(value)) if value.is_integer() else f"{value:.2f}"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _dataframe_to_markdown(
    dataframe: pd.DataFrame,
    max_rows: int = TABLE_PREVIEW_ROWS,
) -> str:
    """Render a DataFrame preview as markdown.

    Args:
        dataframe: DataFrame to render.
        max_rows: Maximum rows to include.

    Returns:
        Markdown table.
    """
    if dataframe.empty:
        return "_Tablo boş._"

    preview = dataframe.head(max_rows).copy()
    headers = [_escape_markdown_cell(column) for column in preview.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in preview.iterrows():
        values = [_escape_markdown_cell(value) for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")

    if len(dataframe) > max_rows:
        lines.append(f"\n_İlk {max_rows} satır gösterildi; toplam {len(dataframe)} satır._")

    return "\n".join(lines)


def _value_counts_table(
    dataframe: pd.DataFrame,
    column: str,
    value_column: str,
    count_column: str = "count",
    top_n: int | None = None,
) -> pd.DataFrame:
    """Build a value-counts table with percentages.

    Args:
        dataframe: Source DataFrame.
        column: Column to count.
        value_column: Output column for values.
        count_column: Output column for counts.
        top_n: Optional number of rows to keep.

    Returns:
        Value-counts table.
    """
    counts = dataframe[column].value_counts(dropna=False)
    if top_n is not None:
        counts = counts.head(top_n)

    table = counts.rename_axis(value_column).reset_index(name=count_column)
    total_count = int(dataframe.shape[0])
    table["percentage"] = table[count_column] / total_count * 100.0
    return table


def _save_bar_plot(
    table: pd.DataFrame,
    value_column: str,
    count_column: str,
    title: str,
    xlabel: str,
    output_path: Path,
    horizontal: bool = False,
) -> Path:
    """Save a bar plot from a count table.

    Args:
        table: Count table.
        value_column: Column containing labels.
        count_column: Column containing counts.
        title: Figure title.
        xlabel: X-axis label.
        output_path: Target figure path.
        horizontal: Whether to draw a horizontal bar chart.

    Returns:
        Saved figure path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_table = table[[value_column, count_column]].copy()
    if horizontal:
        plot_table = plot_table.sort_values(count_column, ascending=True)
        plt.figure(figsize=BAR_FIGURE_SIZE)
        plt.barh(
            plot_table[value_column].astype(str),
            plot_table[count_column],
            color=COLOR_PRIMARY,
        )
        plt.xlabel(xlabel)
        plt.ylabel("")
    else:
        plt.figure(figsize=DEFAULT_FIGURE_SIZE)
        plt.bar(
            plot_table[value_column].astype(str),
            plot_table[count_column],
            color=COLOR_SECONDARY,
        )
        plt.xlabel(value_column.replace("_", " ").title())
        plt.ylabel(xlabel)
        plt.xticks(rotation=35, ha="right")

    plt.title(title, fontsize=14, pad=12)
    plt.grid(axis="y", alpha=GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved negative sampling figure: %s", output_path)
    return output_path


def create_negative_sampling_figures(
    dataset: pd.DataFrame,
    output_dir: Path = NEGATIVE_SAMPLING_FIGURES_DIR,
) -> dict[str, Path]:
    """Create standard negative sampling report figures.

    Args:
        dataset: Final positive and negative training dataset.
        output_dir: Directory where figures will be saved.

    Returns:
        Mapping from figure key to saved path.
    """
    logger.info("Creating negative sampling figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_type_table = _value_counts_table(
        dataset,
        column=SAMPLE_TYPE_COLUMN,
        value_column=SAMPLE_TYPE_COLUMN,
    )
    label_table = _value_counts_table(
        dataset,
        column=LABEL_COLUMN,
        value_column=LABEL_COLUMN,
    )
    category_table = _value_counts_table(
        dataset,
        column=CATEGORY_COLUMN,
        value_column=CATEGORY_COLUMN,
        top_n=20,
    )

    return {
        "sample_type_distribution": _save_bar_plot(
            table=sample_type_table,
            value_column=SAMPLE_TYPE_COLUMN,
            count_column="count",
            title="Sample Type Dağılımı",
            xlabel="Kayıt Sayısı",
            output_path=output_dir / "negative_sample_type_distribution.png",
        ),
        "label_distribution": _save_bar_plot(
            table=label_table,
            value_column=LABEL_COLUMN,
            count_column="count",
            title="Label Dağılımı",
            xlabel="Kayıt Sayısı",
            output_path=output_dir / "negative_label_distribution.png",
        ),
        "category_distribution": _save_bar_plot(
            table=category_table,
            value_column=CATEGORY_COLUMN,
            count_column="count",
            title="Top 20 Kategori Dağılımı",
            xlabel="Kayıt Sayısı",
            output_path=output_dir / "negative_category_distribution.png",
            horizontal=True,
        ),
    }


def _relative_figure_link(path: Path, report_path: Path) -> str:
    """Build a markdown image link relative to a report path.

    Args:
        path: Figure path.
        report_path: Markdown report path.

    Returns:
        Markdown image link.
    """
    relative_path = os.path.relpath(path, start=report_path.parent)
    alt_text = path.stem.replace("_", " ").title()
    return f"![{alt_text}]({relative_path})"


def _sampling_results_table(
    sampling_results: dict[str, SamplingResult],
) -> pd.DataFrame:
    """Convert sampling results into a markdown-friendly table.

    Args:
        sampling_results: Mapping of sample type to sampling result.

    Returns:
        Sampling result summary table.
    """
    rows = []
    for sample_type, result in sampling_results.items():
        rows.append(
            {
                "sample_type": sample_type,
                "requested_count": result.requested_count,
                "generated_before_validation": result.generated_count,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
    return pd.DataFrame(rows)


def _validation_report_table(report: ValidationReport) -> pd.DataFrame:
    """Convert validation metadata into a one-row table.

    Args:
        report: Validation report.

    Returns:
        Validation summary table.
    """
    return pd.DataFrame(
        [
            {
                "input_count": report.input_count,
                "output_count": report.output_count,
                "known_positive_pair_count": report.known_positive_pair_count,
                "same_item_count": report.same_item_count,
                "exact_query_title_match_count": report.exact_query_title_match_count,
                "high_risk_similarity_count": report.high_risk_similarity_count,
                "dropped_high_risk_count": report.dropped_high_risk_count,
            }
        ]
    )


def build_negative_sampling_report_markdown(
    dataset: pd.DataFrame,
    sampling_results: dict[str, SamplingResult],
    validation_report: ValidationReport,
    performance: PipelinePerformance,
    figure_paths: dict[str, Path],
    report_path: Path = NEGATIVE_SAMPLING_REPORT_FILE,
) -> str:
    """Build markdown content for the Sprint 3 report.

    Args:
        dataset: Final positive and negative training dataset.
        sampling_results: Sampling result metadata by sample type.
        validation_report: False-negative validation summary.
        performance: Pipeline runtime and size metadata.
        figure_paths: Generated figure paths.
        report_path: Target report path, used for relative figure links.

    Returns:
        Markdown report content.
    """
    sample_type_distribution = _value_counts_table(
        dataset,
        column=SAMPLE_TYPE_COLUMN,
        value_column=SAMPLE_TYPE_COLUMN,
    )
    label_distribution = _value_counts_table(
        dataset,
        column=LABEL_COLUMN,
        value_column=LABEL_COLUMN,
    )
    category_distribution = _value_counts_table(
        dataset,
        column=CATEGORY_COLUMN,
        value_column=CATEGORY_COLUMN,
        top_n=20,
    )

    lines = [
        "# Sprint 3 Negative Sampling Raporu",
        "",
        "Bu rapor yalnızca negative sampling pipeline çıktısını açıklar. Model eğitimi, TF-IDF, embedding, Sentence Transformer, LightGBM, CatBoost, XGBoost veya submission üretimi yapılmamıştır.",
        "",
        "## Sampling Stratejisi",
        "",
        "Sampling oranları config üzerinden yönetilir:",
        "",
        *[
            f"- `{sample_type}`: %{ratio * 100:.2f}"
            for sample_type, ratio in NEGATIVE_SAMPLING_RATIOS.items()
        ],
        "",
        "Üretilen pozitif ve negatif örneklerde final kolon seti `query`, `item_id`, `title`, `category`, `brand`, `attributes`, `label`, `sample_type` şeklindedir.",
        "",
        "### Sampling Sonuçları",
        _dataframe_to_markdown(_sampling_results_table(sampling_results)),
        "",
        "## Easy Sampling",
        "",
        "Easy negative örnekler, pozitif ürünün ana kategorisinden farklı ana kategoriden seçilir. Bu grup false-negative riski düşük ama model için ayrıştırıcı temel kontrast sağlar.",
        "",
        "## Medium Sampling",
        "",
        "Medium negative örnekler, aynı ana kategori içinde ancak farklı ilk alt kategoriden seçilir. Bu grup kategori bağlamını koruyarak daha gerçekçi negatifler üretir.",
        "",
        "## Hard Sampling",
        "",
        "Hard negative örnekler, aynı full kategori içinde query-title lexical overlap skoru yüksek ürünlerden seçilir. TF-IDF veya embedding kullanılmaz; yalnızca token Jaccard benzerliği sampling amacıyla hesaplanır.",
        "",
        "## False Negative Analizi",
        "",
        _dataframe_to_markdown(_validation_report_table(validation_report)),
        "",
        "Validator bilinen pozitif pair'leri, aynı item seçimlerini, query-title birebir eşleşmelerini ve yüksek lexical similarity risklerini kontrol eder.",
        "",
        "## Veri Dağılımı",
        "",
        _relative_figure_link(figure_paths["sample_type_distribution"], report_path),
        "",
        _dataframe_to_markdown(sample_type_distribution),
        "",
        _relative_figure_link(figure_paths["label_distribution"], report_path),
        "",
        _dataframe_to_markdown(label_distribution),
        "",
        _relative_figure_link(figure_paths["category_distribution"], report_path),
        "",
        _dataframe_to_markdown(category_distribution),
        "",
        "## Pipeline Performansı",
        "",
        f"- Toplam süre: {_format_value(performance.elapsed_seconds)} saniye",
        f"- Bellek kullanımı: {_format_value(performance.memory_usage_mb)} MB",
        f"- Pozitif örnek sayısı: {performance.positive_count}",
        f"- Negatif örnek sayısı: {performance.negative_count}",
        f"- Toplam örnek sayısı: {performance.total_count}",
        "",
        "## Sprint 4 İçin Hazırlık",
        "",
        "- Sprint 4'te modelleme öncesi bu negatiflerin dağılımı tekrar kontrol edilmeli.",
        "- Hard negative kayıtlarında false-negative riski izlenmeye devam edilmeli.",
        "- Feature Engineering aşamasında `query`, `title`, `category`, `brand` ve `attributes` alanları öncelikli değerlendirilmeli.",
        "- Eğitim/validasyon ayrımı yapılırken aynı query veya aynı ürün sızıntısı riski ayrıca incelenmeli.",
        "",
    ]
    return "\n".join(lines)


def write_negative_sampling_report(
    dataset: pd.DataFrame,
    sampling_results: dict[str, SamplingResult],
    validation_report: ValidationReport,
    performance: PipelinePerformance,
    report_path: Path = NEGATIVE_SAMPLING_REPORT_FILE,
    figures_dir: Path = NEGATIVE_SAMPLING_FIGURES_DIR,
) -> Path:
    """Create figures and write the Sprint 3 markdown report.

    Args:
        dataset: Final positive and negative training dataset.
        sampling_results: Sampling result metadata by sample type.
        validation_report: False-negative validation summary.
        performance: Pipeline runtime and size metadata.
        report_path: Target report path.
        figures_dir: Target figure directory.

    Returns:
        Generated report path.
    """
    logger.info("Writing negative sampling report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = create_negative_sampling_figures(
        dataset=dataset,
        output_dir=figures_dir,
    )
    report_markdown = build_negative_sampling_report_markdown(
        dataset=dataset,
        sampling_results=sampling_results,
        validation_report=validation_report,
        performance=performance,
        figure_paths=figure_paths,
        report_path=report_path,
    )
    report_path.write_text(report_markdown, encoding="utf-8")
    logger.info("Negative sampling report written to %s", report_path)
    return report_path
