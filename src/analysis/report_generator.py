"""Markdown report generator for Sprint 2 exploratory data analysis."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.attribute_analysis import analyze_attributes
from src.analysis.category_analysis import analyze_categories
from src.analysis.common import AnalysisResult
from src.analysis.item_analysis import analyze_items
from src.analysis.quality_analysis import analyze_data_quality
from src.analysis.query_analysis import analyze_queries
from src.analysis.train_test_analysis import analyze_train_test_overlap
from src.analysis.training_analysis import analyze_training_pairs
from src.analysis.visualization import create_eda_visualizations
from src.config.paths import PROJECT_ROOT
from src.data.data_loader import (
    load_items,
    load_sample_submission,
    load_submission_pairs,
    load_terms,
    load_training_pairs,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
DEFAULT_REPORT_PATH: Path = REPORTS_DIR / "sprint_2_eda_report.md"
TABLE_PREVIEW_ROWS: int = 10


def _format_scalar(value: Any) -> str:
    """Format scalar values for markdown output.

    Args:
        value: Any metric value.

    Returns:
        Human-readable string representation.
    """
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _metrics_to_markdown(metrics: dict[str, Any]) -> str:
    """Convert a metric dictionary into markdown bullets.

    Args:
        metrics: Analysis metrics.

    Returns:
        Markdown bullet list.
    """
    lines = []
    for key, value in metrics.items():
        label = key.replace("_", " ")
        lines.append(f"- **{label}:** {_format_scalar(value)}")
    return "\n".join(lines)


def _escape_markdown_cell(value: Any) -> str:
    """Escape a value for safe use in a markdown table cell.

    Args:
        value: Raw table cell value.

    Returns:
        Escaped markdown cell text.
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
    """Convert a DataFrame preview into a simple markdown table.

    Args:
        dataframe: DataFrame to render.
        max_rows: Maximum number of rows to include.

    Returns:
        Markdown table text.
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


def _comments_to_markdown(comments: list[str]) -> str:
    """Convert analysis comments into markdown bullets.

    Args:
        comments: Analysis comments.

    Returns:
        Markdown bullet list.
    """
    return "\n".join(f"- {comment}" for comment in comments)


def _figure_link(path: Path) -> str:
    """Build a relative markdown image link for a figure.

    Args:
        path: Figure path.

    Returns:
        Markdown image link.
    """
    relative_path = path.relative_to(REPORTS_DIR)
    alt_text = path.stem.replace("_", " ").title()
    return f"![{alt_text}]({relative_path})"


def _section(
    title: str,
    result: AnalysisResult,
    table_names: list[str],
    figure_paths: list[Path] | None = None,
) -> str:
    """Render one analysis result as a markdown section.

    Args:
        title: Section title.
        result: Analysis result.
        table_names: Result table names to include.
        figure_paths: Optional figures to embed.

    Returns:
        Markdown section text.
    """
    lines = [f"## {title}", "", "### Metrikler", _metrics_to_markdown(result.metrics)]

    if figure_paths:
        lines.extend(["", "### Grafikler"])
        lines.extend(_figure_link(path) for path in figure_paths)

    for table_name in table_names:
        if table_name not in result.tables:
            continue
        lines.extend(
            [
                "",
                f"### {table_name.replace('_', ' ').title()}",
                _dataframe_to_markdown(result.tables[table_name]),
            ]
        )

    lines.extend(["", "### Yorum", _comments_to_markdown(result.comments), ""])
    return "\n".join(lines)


def _build_key_insights(
    query_result: AnalysisResult,
    item_result: AnalysisResult,
    attribute_result: AnalysisResult,
    training_result: AnalysisResult,
    train_test_result: AnalysisResult,
    quality_result: AnalysisResult,
) -> list[str]:
    """Build the key insight list from analysis metrics.

    Args:
        query_result: Query analysis result.
        item_result: Item analysis result.
        attribute_result: Attribute analysis result.
        training_result: Training-pair analysis result.
        train_test_result: Train-test analysis result.
        quality_result: Data quality analysis result.

    Returns:
        Key insight bullets.
    """
    return [
        (
            "Train query ve test query kümeleri ayrık görünüyor; test query "
            f"overlap oranı %{train_test_result.metrics['test_query_seen_in_train_ratio']:.2f}."
        ),
        (
            "Training pairs yalnızca pozitif label içeriyor; pozitif oran "
            f"%{training_result.metrics['positive_label_ratio']:.2f}."
        ),
        (
            "Attribute alanı güçlü bir katalog sinyali sunuyor; doluluk oranı "
            f"%{attribute_result.metrics['attribute_fill_ratio']:.2f}."
        ),
        (
            "Ürün metadata tarafındaki en büyük kalite riski gender ve age_group "
            f"alanlarında; en yüksek unknown oranı "
            f"%{item_result.metrics['highest_unknown_ratio']:.2f}."
        ),
        (
            "Query seti tamamen benzersiz görünüyor; benzersizlik oranı "
            f"%{query_result.metrics['unique_query_ratio']:.2f}."
        ),
        (
            "Genel duplicate riski düşük; tüm datasetlerde duplicate row sayısı "
            f"{quality_result.metrics['total_duplicate_rows']}."
        ),
    ]


def _build_sprint_3_recommendations() -> list[str]:
    """Build recommendations for Sprint 3.

    Returns:
        Sprint 3 recommendation bullets.
    """
    return [
        "Feature Engineering çalışmalarında query text, title, category ve attributes önceliklendirilmeli.",
        "Train/test query overlap olmadığı için id ezberine dayalı yaklaşımlardan kaçınılmalı.",
        "Negative sampling stratejisi kategori ve title benzerliğini dikkate alarak zor negatifleri kontrollü üretmeli.",
        "Gender ve age_group yüksek unknown oranları nedeniyle doğrudan kullanılmadan önce kalite etkisi ölçülmeli.",
        "Attribute anahtarları çok çeşitli olduğu için Sprint 3'te en sık ve en anlamlı anahtarlar seçilerek ilerlenmeli.",
    ]


def _build_final_answers(
    attribute_result: AnalysisResult,
    train_test_result: AnalysisResult,
) -> list[str]:
    """Build final required Q&A answers for the report.

    Args:
        attribute_result: Attribute analysis result.
        train_test_result: Train-test analysis result.

    Returns:
        Numbered answer lines.
    """
    return [
        (
            "1. Yarışmadaki en büyük veri avantajımız, title/category/attributes "
            "gibi zengin ürün içerik alanlarının bulunması ve attribute doluluk "
            f"oranının %{attribute_result.metrics['attribute_fill_ratio']:.2f} olmasıdır."
        ),
        (
            "2. En büyük veri problemi, train tarafının pozitif-only görünmesi ve "
            "test query'lerinin train'de hiç görülmemesidir."
        ),
        (
            "3. Negative Sampling yaparken aynı kategori içindeki benzer ürünlere, "
            "popüler ürün bias'ına, train/test ayrımına ve false negative riskine "
            "dikkat etmeliyiz."
        ),
        (
            "4. Feature Engineering için en değerli kolonlar query, title, category, "
            "brand ve attributes alanlarıdır."
        ),
        (
            "5. Sprint 3'e geçmeye hazırız; özellikle cold-start query yapısı ve "
            f"%{train_test_result.metrics['test_item_seen_in_train_ratio']:.2f} "
            "item overlap oranı artık net şekilde biliniyor."
        ),
    ]


def build_report_markdown(
    query_result: AnalysisResult,
    item_result: AnalysisResult,
    category_result: AnalysisResult,
    attribute_result: AnalysisResult,
    training_result: AnalysisResult,
    train_test_result: AnalysisResult,
    quality_result: AnalysisResult,
    figure_paths: dict[str, Path],
) -> str:
    """Build the complete Sprint 2 markdown report.

    Args:
        query_result: Query analysis result.
        item_result: Item analysis result.
        category_result: Category analysis result.
        attribute_result: Attribute analysis result.
        training_result: Training-pair analysis result.
        train_test_result: Train-test analysis result.
        quality_result: Data quality analysis result.
        figure_paths: Mapping of generated figure paths.

    Returns:
        Complete markdown report content.
    """
    lines = [
        "# Sprint 2 EDA ve Veri Anlama Raporu",
        "",
        "Bu rapor Sprint 2 kapsamında yalnızca exploratory data analysis amacıyla üretilmiştir. Model eğitimi, feature engineering, embedding, TF-IDF, negative sampling veya submission üretimi yapılmamıştır.",
        "",
    ]

    lines.append(
        _section(
            title="Query Analizi",
            result=query_result,
            table_names=[
                "query_length_histogram",
                "query_word_count_histogram",
                "top_words",
            ],
            figure_paths=[
                figure_paths["query_length"],
                figure_paths["query_word_count"],
            ],
        )
    )
    lines.append(
        _section(
            title="Ürün Analizi",
            result=item_result,
            table_names=[
                "top_20_brands",
                "gender_distribution",
                "age_group_distribution",
                "top_20_categories",
                "unknown_summary",
            ],
            figure_paths=[
                figure_paths["title_length"],
                figure_paths["top_20_brands"],
                figure_paths["gender_distribution"],
                figure_paths["age_group_distribution"],
            ],
        )
    )
    lines.append(
        _section(
            title="Category Analizi",
            result=category_result,
            table_names=[
                "top_20_main_categories",
                "top_20_subcategories",
                "category_depth_distribution",
                "category_tree_summary",
            ],
            figure_paths=[figure_paths["top_20_categories"]],
        )
    )
    lines.append(
        _section(
            title="Attribute Analizi",
            result=attribute_result,
            table_names=[
                "top_30_attribute_keys",
                "attribute_count_distribution",
            ],
            figure_paths=[figure_paths["top_attribute_keys"]],
        )
    )
    lines.append(
        _section(
            title="Training Pair Analizi",
            result=training_result,
            table_names=[
                "label_distribution",
                "items_per_query_distribution",
                "queries_per_item_distribution",
                "top_20_queries",
                "top_20_items",
            ],
        )
    )
    lines.append(
        _section(
            title="Train/Test Analizi",
            result=train_test_result,
            table_names=["query_overlap", "item_overlap"],
        )
    )
    lines.append(
        _section(
            title="Veri Kalitesi",
            result=quality_result,
            table_names=["dataset_quality_summary", "column_quality_summary"],
        )
    )

    lines.extend(
        [
            "## Öğrendiğimiz En Önemli İçgörüler",
            "",
            *[f"- {insight}" for insight in _build_key_insights(
                query_result=query_result,
                item_result=item_result,
                attribute_result=attribute_result,
                training_result=training_result,
                train_test_result=train_test_result,
                quality_result=quality_result,
            )],
            "",
            "## Sprint 3 İçin Öneriler",
            "",
            *[f"- {recommendation}" for recommendation in _build_sprint_3_recommendations()],
            "",
            "## Sprint Sonu Soruları",
            "",
            *_build_final_answers(
                attribute_result=attribute_result,
                train_test_result=train_test_result,
            ),
            "",
        ]
    )

    return "\n".join(lines)


def generate_sprint_2_report(
    output_path: Path = DEFAULT_REPORT_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """Run all Sprint 2 analyses and write the markdown report.

    Args:
        output_path: Target markdown report path.
        figures_dir: Directory where figures will be saved.

    Returns:
        Path of the generated markdown report.
    """
    logger.info("Generating Sprint 2 EDA report")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    terms = load_terms()
    items = load_items()
    training_pairs = load_training_pairs()
    submission_pairs = load_submission_pairs()
    sample_submission = load_sample_submission()

    query_result = analyze_queries(terms)
    item_result = analyze_items(items)
    category_result = analyze_categories(items)
    attribute_result = analyze_attributes(items)
    training_result = analyze_training_pairs(training_pairs)
    train_test_result = analyze_train_test_overlap(training_pairs, submission_pairs)
    quality_result = analyze_data_quality(
        {
            "terms": terms,
            "items": items,
            "training_pairs": training_pairs,
            "submission_pairs": submission_pairs,
            "sample_submission": sample_submission,
        }
    )

    figure_paths = create_eda_visualizations(
        terms=terms,
        items=items,
        item_result=item_result,
        category_result=category_result,
        attribute_result=attribute_result,
        output_dir=figures_dir,
    )

    report_markdown = build_report_markdown(
        query_result=query_result,
        item_result=item_result,
        category_result=category_result,
        attribute_result=attribute_result,
        training_result=training_result,
        train_test_result=train_test_result,
        quality_result=quality_result,
        figure_paths=figure_paths,
    )

    output_path.write_text(report_markdown, encoding="utf-8")
    logger.info("Sprint 2 EDA report generated at %s", output_path)
    return output_path


def main() -> None:
    """Run the Sprint 2 report generator as a command-line entrypoint."""
    report_path = generate_sprint_2_report()
    print(report_path)


if __name__ == "__main__":
    main()
