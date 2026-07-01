"""Item-level exploratory data analysis for the products dataset."""

import pandas as pd

from src.analysis.common import (
    AnalysisResult,
    build_value_counts_table,
    is_unknown_like,
    normalize_text,
    percentage,
)
from src.data.data_loader import load_items
from src.utils.logger import get_logger


logger = get_logger(__name__)

ITEM_ID_COLUMN: str = "item_id"
TITLE_COLUMN: str = "title"
CATEGORY_COLUMN: str = "category"
BRAND_COLUMN: str = "brand"
GENDER_COLUMN: str = "gender"
AGE_GROUP_COLUMN: str = "age_group"
ATTRIBUTES_COLUMN: str = "attributes"
TOP_DISTRIBUTION_LIMIT: int = 20


def _validate_items_columns(items: pd.DataFrame) -> None:
    """Validate that the items dataset has the required columns.

    Args:
        items: Items dataset.

    Raises:
        ValueError: If a required column is missing.
    """
    required_columns = {
        ITEM_ID_COLUMN,
        TITLE_COLUMN,
        CATEGORY_COLUMN,
        BRAND_COLUMN,
        GENDER_COLUMN,
        AGE_GROUP_COLUMN,
        ATTRIBUTES_COLUMN,
    }
    missing_columns = required_columns.difference(items.columns)

    if missing_columns:
        message = f"Missing required items columns: {sorted(missing_columns)}"
        logger.error(message)
        raise ValueError(message)


def _unknown_summary(items: pd.DataFrame) -> pd.DataFrame:
    """Build unknown-like value ratios for key item columns.

    Args:
        items: Items dataset.

    Returns:
        DataFrame with unknown counts and percentages by column.
    """
    columns = [
        TITLE_COLUMN,
        CATEGORY_COLUMN,
        BRAND_COLUMN,
        GENDER_COLUMN,
        AGE_GROUP_COLUMN,
        ATTRIBUTES_COLUMN,
    ]
    total_item_count = int(items.shape[0])
    rows: list[dict[str, object]] = []

    for column in columns:
        unknown_count = int(items[column].map(is_unknown_like).sum())
        rows.append(
            {
                "column": column,
                "unknown_count": unknown_count,
                "unknown_percentage": percentage(unknown_count, total_item_count),
            }
        )

    return pd.DataFrame(rows)


def _get_extreme_title(
    titles: pd.Series,
    title_lengths: pd.Series,
    find_longest: bool,
) -> str:
    """Get the shortest or longest title.

    Args:
        titles: Product title series.
        title_lengths: Character length per title.
        find_longest: Whether to return the longest title.

    Returns:
        Extreme title text. Returns an empty string for empty inputs.
    """
    if titles.empty:
        return ""

    index = title_lengths.idxmax() if find_longest else title_lengths.idxmin()
    return str(titles.loc[index])


def _build_item_comments(metrics: dict[str, object]) -> list[str]:
    """Create short comments for the item analysis report.

    Args:
        metrics: Item analysis metrics.

    Returns:
        Human-readable interpretation comments.
    """
    brand_coverage_ratio = metrics["brand_coverage_ratio"]
    average_title_length = metrics["average_title_length"]
    highest_unknown_column = metrics["highest_unknown_column"]
    highest_unknown_ratio = metrics["highest_unknown_ratio"]

    return [
        (
            f"Marka doluluk oranı %{brand_coverage_ratio:.2f}; marka bilgisinin "
            "katalog içinde ne kadar güvenilir sinyal olabileceğini gösterir."
        ),
        (
            f"Ortalama title uzunluğu {average_title_length:.2f} karakter; ürün "
            "başlıkları query ile eşleşme niyetini anlamak için temel metin "
            "kaynağıdır."
        ),
        (
            f"En yüksek unknown oranı {highest_unknown_column} kolonunda "
            f"%{highest_unknown_ratio:.2f}; veri temizliği ve modelleme öncesi "
            "bu alan ayrıca izlenmelidir."
        ),
        (
            "Gender, age group ve category dağılımları katalog segmentlerinin "
            "dengeli olup olmadığını anlamak için kritik bağlam sağlar."
        ),
    ]


def analyze_items(items: pd.DataFrame) -> AnalysisResult:
    """Analyze the items dataset without changing the source data.

    Args:
        items: Items dataset with catalog metadata columns.

    Returns:
        Analysis result containing item metrics, tables, and comments.
    """
    logger.info("Starting item analysis")
    _validate_items_columns(items)

    total_item_count = int(items.shape[0])
    titles = items[TITLE_COLUMN].fillna("").astype(str)
    normalized_titles = titles.map(normalize_text)
    title_lengths = normalized_titles.str.len()

    normalized_brands = items[BRAND_COLUMN].map(normalize_text)
    non_unknown_brands = normalized_brands[~normalized_brands.map(is_unknown_like)]
    unique_brand_count = int(non_unknown_brands.nunique(dropna=True))
    brand_unknown_count = int(normalized_brands.map(is_unknown_like).sum())

    unknown_table = _unknown_summary(items)
    highest_unknown_row = unknown_table.sort_values(
        "unknown_percentage",
        ascending=False,
    ).iloc[0]

    metrics: dict[str, object] = {
        "total_item_count": total_item_count,
        "unique_brand_count": unique_brand_count,
        "brand_unknown_count": brand_unknown_count,
        "brand_unknown_ratio": percentage(brand_unknown_count, total_item_count),
        "brand_coverage_ratio": percentage(
            total_item_count - brand_unknown_count,
            total_item_count,
        ),
        "average_title_length": float(title_lengths.mean())
        if total_item_count
        else 0.0,
        "shortest_title": _get_extreme_title(
            titles=titles,
            title_lengths=title_lengths,
            find_longest=False,
        ),
        "shortest_title_length": int(title_lengths.min()) if total_item_count else 0,
        "longest_title": _get_extreme_title(
            titles=titles,
            title_lengths=title_lengths,
            find_longest=True,
        ),
        "longest_title_length": int(title_lengths.max()) if total_item_count else 0,
        "highest_unknown_column": str(highest_unknown_row["column"]),
        "highest_unknown_ratio": float(highest_unknown_row["unknown_percentage"]),
    }

    tables = {
        "brand_distribution": build_value_counts_table(
            normalized_brands,
            value_column="brand",
        ),
        "top_20_brands": build_value_counts_table(
            normalized_brands,
            value_column="brand",
            top_n=TOP_DISTRIBUTION_LIMIT,
        ),
        "gender_distribution": build_value_counts_table(
            items[GENDER_COLUMN].map(normalize_text),
            value_column="gender",
        ),
        "age_group_distribution": build_value_counts_table(
            items[AGE_GROUP_COLUMN].map(normalize_text),
            value_column="age_group",
        ),
        "category_distribution": build_value_counts_table(
            items[CATEGORY_COLUMN].map(normalize_text),
            value_column="category",
        ),
        "top_20_categories": build_value_counts_table(
            items[CATEGORY_COLUMN].map(normalize_text),
            value_column="category",
            top_n=TOP_DISTRIBUTION_LIMIT,
        ),
        "unknown_summary": unknown_table,
    }

    result = AnalysisResult(
        title="Ürün Analizi",
        metrics=metrics,
        comments=_build_item_comments(metrics),
        tables=tables,
    )
    logger.info("Item analysis completed")
    return result


def run_item_analysis() -> AnalysisResult:
    """Load items data and run item analysis.

    Returns:
        Item analysis result.
    """
    items = load_items()
    return analyze_items(items)


def main() -> None:
    """Run item analysis as a command-line entrypoint."""
    result = run_item_analysis()
    print(result)


if __name__ == "__main__":
    main()
