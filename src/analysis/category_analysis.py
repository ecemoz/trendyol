"""Category hierarchy exploratory analysis for the items dataset."""

from collections import defaultdict

import pandas as pd

from src.analysis.common import AnalysisResult, normalize_text, percentage
from src.data.data_loader import load_items
from src.utils.logger import get_logger


logger = get_logger(__name__)

CATEGORY_COLUMN: str = "category"
TOP_CATEGORY_LIMIT: int = 20
CATEGORY_SEPARATOR: str = "/"


def _validate_category_column(items: pd.DataFrame) -> None:
    """Validate that the items dataset has a category column.

    Args:
        items: Items dataset.

    Raises:
        ValueError: If the category column is missing.
    """
    if CATEGORY_COLUMN not in items.columns:
        message = f"Missing required items column: {CATEGORY_COLUMN}"
        logger.error(message)
        raise ValueError(message)


def split_category_path(category: object) -> list[str]:
    """Split a raw category path into normalized hierarchy levels.

    Args:
        category: Raw category value such as ``aksesuar/çanta/omuz çantası``.

    Returns:
        Ordered category levels. Empty levels are removed.
    """
    normalized_category = normalize_text(category)
    if not normalized_category:
        return []

    return [
        level.strip()
        for level in normalized_category.split(CATEGORY_SEPARATOR)
        if level.strip()
    ]


def _build_category_levels_table(categories: pd.Series) -> pd.DataFrame:
    """Build a table with one row per category and extracted levels.

    Args:
        categories: Raw category series.

    Returns:
        DataFrame containing raw category, levels, depth, main, and leaf values.
    """
    paths = categories.map(split_category_path)
    max_depth = int(paths.map(len).max()) if not paths.empty else 0

    rows: list[dict[str, object]] = []
    for raw_category, path in zip(categories, paths, strict=False):
        row: dict[str, object] = {
            "category": normalize_text(raw_category),
            "depth": len(path),
            "main_category": path[0] if path else "",
            "leaf_category": path[-1] if path else "",
        }
        for level_index in range(max_depth):
            row[f"level_{level_index + 1}"] = (
                path[level_index] if level_index < len(path) else ""
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _build_top_subcategories(levels_table: pd.DataFrame) -> pd.DataFrame:
    """Build a top subcategory table from all non-main category levels.

    Args:
        levels_table: Category levels table.

    Returns:
        Top subcategories across level 2 and deeper.
    """
    level_columns = [
        column
        for column in levels_table.columns
        if column.startswith("level_") and column != "level_1"
    ]
    if not level_columns:
        return pd.DataFrame(columns=["subcategory", "count", "percentage"])

    stacked_subcategories = (
        levels_table[level_columns]
        .replace("", pd.NA)
        .stack()
        .reset_index(drop=True)
    )
    total_subcategory_count = int(stacked_subcategories.shape[0])
    table = (
        stacked_subcategories.value_counts()
        .head(TOP_CATEGORY_LIMIT)
        .rename_axis("subcategory")
        .reset_index(name="count")
    )
    table["percentage"] = table["count"].apply(
        lambda value: percentage(value, total_subcategory_count)
    )
    return table


def _build_tree_summary(levels_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize the category tree by main category.

    Args:
        levels_table: Category levels table.

    Returns:
        DataFrame with item counts, leaf counts, and depth statistics.
    """
    grouped = levels_table.groupby("main_category", dropna=False)
    summary = grouped.agg(
        item_count=("category", "size"),
        unique_leaf_count=("leaf_category", "nunique"),
        average_depth=("depth", "mean"),
        max_depth=("depth", "max"),
    )
    return summary.sort_values("item_count", ascending=False).reset_index()


def _find_deepest_category(levels_table: pd.DataFrame) -> str:
    """Find one category path with the maximum hierarchy depth.

    Args:
        levels_table: Category levels table.

    Returns:
        Category path text with maximum depth.
    """
    if levels_table.empty:
        return ""
    deepest_index = levels_table["depth"].idxmax()
    return str(levels_table.loc[deepest_index, "category"])


def _build_category_comments(metrics: dict[str, object]) -> list[str]:
    """Create short comments for the category analysis report.

    Args:
        metrics: Category analysis metrics.

    Returns:
        Human-readable interpretation comments.
    """
    return [
        (
            f"Katalogda {metrics['main_category_count']} ana kategori var; bu "
            "çeşitlilik model seçiminde kategori bazlı genelleme ihtiyacını "
            "gösterir."
        ),
        (
            f"Ortalama kategori derinliği {metrics['average_category_depth']:.2f}; "
            "derin yapı query niyetini daha ince segmentlere bağlamak için "
            "kullanışlı bağlam sağlayabilir."
        ),
        (
            f"En derin kategori {metrics['deepest_category_depth']} seviyeye "
            "sahip; hiyerarşideki uç seviyeler ürün ayrıştırmada önemli olabilir."
        ),
        (
            "Ana kategori ve alt kategori dağılımları negatif örnekleme sırasında "
            "benzer ama yanlış ürünleri seçerken dikkat edilmesi gereken alanları "
            "işaret eder."
        ),
    ]


def analyze_categories(items: pd.DataFrame) -> AnalysisResult:
    """Analyze category hierarchy in the items dataset.

    Args:
        items: Items dataset with a ``category`` column.

    Returns:
        Analysis result containing category metrics, tables, and comments.
    """
    logger.info("Starting category analysis")
    _validate_category_column(items)

    categories = items[CATEGORY_COLUMN].fillna("").astype(str)
    levels_table = _build_category_levels_table(categories)

    total_item_count = int(items.shape[0])
    main_category_counts = levels_table["main_category"].value_counts()
    depth_counts = levels_table["depth"].value_counts().sort_index()
    deepest_category_depth = int(levels_table["depth"].max()) if total_item_count else 0

    category_tree: dict[str, set[str]] = defaultdict(set)
    for _, row in levels_table.iterrows():
        path = [
            str(row[column])
            for column in levels_table.columns
            if column.startswith("level_") and str(row[column])
        ]
        for parent, child in zip(path, path[1:], strict=False):
            category_tree[parent].add(child)

    metrics: dict[str, object] = {
        "total_category_rows": total_item_count,
        "main_category_count": int(main_category_counts.shape[0]),
        "average_category_depth": float(levels_table["depth"].mean())
        if total_item_count
        else 0.0,
        "deepest_category": _find_deepest_category(levels_table),
        "deepest_category_depth": deepest_category_depth,
        "unique_full_category_count": int(levels_table["category"].nunique()),
        "category_tree_parent_count": len(category_tree),
        "category_tree_edge_count": sum(len(children) for children in category_tree.values()),
    }

    top_main_categories = (
        main_category_counts.head(TOP_CATEGORY_LIMIT)
        .rename_axis("main_category")
        .reset_index(name="count")
    )
    top_main_categories["percentage"] = top_main_categories["count"].apply(
        lambda value: percentage(value, total_item_count)
    )

    category_depth_distribution = depth_counts.rename_axis("depth").reset_index(
        name="count"
    )
    category_depth_distribution["percentage"] = category_depth_distribution[
        "count"
    ].apply(lambda value: percentage(value, total_item_count))

    tables = {
        "category_levels": levels_table,
        "top_20_main_categories": top_main_categories,
        "top_20_subcategories": _build_top_subcategories(levels_table),
        "category_depth_distribution": category_depth_distribution,
        "category_tree_summary": _build_tree_summary(levels_table),
    }

    result = AnalysisResult(
        title="Category Analizi",
        metrics=metrics,
        comments=_build_category_comments(metrics),
        tables=tables,
    )
    logger.info("Category analysis completed")
    return result


def run_category_analysis() -> AnalysisResult:
    """Load items data and run category analysis.

    Returns:
        Category analysis result.
    """
    items = load_items()
    return analyze_categories(items)


def main() -> None:
    """Run category analysis as a command-line entrypoint."""
    result = run_category_analysis()
    print(result)


if __name__ == "__main__":
    main()
