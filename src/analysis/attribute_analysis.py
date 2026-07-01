"""Attribute exploratory analysis for the items dataset."""

from collections import Counter

import pandas as pd

from src.analysis.common import AnalysisResult, is_unknown_like, normalize_text, percentage
from src.data.data_loader import load_items
from src.utils.logger import get_logger


logger = get_logger(__name__)

ATTRIBUTES_COLUMN: str = "attributes"
ATTRIBUTE_SEPARATOR: str = ","
KEY_VALUE_SEPARATOR: str = ":"
TOP_ATTRIBUTE_LIMIT: int = 30


def _validate_attributes_column(items: pd.DataFrame) -> None:
    """Validate that the items dataset has an attributes column.

    Args:
        items: Items dataset.

    Raises:
        ValueError: If the attributes column is missing.
    """
    if ATTRIBUTES_COLUMN not in items.columns:
        message = f"Missing required items column: {ATTRIBUTES_COLUMN}"
        logger.error(message)
        raise ValueError(message)


def extract_attribute_keys(attributes: object) -> list[str]:
    """Extract normalized attribute keys from a raw attributes cell.

    Args:
        attributes: Raw attributes text such as ``renk: siyah, materyal: pamuk``.

    Returns:
        Ordered list of normalized attribute keys found in the cell.
    """
    if is_unknown_like(attributes):
        return []

    keys: list[str] = []
    for attribute_part in str(attributes).split(ATTRIBUTE_SEPARATOR):
        if KEY_VALUE_SEPARATOR not in attribute_part:
            continue

        key = normalize_text(attribute_part.split(KEY_VALUE_SEPARATOR, maxsplit=1)[0])
        if key:
            keys.append(key)

    return keys


def _build_attribute_key_table(attribute_keys: pd.Series) -> pd.DataFrame:
    """Build a frequency table for attribute keys.

    Args:
        attribute_keys: Series where each row contains a list of keys.

    Returns:
        Attribute key frequency table.
    """
    counter: Counter[str] = Counter()
    for keys in attribute_keys:
        counter.update(keys)

    table = pd.DataFrame(
        counter.most_common(TOP_ATTRIBUTE_LIMIT),
        columns=["attribute_key", "count"],
    )

    total_key_count = int(sum(counter.values()))
    if not table.empty:
        table["percentage"] = table["count"].apply(
            lambda value: percentage(value, total_key_count)
        )

    return table


def _build_attribute_count_distribution(attribute_counts: pd.Series) -> pd.DataFrame:
    """Build a distribution table for attribute counts per item.

    Args:
        attribute_counts: Number of parsed attribute keys per item.

    Returns:
        Attribute count distribution table.
    """
    total_item_count = int(attribute_counts.shape[0])
    table = (
        attribute_counts.value_counts()
        .sort_index()
        .rename_axis("attribute_count")
        .reset_index(name="item_count")
    )
    table["percentage"] = table["item_count"].apply(
        lambda value: percentage(value, total_item_count)
    )
    return table


def _build_attribute_comments(metrics: dict[str, object]) -> list[str]:
    """Create short comments for the attribute analysis report.

    Args:
        metrics: Attribute analysis metrics.

    Returns:
        Human-readable interpretation comments.
    """
    return [
        (
            f"Attribute doluluk oranı %{metrics['attribute_fill_ratio']:.2f}; "
            "ürün açıklamasının title dışındaki yapılandırılmış sinyal gücünü "
            "gösterir."
        ),
        (
            f"Ortalama attribute sayısı {metrics['average_attribute_count']:.2f}; "
            "katalogda ürünlerin ne kadar detaylı tanımlandığını anlamamızı "
            "sağlar."
        ),
        (
            f"Toplam {metrics['unique_attribute_key_count']} benzersiz attribute "
            "anahtarı var; bu çeşitlilik ileride kontrollü feature engineering "
            "tasarımı gerektirebilir."
        ),
        (
            "Renk, materyal, beden, garanti ve uyumlu model gibi attribute "
            "anahtarları query niyetiyle doğrudan eşleşebileceği için Sprint 3 "
            "kararlarında güçlü aday alanlardır."
        ),
    ]


def analyze_attributes(items: pd.DataFrame) -> AnalysisResult:
    """Analyze the attributes column in the items dataset.

    Args:
        items: Items dataset with an ``attributes`` column.

    Returns:
        Analysis result containing attribute metrics, tables, and comments.
    """
    logger.info("Starting attribute analysis")
    _validate_attributes_column(items)

    attributes = items[ATTRIBUTES_COLUMN]
    total_item_count = int(items.shape[0])
    attribute_keys = attributes.map(extract_attribute_keys)
    attribute_counts = attribute_keys.map(len)
    non_empty_attribute_count = int((attribute_counts > 0).sum())

    all_keys = [key for keys in attribute_keys for key in keys]
    unique_attribute_key_count = len(set(all_keys))

    metrics: dict[str, object] = {
        "total_item_count": total_item_count,
        "items_with_attributes": non_empty_attribute_count,
        "items_without_attributes": total_item_count - non_empty_attribute_count,
        "attribute_fill_ratio": percentage(non_empty_attribute_count, total_item_count),
        "average_attribute_count": float(attribute_counts.mean())
        if total_item_count
        else 0.0,
        "maximum_attribute_count": int(attribute_counts.max()) if total_item_count else 0,
        "minimum_attribute_count": int(attribute_counts.min()) if total_item_count else 0,
        "unique_attribute_key_count": unique_attribute_key_count,
        "total_parsed_attribute_key_count": int(len(all_keys)),
    }

    tables = {
        "top_30_attribute_keys": _build_attribute_key_table(attribute_keys),
        "attribute_count_distribution": _build_attribute_count_distribution(
            attribute_counts
        ),
    }

    result = AnalysisResult(
        title="Attribute Analizi",
        metrics=metrics,
        comments=_build_attribute_comments(metrics),
        tables=tables,
    )
    logger.info("Attribute analysis completed")
    return result


def run_attribute_analysis() -> AnalysisResult:
    """Load items data and run attribute analysis.

    Returns:
        Attribute analysis result.
    """
    items = load_items()
    return analyze_attributes(items)


def main() -> None:
    """Run attribute analysis as a command-line entrypoint."""
    result = run_attribute_analysis()
    print(result)


if __name__ == "__main__":
    main()
