"""Attribute-based feature extraction utilities."""

from dataclasses import dataclass

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)

DEFAULT_ATTRIBUTES_COLUMN: str = "attributes"
ATTRIBUTE_SEPARATOR: str = ","
KEY_VALUE_SEPARATOR: str = ":"
UNKNOWN_ATTRIBUTE_VALUES: frozenset[str] = frozenset(
    {
        "unknown",
        "unk",
        "none",
        "null",
        "nan",
        "na",
        "n/a",
        "belirsiz",
        "bilinmiyor",
    }
)
COLOR_KEYS: frozenset[str] = frozenset({"renk", "color", "color detail"})
MATERIAL_KEYS: frozenset[str] = frozenset(
    {"materyal", "material", "kumaş", "kumas", "kumaş tipi", "materyal bileşeni"}
)
SIZE_KEYS: frozenset[str] = frozenset({"beden", "size", "ölçü", "olcu", "numara"})
PATTERN_KEYS: frozenset[str] = frozenset({"desen", "pattern"})
MODEL_KEYS: frozenset[str] = frozenset(
    {"model", "uyumlu model", "compatible model", "model kodu"}
)
BRAND_KEYS: frozenset[str] = frozenset({"marka", "brand"})


@dataclass(frozen=True)
class AttributeFeatureExtractor:
    """Extract reusable attribute-only features from item attributes.

    Attributes:
        attributes_column: Name of the column containing raw attributes text.
        output_prefix: Prefix used for generated feature column names.
    """

    attributes_column: str = DEFAULT_ATTRIBUTES_COLUMN
    output_prefix: str = "attribute"

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract attribute-based features from a DataFrame.

        Args:
            dataframe: Input DataFrame containing the configured attributes column.

        Returns:
            DataFrame containing only generated attribute feature columns.

        Raises:
            ValueError: If the configured attributes column is missing.
        """
        if self.attributes_column not in dataframe.columns:
            message = f"Missing attributes column: {self.attributes_column}"
            logger.error(message)
            raise ValueError(message)

        logger.info(
            "Extracting attribute features from column: %s",
            self.attributes_column,
        )
        attributes = dataframe[self.attributes_column].map(self._normalize_text)
        parsed_attributes = attributes.map(self._parse_attributes)
        attribute_keys = parsed_attributes.map(lambda values: list(values.keys()))
        attribute_values = parsed_attributes.map(list)

        features = pd.DataFrame(index=dataframe.index)
        features[f"{self.output_prefix}_count"] = parsed_attributes.map(len)
        features[f"{self.output_prefix}_key_count"] = attribute_keys.map(
            lambda keys: len(set(keys))
        )
        features[f"{self.output_prefix}_value_count"] = attribute_values.map(
            lambda values: sum(1 for value in values if value)
        )
        features[f"{self.output_prefix}_text_length"] = attributes.str.len()
        features[f"{self.output_prefix}_has_color"] = attribute_keys.map(
            lambda keys: self._has_any_key(keys, COLOR_KEYS)
        ).astype(int)
        features[f"{self.output_prefix}_has_material"] = attribute_keys.map(
            lambda keys: self._has_any_key(keys, MATERIAL_KEYS)
        ).astype(int)
        features[f"{self.output_prefix}_has_size"] = attribute_keys.map(
            lambda keys: self._has_any_key(keys, SIZE_KEYS)
        ).astype(int)
        features[f"{self.output_prefix}_has_pattern"] = attribute_keys.map(
            lambda keys: self._has_any_key(keys, PATTERN_KEYS)
        ).astype(int)
        features[f"{self.output_prefix}_has_model"] = attribute_keys.map(
            lambda keys: self._has_any_key(keys, MODEL_KEYS)
        ).astype(int)
        features[f"{self.output_prefix}_has_brand"] = attribute_keys.map(
            lambda keys: self._has_any_key(keys, BRAND_KEYS)
        ).astype(int)
        features[f"{self.output_prefix}_is_empty"] = attributes.map(
            self._is_empty_attribute_text
        ).astype(int)

        logger.info("Attribute feature extraction completed")
        return features

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract attribute features with a scikit-learn-like interface.

        Args:
            dataframe: Input DataFrame containing the configured attributes column.

        Returns:
            DataFrame containing only generated attribute feature columns.
        """
        return self.transform(dataframe)

    @staticmethod
    def _normalize_text(value: object) -> str:
        """Normalize raw attribute text with trim and lowercase.

        Args:
            value: Raw attribute value.

        Returns:
            Normalized text. Missing values become an empty string.
        """
        if pd.isna(value):
            return ""
        return str(value).strip().lower()

    @classmethod
    def _parse_attributes(cls, attributes: str) -> dict[str, str]:
        """Parse normalized attributes text into key-value pairs.

        Args:
            attributes: Normalized attributes text.

        Returns:
            Parsed attribute dictionary. Malformed parts are skipped safely.
        """
        if cls._is_empty_attribute_text(attributes):
            return {}

        parsed: dict[str, str] = {}
        for attribute_part in attributes.split(ATTRIBUTE_SEPARATOR):
            if KEY_VALUE_SEPARATOR not in attribute_part:
                continue

            raw_key, raw_value = attribute_part.split(KEY_VALUE_SEPARATOR, maxsplit=1)
            key = raw_key.strip().lower()
            value = raw_value.strip().lower()
            if key:
                parsed[key] = value

        return parsed

    @staticmethod
    def _has_any_key(keys: list[str], target_keys: frozenset[str]) -> bool:
        """Check whether any parsed key matches a target key set.

        Args:
            keys: Parsed normalized attribute keys.
            target_keys: Target normalized keys.

        Returns:
            True when at least one parsed key is in the target set.
        """
        return any(key in target_keys for key in keys)

    @staticmethod
    def _is_empty_attribute_text(attributes: str) -> bool:
        """Check whether normalized attributes text is empty or unknown-like.

        Args:
            attributes: Normalized attributes text.

        Returns:
            True when text is empty or a known unknown placeholder.
        """
        return attributes == "" or attributes in UNKNOWN_ATTRIBUTE_VALUES
