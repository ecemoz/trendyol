"""Feature pipeline that combines existing feature extractors."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config.paths import PROCESSED_DATA_DIR
from src.features.attribute_features import AttributeFeatureExtractor
from src.features.brand_features import BrandFeatureExtractor
from src.features.category_features import CategoryFeatureExtractor
from src.features.query_features import QueryFeatureExtractor
from src.features.similarity_features import SimilarityFeatureExtractor
from src.features.tfidf_features import TfidfSimilarityFeatureExtractor
from src.features.title_features import TitleFeatureExtractor
from src.utils.logger import get_logger


logger = get_logger(__name__)

DEFAULT_INPUT_FILE: Path = PROCESSED_DATA_DIR / "training_dataset_with_negatives.csv"
DEFAULT_OUTPUT_FILE: Path = PROCESSED_DATA_DIR / "features.parquet"
LABEL_COLUMN: str = "label"
SAMPLE_TYPE_COLUMN: str = "sample_type"


@dataclass(frozen=True)
class FeaturePipeline:
    """Run all existing feature extractors and persist the feature matrix.

    Attributes:
        input_file: Processed training dataset path.
        output_file: Feature parquet output path.
    """

    input_file: Path = DEFAULT_INPUT_FILE
    output_file: Path = DEFAULT_OUTPUT_FILE

    def run(self) -> pd.DataFrame:
        """Load the processed training dataset, build features, and save output.

        Returns:
            Generated feature DataFrame.
        """
        logger.info("Starting feature pipeline")
        dataframe = self.load_input()
        features = self.transform(dataframe)
        self.save(features)
        logger.info("Feature pipeline completed")
        return features

    def load_input(self) -> pd.DataFrame:
        """Load the processed training dataset.

        Returns:
            Processed training dataset.

        Raises:
            FileNotFoundError: If the configured input file does not exist.
        """
        if not self.input_file.exists():
            message = f"Processed training dataset not found: {self.input_file}"
            logger.error(message)
            raise FileNotFoundError(message)

        logger.info("Reading processed training dataset from %s", self.input_file)
        return pd.read_csv(self.input_file)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Run all existing feature extractors on an input DataFrame.

        Args:
            dataframe: Processed training dataset containing text fields, label,
                and sample type columns.

        Returns:
            Combined feature DataFrame preserving ``label`` and ``sample_type``.

        Raises:
            ValueError: If required output columns are missing.
        """
        self._validate_output_columns(dataframe)
        logger.info("Running query feature extractor")
        query_features = QueryFeatureExtractor().transform(dataframe)

        logger.info("Running title feature extractor")
        title_features = TitleFeatureExtractor().transform(dataframe)

        logger.info("Running category feature extractor")
        category_features = CategoryFeatureExtractor().transform(dataframe)

        logger.info("Running brand feature extractor")
        brand_features = BrandFeatureExtractor().transform(dataframe)

        logger.info("Running attribute feature extractor")
        attribute_features = AttributeFeatureExtractor().transform(dataframe)

        logger.info("Running similarity feature extractor")
        similarity_features = SimilarityFeatureExtractor().transform(dataframe)

        logger.info("Running TF-IDF similarity feature extractor")
        tfidf_features = TfidfSimilarityFeatureExtractor().fit_transform(dataframe)

        preserved_columns = dataframe[[LABEL_COLUMN, SAMPLE_TYPE_COLUMN]].copy()
        features = pd.concat(
            [
                query_features,
                title_features,
                category_features,
                brand_features,
                attribute_features,
                similarity_features,
                tfidf_features,
                preserved_columns,
            ],
            axis=1,
        )
        logger.info("Feature extraction completed | shape=%s", features.shape)
        return features

    def save(self, features: pd.DataFrame) -> None:
        """Save generated features as a parquet file.

        Args:
            features: Feature DataFrame to persist.
        """
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing features to %s", self.output_file)
        features.to_parquet(self.output_file, index=False)

    @staticmethod
    def _validate_output_columns(dataframe: pd.DataFrame) -> None:
        """Validate columns that must be preserved in the feature output.

        Args:
            dataframe: Input DataFrame.

        Raises:
            ValueError: If ``label`` or ``sample_type`` is missing.
        """
        required_columns = {LABEL_COLUMN, SAMPLE_TYPE_COLUMN}
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            message = f"Missing required pipeline columns: {sorted(missing_columns)}"
            logger.error(message)
            raise ValueError(message)


def run_feature_pipeline() -> pd.DataFrame:
    """Run the default feature pipeline.

    Returns:
        Generated feature DataFrame.
    """
    return FeaturePipeline().run()


def main() -> None:
    """Run the feature pipeline as a command-line entrypoint."""
    run_feature_pipeline()


if __name__ == "__main__":
    main()
