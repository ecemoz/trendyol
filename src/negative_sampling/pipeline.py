"""End-to-end negative sampling pipeline for Sprint 3."""

from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from src.config.negative_sampling import (
    NEGATIVE_LABEL,
    NEGATIVE_SAMPLING_OUTPUT_FILE,
    NEGATIVE_SAMPLING_RATIOS,
    NEGATIVE_SAMPLING_REPORT_FILE,
    NEGATIVES_PER_POSITIVE,
    POSITIVE_LABEL,
    RANDOM_SEED,
    SAMPLE_TYPE_EASY,
    SAMPLE_TYPE_HARD,
    SAMPLE_TYPE_MEDIUM,
    validate_negative_sampling_config,
)
from src.data.data_loader import load_items, load_terms, load_training_pairs
from src.negative_sampling.easy_sampler import EasyNegativeSampler
from src.negative_sampling.hard_sampler import HardNegativeSampler
from src.negative_sampling.medium_sampler import MediumNegativeSampler
from src.negative_sampling.report import PipelinePerformance, write_negative_sampling_report
from src.negative_sampling.sampler import (
    LABEL_COLUMN,
    SAMPLE_TYPE_COLUMN,
    SamplingResult,
    build_positive_pair_set,
    prepare_items_for_sampling,
    prepare_positive_samples,
    to_training_output,
)
from src.negative_sampling.validator import (
    ValidationReport,
    validate_negative_samples,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

BYTES_PER_MEGABYTE: int = 1024 * 1024


@dataclass(frozen=True)
class NegativeSamplingPipelineResult:
    """Container for pipeline outputs and execution metadata.

    Attributes:
        dataset: Final positive and negative training dataset.
        sampling_results: Sampling metadata by negative sample type.
        validation_report: False-negative validation summary.
        performance: Pipeline runtime and size metadata.
        output_file: Written dataset path.
        report_file: Written markdown report path.
    """

    dataset: pd.DataFrame
    sampling_results: dict[str, SamplingResult]
    validation_report: ValidationReport
    performance: PipelinePerformance
    output_file: object
    report_file: object


def calculate_sampling_targets(
    positive_count: int,
    negatives_per_positive: int = NEGATIVES_PER_POSITIVE,
    sampling_ratios: dict[str, float] = NEGATIVE_SAMPLING_RATIOS,
) -> dict[str, int]:
    """Calculate target negative counts by sample type.

    Args:
        positive_count: Number of positive rows.
        negatives_per_positive: Number of negatives to target per positive.
        sampling_ratios: Sampling ratio by negative type.

    Returns:
        Integer target counts by sample type.
    """
    total_negative_count = positive_count * negatives_per_positive
    raw_targets = {
        sample_type: total_negative_count * ratio
        for sample_type, ratio in sampling_ratios.items()
    }
    targets = {
        sample_type: int(raw_target)
        for sample_type, raw_target in raw_targets.items()
    }

    remainder = total_negative_count - sum(targets.values())
    if remainder <= 0:
        return targets

    fractional_order = sorted(
        raw_targets,
        key=lambda sample_type: raw_targets[sample_type] - targets[sample_type],
        reverse=True,
    )
    for sample_type in fractional_order[:remainder]:
        targets[sample_type] += 1

    return targets


def _combine_validation_reports(reports: list[ValidationReport]) -> ValidationReport:
    """Combine multiple validation reports into one summary.

    Args:
        reports: Validation reports to combine.

    Returns:
        Aggregated validation report.
    """
    return ValidationReport(
        input_count=sum(report.input_count for report in reports),
        output_count=sum(report.output_count for report in reports),
        known_positive_pair_count=sum(
            report.known_positive_pair_count for report in reports
        ),
        same_item_count=sum(report.same_item_count for report in reports),
        exact_query_title_match_count=sum(
            report.exact_query_title_match_count for report in reports
        ),
        high_risk_similarity_count=sum(
            report.high_risk_similarity_count for report in reports
        ),
        dropped_high_risk_count=sum(report.dropped_high_risk_count for report in reports),
    )


def _memory_usage_mb(dataframe: pd.DataFrame) -> float:
    """Calculate DataFrame memory usage in megabytes.

    Args:
        dataframe: DataFrame to inspect.

    Returns:
        Deep memory usage in megabytes.
    """
    return float(dataframe.memory_usage(deep=True).sum() / BYTES_PER_MEGABYTE)


def run_negative_sampling_pipeline(
    terms: pd.DataFrame,
    items: pd.DataFrame,
    training_pairs: pd.DataFrame,
) -> NegativeSamplingPipelineResult:
    """Run the negative sampling pipeline from loaded raw datasets.

    Args:
        terms: Raw terms dataset.
        items: Raw items dataset.
        training_pairs: Raw training pairs dataset.

    Returns:
        Negative sampling pipeline result.
    """
    logger.info("Starting negative sampling pipeline")
    start_time = perf_counter()
    validate_negative_sampling_config()

    prepared_items = prepare_items_for_sampling(items)
    positive_samples = prepare_positive_samples(
        training_pairs=training_pairs,
        terms=terms,
        prepared_items=prepared_items,
    )
    positive_pair_set = build_positive_pair_set(positive_samples)
    positive_count = int(positive_samples.shape[0])
    sampling_targets = calculate_sampling_targets(positive_count=positive_count)
    logger.info("Sampling targets calculated: %s", sampling_targets)

    samplers = {
        SAMPLE_TYPE_EASY: EasyNegativeSampler(
            items=prepared_items,
            random_seed=RANDOM_SEED,
        ),
        SAMPLE_TYPE_MEDIUM: MediumNegativeSampler(
            items=prepared_items,
            random_seed=RANDOM_SEED + 1,
        ),
        SAMPLE_TYPE_HARD: HardNegativeSampler(
            items=prepared_items,
            random_seed=RANDOM_SEED + 2,
        ),
    }

    sampling_results: dict[str, SamplingResult] = {}
    validated_negative_frames: list[pd.DataFrame] = []
    validation_reports: list[ValidationReport] = []

    for sample_type, sampler in samplers.items():
        target_count = sampling_targets.get(sample_type, 0)
        sampling_result = sampler.sample(
            positive_samples=positive_samples,
            target_count=target_count,
        )
        sampling_results[sample_type] = sampling_result

        validated_negatives, validation_report = validate_negative_samples(
            negatives=sampling_result.samples,
            positive_pair_set=positive_pair_set,
        )
        validated_negative_frames.append(validated_negatives)
        validation_reports.append(validation_report)

        logger.info(
            "%s validation summary | input=%s | output=%s",
            sample_type,
            validation_report.input_count,
            validation_report.output_count,
        )

    combined_validation_report = _combine_validation_reports(validation_reports)
    negative_samples = (
        pd.concat(validated_negative_frames, ignore_index=True)
        if validated_negative_frames
        else pd.DataFrame()
    )

    final_samples = pd.concat(
        [
            positive_samples,
            negative_samples,
        ],
        ignore_index=True,
    )
    output_dataset = to_training_output(final_samples)
    output_dataset[LABEL_COLUMN] = output_dataset[LABEL_COLUMN].astype(int)

    positive_output_count = int((output_dataset[LABEL_COLUMN] == POSITIVE_LABEL).sum())
    negative_output_count = int((output_dataset[LABEL_COLUMN] == NEGATIVE_LABEL).sum())
    elapsed_seconds = perf_counter() - start_time
    memory_usage_mb = _memory_usage_mb(output_dataset)

    performance = PipelinePerformance(
        elapsed_seconds=elapsed_seconds,
        memory_usage_mb=memory_usage_mb,
        positive_count=positive_output_count,
        negative_count=negative_output_count,
        total_count=int(output_dataset.shape[0]),
    )

    NEGATIVE_SAMPLING_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_dataset.to_csv(NEGATIVE_SAMPLING_OUTPUT_FILE, index=False)
    logger.info(
        "Negative sampling dataset written | path=%s | rows=%s | memory=%.3f MB",
        NEGATIVE_SAMPLING_OUTPUT_FILE,
        output_dataset.shape[0],
        memory_usage_mb,
    )

    report_file = write_negative_sampling_report(
        dataset=output_dataset,
        sampling_results=sampling_results,
        validation_report=combined_validation_report,
        performance=performance,
        report_path=NEGATIVE_SAMPLING_REPORT_FILE,
    )

    logger.info(
        "Negative sampling pipeline completed | elapsed=%.3fs | positives=%s | "
        "negatives=%s | total=%s",
        elapsed_seconds,
        positive_output_count,
        negative_output_count,
        output_dataset.shape[0],
    )

    return NegativeSamplingPipelineResult(
        dataset=output_dataset,
        sampling_results=sampling_results,
        validation_report=combined_validation_report,
        performance=performance,
        output_file=NEGATIVE_SAMPLING_OUTPUT_FILE,
        report_file=report_file,
    )


def run_negative_sampling_pipeline_from_raw() -> NegativeSamplingPipelineResult:
    """Load raw datasets and run the full negative sampling pipeline.

    Returns:
        Negative sampling pipeline result.
    """
    logger.info("Loading raw datasets for negative sampling")
    terms = load_terms()
    items = load_items()
    training_pairs = load_training_pairs()
    return run_negative_sampling_pipeline(
        terms=terms,
        items=items,
        training_pairs=training_pairs,
    )


def main() -> None:
    """Run the negative sampling pipeline as a command-line entrypoint."""
    result = run_negative_sampling_pipeline_from_raw()
    print(result.output_file)
    print(result.report_file)


if __name__ == "__main__":
    main()
