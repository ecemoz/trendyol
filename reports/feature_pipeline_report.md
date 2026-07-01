# Feature Pipeline Execution Report

## Summary

FeaturePipeline was executed on the processed training dataset and wrote the
feature output to:

```text
data/processed/features.parquet
```

## Output Shape

- Row count: 489,204
- Column count: 57
- Memory usage: 255.13 MB
- Runtime: 26.74 seconds
- Missing value count: 0

## Feature Columns

- query_char_count
- query_word_count
- query_is_single_word
- query_contains_number
- query_contains_turkish_char
- query_is_lowercase
- query_token_count
- query_unique_token_count
- query_avg_token_length
- title_char_count
- title_word_count
- title_token_count
- title_unique_token_count
- title_avg_token_length
- title_contains_number
- title_contains_turkish_char
- title_is_lowercase
- category_depth
- main_category
- leaf_category
- category_level_1
- category_level_2
- category_level_3
- category_level_4
- category_level_5
- category_level_6
- category_has_multiple_levels
- brand_normalized
- brand_char_count
- brand_word_count
- brand_token_count
- brand_contains_number
- brand_contains_turkish_char
- brand_is_unknown
- brand_is_empty
- attribute_count
- attribute_key_count
- attribute_value_count
- attribute_text_length
- attribute_has_color
- attribute_has_material
- attribute_has_size
- attribute_has_pattern
- attribute_has_model
- attribute_has_brand
- attribute_is_empty
- query_title_word_overlap_count
- query_title_word_overlap_ratio
- query_title_jaccard_similarity
- query_brand_exact_match
- query_contains_brand
- query_category_word_overlap_count
- query_category_word_overlap_ratio
- query_attribute_word_overlap_count
- query_attribute_word_overlap_ratio

Preserved columns:

- label
- sample_type

## Label Distribution

| label | count |
| --- | --- |
| 0 | 239,204 |
| 1 | 250,000 |

## Sample Type Distribution

| sample_type | count |
| --- | --- |
| positive | 250,000 |
| hard | 89,208 |
| easy | 74,999 |
| medium | 74,997 |

## Validation

- `data/processed/training_dataset_with_negatives.csv` was read successfully.
- `data/processed/features.parquet` was written successfully.
- The parquet output was read back successfully.
- Output contains 489,204 rows and 57 columns.
- `label` and `sample_type` columns were preserved.
- Missing value count is 0.
- No model training, TF-IDF, BM25, or embedding generation was performed.
