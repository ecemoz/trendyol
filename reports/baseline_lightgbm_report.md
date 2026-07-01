# Baseline LightGBM Report

## Scope

This report contains the baseline LightGBM model. Only numeric and boolean feature columns were used. String/categorical columns were excluded. No CatBoost, TF-IDF, BM25, embeddings, threshold optimization, or submission generation was performed.

## Split Summary

- Train rows: 391363
- Validation rows: 97841
- Available feature columns before numeric filtering: 55
- Numeric/boolean feature columns used: 46
- Runtime: 3.27 seconds

## Label Distribution

### Train
| label | count |
| --- | --- |
| 0 | 191363 |
| 1 | 200000 |

### Validation
| label | count |
| --- | --- |
| 0 | 47841 |
| 1 | 50000 |

## Sample Type Distribution

### Train
| sample_type | count |
| --- | --- |
| easy | 60033 |
| hard | 71543 |
| medium | 59787 |
| positive | 200000 |

### Validation
| sample_type | count |
| --- | --- |
| easy | 14966 |
| hard | 17665 |
| medium | 15210 |
| positive | 50000 |

## Metrics

- accuracy: 0.884363
- precision: 0.879141
- recall: 0.897040
- f1: 0.888000
- macro_f1: 0.884241

## Confusion Matrix

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41675 | 6166 |
| 1 | 5148 | 44852 |

## Top 20 Feature Importance

| feature | importance |
| --- | --- |
| query_title_jaccard_similarity | 785 |
| query_category_word_overlap_ratio | 717 |
| query_avg_token_length | 711 |
| query_char_count | 669 |
| category_depth | 602 |
| query_title_word_overlap_ratio | 599 |
| attribute_count | 534 |
| attribute_text_length | 525 |
| query_attribute_word_overlap_ratio | 390 |
| title_word_count | 378 |
| brand_char_count | 281 |
| query_contains_brand | 272 |
| title_char_count | 256 |
| title_avg_token_length | 247 |
| query_category_word_overlap_count | 238 |
| attribute_has_material | 200 |
| title_unique_token_count | 180 |
| query_contains_turkish_char | 157 |
| brand_token_count | 126 |
| query_token_count | 123 |

## Feature Columns Used

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
- category_has_multiple_levels
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
