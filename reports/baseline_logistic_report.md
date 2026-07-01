# Baseline Logistic Regression Report

## Scope

This report contains the first baseline Logistic Regression model. Only numeric and boolean feature columns were used. String/categorical columns were excluded. No LightGBM, CatBoost, TF-IDF, BM25, embeddings, threshold optimization, or submission generation was performed.

## Split Summary

- Train rows: 391363
- Validation rows: 97841
- Available feature columns before numeric filtering: 55
- Numeric/boolean feature columns used: 46
- Runtime: 12.11 seconds

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

- accuracy: 0.843511
- precision: 0.859404
- recall: 0.829480
- f1: 0.844177
- macro_f1: 0.843509

## Confusion Matrix

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41056 | 6785 |
| 1 | 8526 | 41474 |

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
