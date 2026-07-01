# LightGBM with TF-IDF + BM25 Features Report

## Scope

This report documents the performance of the LightGBM classifier after integrating both TF-IDF and BM25 similarity features. 
Only numeric and boolean feature columns were used. String/categorical columns were excluded. 
No CatBoost, threshold optimization, or submission generation was performed.

## Split Summary

- Train rows: 391363
- Validation rows: 97841
- Available feature columns before numeric filtering: 61
- Numeric/boolean feature columns used: 52
- Runtime: 3.44 seconds

## Metrics & Comparison

| Metric | Baseline | TF-IDF Only | With TF-IDF + BM25 | BM25 Impact | Total Change |
| --- | --- | --- | --- | --- | --- |
| accuracy | 0.884363 | 0.889596 | 0.892622 | +0.003026 | +0.008259 |
| precision | 0.879141 | 0.884385 | 0.888812 | +0.004427 | +0.009671 |
| recall | 0.897040 | 0.901860 | 0.902820 | +0.000960 | +0.005780 |
| f1 | 0.888000 | 0.893037 | 0.895761 | +0.002724 | +0.007761 |
| macro_f1 | 0.884241 | 0.889482 | 0.892524 | +0.003042 | +0.008283 |

## Confusion Matrix

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 42194 | 5647 |
| 1 | 4859 | 45141 |

## Top 20 Feature Importance

| feature | importance |
| --- | --- |
| bm25_query_category_score | 872 |
| bm25_query_title_score | 688 |
| tfidf_query_category_similarity | 646 |
| query_title_jaccard_similarity | 584 |
| tfidf_query_title_cosine_similarity | 483 |
| query_char_count | 465 |
| query_avg_token_length | 423 |
| category_depth | 422 |
| attribute_count | 417 |
| attribute_text_length | 340 |
| title_word_count | 325 |
| query_title_word_overlap_ratio | 292 |
| query_contains_brand | 265 |
| tfidf_query_attribute_similarity | 220 |
| bm25_query_attribute_score | 210 |
| query_category_word_overlap_ratio | 198 |
| query_attribute_word_overlap_ratio | 197 |
| brand_char_count | 188 |
| title_char_count | 174 |
| title_avg_token_length | 150 |

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
- tfidf_query_title_cosine_similarity
- tfidf_query_category_similarity
- tfidf_query_attribute_similarity
- bm25_query_title_score
- bm25_query_category_score
- bm25_query_attribute_score
