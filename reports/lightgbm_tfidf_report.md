# LightGBM with TF-IDF Features Report

## Scope

This report documents the performance of the LightGBM classifier after integrating TF-IDF similarity features. 
Only numeric and boolean feature columns were used. String/categorical columns were excluded. 
No CatBoost, BM25, embeddings, threshold optimization, or submission generation was performed.

## Split Summary

- Train rows: 391363
- Validation rows: 97841
- Available feature columns before numeric filtering: 58
- Numeric/boolean feature columns used: 49
- Runtime: 3.28 seconds

## Metrics & Comparison

| Metric | Baseline | With TF-IDF | Change |
| --- | --- | --- | --- |
| accuracy | 0.884363 | 0.889596 | +0.005233 |
| precision | 0.879141 | 0.884385 | +0.005244 |
| recall | 0.897040 | 0.901860 | +0.004820 |
| f1 | 0.888000 | 0.893037 | +0.005037 |
| macro_f1 | 0.884241 | 0.889482 | +0.005241 |

## Confusion Matrix

| actual \ predicted | 0 | 1 |
| --- | --- | --- |
| 0 | 41946 | 5895 |
| 1 | 4907 | 45093 |

## Top 20 Feature Importance

| feature | importance |
| --- | --- |
| tfidf_query_category_similarity | 995 |
| tfidf_query_title_cosine_similarity | 764 |
| query_title_jaccard_similarity | 661 |
| query_avg_token_length | 579 |
| query_char_count | 578 |
| attribute_count | 485 |
| category_depth | 460 |
| attribute_text_length | 398 |
| query_title_word_overlap_ratio | 345 |
| tfidf_query_attribute_similarity | 335 |
| title_word_count | 325 |
| query_category_word_overlap_ratio | 287 |
| query_contains_brand | 283 |
| brand_char_count | 241 |
| query_attribute_word_overlap_ratio | 235 |
| title_avg_token_length | 209 |
| title_char_count | 190 |
| title_unique_token_count | 170 |
| query_category_word_overlap_count | 146 |
| query_token_count | 130 |

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
