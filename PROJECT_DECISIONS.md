# Project Decisions

## 2026-07-01
Decision:
Use category-aware hard negative sampling.

Reason:
EDA showed a rich category hierarchy and a positive-only training set.

Status:
Implemented.

---

## 2026-07-01
Decision:
Use the generated Sprint 3B negative sampling dataset as the prepared training
dataset for the next project phase.

Result:
The negative sampling pipeline was executed on the full competition training
data. The final dataset contains 250,000 positive samples and 239,204 negative
samples, for a total of 489,204 samples.

Sampling distribution:
- Easy negatives: 74,999
- Medium negatives: 74,997
- Hard negatives: 89,208

False negative protection:
- Removed known positive pairs: 9,775
- Removed exact query-title matches: 1,471
- Removed high lexical similarity samples: 1,540

Performance:
- Runtime: 45 seconds
- Memory usage: approximately 590 MB

Output:
- `data/processed/training_dataset_with_negatives.csv`
- `reports/sprint_3_negative_sampling_report.md`

Reason:
The training data is positive-only, so a controlled negative sampling dataset is
required before feature engineering and model training. The final distribution
keeps easy, medium and hard negatives while removing likely false negatives.

Status:
Completed.

---

## 2026-07-01
Decision:
Use the Sprint 4 feature parquet as the prepared feature dataset for baseline
modeling.

Result:
The feature pipeline was executed on
`data/processed/training_dataset_with_negatives.csv` and wrote the output to
`data/processed/features.parquet`.

Feature coverage:
- Query features
- Title features
- Category features
- Brand features
- Attribute features
- Simple lexical similarity features

Output summary:
- Rows: 489,204
- Columns: 57
- Missing values: 0
- Memory usage: 255.13 MB
- Runtime: 26.74 seconds

Guardrails:
- No model training was performed.
- No TF-IDF features were created.
- No BM25 features were created.
- No embeddings were created.
- No submission file was generated.

Reason:
The project needs a reusable, model-agnostic feature table before baseline
modeling. Sprint 4 intentionally keeps the features simple and interpretable so
the next sprint can focus on validation strategy and baseline model selection.

Status:
Completed.
