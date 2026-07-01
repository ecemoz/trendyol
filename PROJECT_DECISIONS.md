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
