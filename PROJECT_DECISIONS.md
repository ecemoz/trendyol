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

---

## 2026-07-01
Decision:
Use Logistic Regression as the first baseline model on the numeric and boolean
feature subset.

Result:
The baseline model was trained on 46 numeric and boolean features selected from
the 55 available feature columns. The validation split produced 0.843509 macro
F1.

Reason:
Logistic Regression provides a fast and interpretable baseline that is useful
for verifying that the prepared feature table carries predictive signal before
moving to stronger models.

Status:
Completed.

---

## 2026-07-01
Decision:
Use LightGBM as the main baseline model for Sprint 5.

Result:
The LightGBM baseline achieved 0.884241 macro F1 on the validation split.
The same numeric and boolean feature subset was used, with no new features or
parameter family changes beyond the baseline setup.

Reason:
LightGBM is a strong tabular baseline for mixed feature sets and provides a
clear performance jump over Logistic Regression while remaining practical for
iterative experimentation.

Status:
Completed.

---

## 2026-07-01
Decision:
Use Stratified 5-Fold Cross Validation to check the stability of the LightGBM
baseline.

Result:
The cross-validation mean macro F1 was 0.882508 with a standard deviation of
0.000789 across folds.

Reason:
The validation split alone is not enough to judge baseline stability, so a
stratified cross-validation pass helps confirm that the LightGBM result is not
driven by a lucky split.

Status:
Completed.

---

## 2026-07-01
Decision:
Keep the LightGBM decision threshold at 0.50 after threshold analysis.

Result:
Threshold scanning showed the best macro F1 at 0.50, matching the default
threshold and confirming that threshold tuning does not improve the baseline
score on the validation split.

Reason:
The threshold scan was run to check whether the validation metric could be
improved without retraining. Since the default threshold already matches the
best observed macro F1, no adjustment is needed for the current baseline.

Status:
Completed.
