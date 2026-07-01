# 🚀 Trendyol Datathon 2026 Roadmap

> Goal: Build a production-ready, explainable and high-performing machine learning solution for the Trendyol Datathon 2026.

---

# 📌 Current Progress

| Phase | Sprint | Status |
|--------|--------|--------|
| Foundation | Sprint 1 | ✅ Completed |
| Data Understanding | Sprint 2 | ✅ Completed |
| Dataset Construction | Sprint 3A | ✅ Completed |
| Dataset Construction | Sprint 3B | ✅ Completed |
| Feature Engineering | Sprint 4 | ✅ Completed |
| Modeling | Sprint 5 | ✅ Completed |
| Advanced Modeling | Sprint 6 | ✅ Completed |
| Optimization | Sprint 7 | ⬜ Planned |
| Explainability | Sprint 8 | ⬜ Planned |
| Deployment | Sprint 9 | ⬜ Planned |
| Final Submission | Sprint 10 | ⬜ Planned |

---

# 🏗 Phase 1 — Foundation

## Sprint 1 ✅

### Objectives

- [x] Create project structure
- [x] Configure project settings
- [x] Build reusable Data Loader
- [x] Configure logging
- [x] Git & GitHub integration
- [x] Read all competition datasets

### Deliverables

- Project skeleton
- Config system
- Data loading pipeline

---

# 📊 Phase 2 — Data Understanding

## Sprint 2 ✅

### Objectives

- [x] Query analysis
- [x] Item analysis
- [x] Category analysis
- [x] Attribute analysis
- [x] Data quality analysis
- [x] Train/Test comparison
- [x] EDA report

### Key Findings

- Query overlap between train and test is 0%.
- Training data contains only positive labels.
- Attributes have very high coverage (~98%).
- Category hierarchy is deep and informative.
- Product metadata is rich enough for feature engineering.

---

# 🧩 Phase 3 — Dataset Construction

## Sprint 3A ✅

### Objectives

- [x] Negative Sampling Architecture
- [x] Easy Sampler
- [x] Medium Sampler
- [x] Hard Sampler
- [x] Sampling Validator
- [x] Pipeline
- [x] Smoke Tests

---

## Sprint 3B ✅

### Objectives

- [x] Execute pipeline on the full dataset
- [x] Generate negative samples
- [x] Validate generated dataset
- [x] False Negative analysis
- [x] Sampling distribution report
- [x] Memory & runtime profiling
- [x] Export training dataset

### Results

- Positive samples: 250,000
- Negative samples: 239,204
- Total samples: 489,204
- Easy negatives: 74,999
- Medium negatives: 74,997
- Hard negatives: 89,208
- Removed known positive pairs: 9,775
- Removed exact query-title matches: 1,471
- Removed high lexical similarity samples: 1,540
- Runtime: 45 seconds
- Memory usage: ~590 MB

### Deliverables

- `data/processed/training_dataset_with_negatives.csv`
- `reports/sprint_3_negative_sampling_report.md`
- `reports/figures/negative_sample_type_distribution.png`
- `reports/figures/negative_label_distribution.png`
- `reports/figures/negative_category_distribution.png`

---

# 🧠 Phase 4 — Feature Engineering

## Sprint 4 ✅

### Objectives

- [x] Query feature extraction
- [x] Title feature extraction
- [x] Category feature extraction
- [x] Brand feature extraction
- [x] Attribute feature extraction
- [x] Simple lexical similarity feature extraction
- [x] Feature pipeline
- [x] Execute feature pipeline on processed training dataset
- [x] Export feature dataset
- [x] Feature pipeline report

### Results

- Feature rows: 489,204
- Feature columns: 57
- Missing values: 0
- Feature parquet memory usage: 255.13 MB
- Feature pipeline runtime: 26.74 seconds

### Deliverables

- `src/features/query_features.py`
- `src/features/title_features.py`
- `src/features/category_features.py`
- `src/features/brand_features.py`
- `src/features/attribute_features.py`
- `src/features/similarity_features.py`
- `src/features/feature_pipeline.py`
- `data/processed/features.parquet`
- `reports/feature_pipeline_report.md`

### Guardrails

- [x] No model training
- [x] No TF-IDF
- [x] No BM25
- [x] No embeddings
- [x] No submission generation

---

# 🤖 Phase 5 — Baseline Modeling

## Sprint 5 ✅

### Objectives

- [x] Train/Validation split
- [x] Cross Validation
- [x] Logistic Regression baseline
- [x] LightGBM baseline
- [x] Evaluation
- [x] Threshold optimization

Deliverable:

### Results

- Logistic Regression macro F1: 0.843509
- LightGBM macro F1: 0.884241
- LightGBM 5-fold CV mean macro F1: 0.882508
- LightGBM 5-fold CV macro F1 std: 0.000789
- Best threshold: 0.50

### Deliverables

- `reports/baseline_logistic_report.md`
- `reports/baseline_lightgbm_report.md`
- `reports/cross_validation_report.md`
- `reports/threshold_optimization_report.md`
- `reports/figures/threshold_vs_macro_f1.png`

### Guardrails

- [x] No new features introduced
- [x] No TF-IDF
- [x] No BM25
- [x] No embeddings
- [x] No CatBoost or neural models
- [x] No submission generation

---

# 🚀 Phase 6 — Advanced Modeling

## Sprint 6 ✅

### Objectives

- [x] Create TF-IDF Similarity Features (query ↔ title, query ↔ category, query ↔ attributes)
- [x] Create BM25 Similarity Features (query ↔ title, query ↔ category, query ↔ attributes)
- [x] Integrate advanced similarity features into the feature pipeline
- [x] Evaluate LightGBM modeling performance with TF-IDF + BM25 features
- [x] Run Stratified 5-Fold Cross Validation to ensure generalization performance
- [ ] Sentence Transformers
- [ ] Dense embeddings
- [ ] CatBoost
- [ ] Hybrid models
- [ ] Embedding similarity features

### Results

- Combined dataset shape: 489,204 rows, 63 columns (61 features)
- LightGBM Macro F1 score: 0.892524 (an increase of +0.008283 over baseline)
- LightGBM 5-fold CV mean Macro F1: 0.890496 (an increase of +0.007988 over baseline CV mean)
- Top 3 predictive features: `bm25_query_category_score`, `bm25_query_title_score`, `tfidf_query_category_similarity`

### Deliverables

- `src/features/tfidf_features.py`
- `src/features/bm25_features.py`
- `src/models/train_lightgbm_tfidf.py`
- `src/models/train_lightgbm_tfidf_bm25.py`
- `src/models/cross_validation_tfidf.py`
- `src/models/cross_validation_tfidf_bm25.py`
- `data/processed/features_with_tfidf.parquet`
- `data/processed/features_with_tfidf_bm25.parquet`
- `reports/lightgbm_tfidf_report.md`
- `reports/lightgbm_tfidf_bm25_report.md`
- `reports/cross_validation_tfidf_report.md`
- `reports/cross_validation_tfidf_bm25_report.md`

---

# 📈 Phase 7 — Optimization

## Sprint 7

### Objectives

- [ ] Hyperparameter tuning
- [ ] Ensemble
- [ ] Error Analysis
- [ ] Threshold tuning
- [ ] Cross-validation improvements

---

# 🔍 Phase 8 — Explainability

## Sprint 8

### Objectives

- [ ] SHAP
- [ ] Feature importance
- [ ] Prediction analysis
- [ ] Explainability report

---

# 🌐 Phase 9 — Deployment

## Sprint 9

### Objectives

- [ ] FastAPI inference service
- [ ] Docker
- [ ] Batch prediction pipeline
- [ ] API documentation

---

# 🏁 Phase 10 — Final Submission

## Sprint 10

### Objectives

- [ ] Final model
- [ ] Final report
- [ ] Final presentation
- [ ] Kaggle submission
- [ ] Hackathon preparation

---

# 🎯 Final Goal

Develop a machine learning solution that is:

- High-performing on Macro F1
- Explainable
- Production-ready
- Memory efficient
- Fast at inference
- Easy to maintain
- Suitable for the Trendyol Hackathon finals

---

# 📝 Notes

This roadmap is a living document.

After completing each sprint:

- Update the status.
- Record major architectural decisions.
- Commit changes to Git.
