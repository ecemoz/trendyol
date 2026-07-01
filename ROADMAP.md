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
| Feature Engineering | Sprint 4 | ⬜ Planned |
| Modeling | Sprint 5 | ⬜ Planned |
| Advanced Modeling | Sprint 6 | ⬜ Planned |
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

## Sprint 4

### Query Features

- [ ] Character length
- [ ] Word count
- [ ] Numeric tokens
- [ ] Brand detection
- [ ] Color detection

### Title Features

- [ ] Character length
- [ ] Word count
- [ ] Keyword extraction

### Similarity Features

- [ ] Word overlap
- [ ] Jaccard similarity
- [ ] BM25
- [ ] TF-IDF cosine similarity

### Category Features

- [ ] Main category match
- [ ] Category depth
- [ ] Leaf category match

### Attribute Features

- [ ] Color match
- [ ] Material match
- [ ] Brand match
- [ ] Model match

---

# 🤖 Phase 5 — Baseline Modeling

## Sprint 5

### Objectives

- [ ] Train/Validation split
- [ ] Cross Validation
- [ ] Logistic Regression baseline
- [ ] LightGBM baseline
- [ ] Evaluation
- [ ] Threshold optimization

Deliverable:

- First Kaggle submission

---

# 🚀 Phase 6 — Advanced Modeling

## Sprint 6

### Objectives

- [ ] Sentence Transformers
- [ ] Dense embeddings
- [ ] CatBoost
- [ ] Hybrid models
- [ ] Embedding similarity features

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
