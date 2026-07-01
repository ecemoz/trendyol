# Trendyol Datathon 2026 - Query Product Relevance

This repository contains the production-oriented machine learning project
foundation for the Trendyol Datathon 2026 query-product relevance task.

The final objective of the project is to predict whether a given
`(query, product)` pair is relevant or not as a binary classification problem.

Sprint 1 is limited to project infrastructure and raw data validation. Sprint 2
adds exploratory data analysis and data understanding. No model development is
included in these sprints.

## Sprint 1 Scope

- Create a clean and maintainable machine learning project structure.
- Centralize filesystem paths with `pathlib`.
- Add reusable logging infrastructure.
- Add raw CSV data loaders.
- Add structural raw data validation reports.
- Document setup and execution steps.

## Out of Scope for Sprint 1

- Feature engineering
- Negative sample generation
- Model training
- Model inference
- Embedding generation
- TF-IDF features
- Sentence Transformer usage
- LightGBM or CatBoost usage
- Submission file generation

## Sprint 2 Scope

Sprint 2 focuses only on exploratory data analysis and data understanding. The
goal is to understand the competition datasets deeply before feature
engineering, negative sampling, and model selection decisions.

Implemented Sprint 2 analysis modules:

- Query analysis for `terms.csv`
- Item analysis for `items.csv`
- Category hierarchy analysis
- Attribute key analysis
- Training pair analysis
- Train/test overlap analysis
- Raw data quality analysis
- Reusable visualization utilities
- Markdown EDA report generation

Sprint 2 outputs:

- `reports/sprint_2_eda_report.md`
- `reports/figures/query_length_histogram.png`
- `reports/figures/query_word_count_histogram.png`
- `reports/figures/title_length_histogram.png`
- `reports/figures/top_20_brands.png`
- `reports/figures/top_20_categories.png`
- `reports/figures/gender_distribution.png`
- `reports/figures/age_group_distribution.png`
- `reports/figures/top_attribute_keys.png`

## Out of Scope for Sprint 2

- Model training
- Feature engineering
- TF-IDF feature creation
- Embedding generation
- Negative sampling
- LightGBM, CatBoost, or XGBoost usage
- Sentence Transformer usage
- Submission file generation

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── attribute_analysis.py
│   │   ├── category_analysis.py
│   │   ├── common.py
│   │   ├── item_analysis.py
│   │   ├── quality_analysis.py
│   │   ├── query_analysis.py
│   │   ├── report_generator.py
│   │   ├── train_test_analysis.py
│   │   ├── training_analysis.py
│   │   └── visualization.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── paths.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_reporter.py
│   │   └── validate_raw_data.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── features/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py
├── models/
├── reports/
│   ├── figures/
│   └── sprint_2_eda_report.md
├── submissions/
├── logs/
├── tests/
├── requirements.txt
└── README.md
```

## Directory Responsibilities

`data/raw/` stores the original competition files:

- `items.csv`
- `terms.csv`
- `training_pairs.csv`
- `submission_pairs.csv`
- `sample_submission.csv`

`data/interim/` is reserved for intermediate datasets in future sprints.

`data/processed/` is reserved for processed datasets in future sprints.

`notebooks/` is reserved for exploratory notebooks. No EDA is performed in
Sprint 1.

`src/config/` contains project-wide configuration such as filesystem paths.

`src/analysis/` contains Sprint 2 EDA modules and the markdown report generator.

`src/data/` contains raw data loading and validation reporting modules.

`src/features/` is reserved for feature engineering in future sprints.

`src/models/` is reserved for modeling code in future sprints.

`src/evaluation/` is reserved for evaluation code in future sprints.

`src/utils/` contains shared utilities such as logging.

`models/` is reserved for trained model artifacts in future sprints.

`reports/` stores generated analysis reports and figures.

`submissions/` is reserved for generated competition submissions in future
sprints.

`logs/` stores runtime log files.

`tests/` is reserved for automated tests.

## Python Version

Recommended Python version:

```text
Python 3.11+
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Required Libraries

The initial Sprint 1 dependencies are:

- pandas
- numpy
- matplotlib
- jupyter
- scikit-learn
- pyyaml
- python-dotenv
- tqdm

## Raw Data Setup

Place the following files under `data/raw/`:

```text
data/raw/items.csv
data/raw/terms.csv
data/raw/training_pairs.csv
data/raw/submission_pairs.csv
data/raw/sample_submission.csv
```

All paths are defined in `src/config/paths.py`. Project modules should import
paths from that module instead of using hardcoded filesystem locations.

## Run Raw Data Validation

After installing dependencies and placing the raw files under `data/raw/`, run:

```bash
python3 -m src.data.validate_raw_data
```

The validation command loads each raw CSV file and reports:

- shape
- column names
- data types
- missing value counts
- memory usage
- duplicate row count
- first 5 rows

Logs are written to:

```text
logs/application.log
```

## Run Sprint 2 EDA Report

After installing dependencies and placing the raw files under `data/raw/`, run:

```bash
python3 -m src.analysis.report_generator
```

The command runs all Sprint 2 analysis modules, creates EDA figures, and writes
the final markdown report to:

```text
reports/sprint_2_eda_report.md
```

Generated figures are written to:

```text
reports/figures/
```

## Sprint 2 Key Findings

- `terms.csv` contains 50,153 queries and all queries are unique.
- Average query length is 16.18 characters and 2.62 words.
- Query text is short; lexical matching signals are expected to be important in
  future sprints.
- `items.csv` contains 962,873 products and 79,788 unique non-empty brands.
- Product titles average 55.02 characters.
- Category hierarchy has 15 main categories and an average depth of 3.80.
- Attribute coverage is high at about 98.02%.
- The most frequent attribute keys include `renk`, `color detail`, `menşei`,
  `materyal`, and `yıkama talimatı`.
- `training_pairs.csv` contains 250,000 pairs and all labels are positive.
- Test queries do not overlap with train queries.
- About 21.12% of test items are seen in train; most test items are new.
- The largest data quality risk is high `unknown` coverage in `gender` and
  `age_group`.

## Sprint 2 Conclusions

The main data advantage is rich product metadata, especially `title`,
`category`, `brand`, and `attributes`.

The main data problem is the positive-only training structure combined with
zero train/test query overlap. Sprint 3 should therefore avoid query-id
memorization and focus on text and catalog metadata signals.

Negative sampling in Sprint 3 should be designed carefully around category,
title similarity, item popularity, and false-negative risk.

## Architecture Notes

The project follows clean architecture principles at the repository level:

- Configuration is isolated under `src/config/`.
- Data access is isolated under `src/data/data_loader.py`.
- Data reporting is isolated under `src/data/data_reporter.py`.
- Exploratory analysis is isolated under `src/analysis/`.
- Shared logging is isolated under `src/utils/logger.py`.
- Future feature, modeling, and evaluation layers have their own packages.

This keeps Sprint 1 focused on infrastructure and Sprint 2 focused on data
understanding while leaving clear extension points for future ML development.
