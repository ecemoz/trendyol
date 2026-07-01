# Trendyol Datathon 2026 - Query Product Relevance

This repository contains the production-oriented machine learning project
foundation for the Trendyol Datathon 2026 query-product relevance task.

The final objective of the project is to predict whether a given
`(query, product)` pair is relevant or not as a binary classification problem.

Sprint 1 is limited to project infrastructure and raw data validation. No model
development is included in this sprint.

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

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
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

`src/data/` contains raw data loading and validation reporting modules.

`src/features/` is reserved for feature engineering in future sprints.

`src/models/` is reserved for modeling code in future sprints.

`src/evaluation/` is reserved for evaluation code in future sprints.

`src/utils/` contains shared utilities such as logging.

`models/` is reserved for trained model artifacts in future sprints.

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

## Architecture Notes

The project follows clean architecture principles at the repository level:

- Configuration is isolated under `src/config/`.
- Data access is isolated under `src/data/data_loader.py`.
- Data reporting is isolated under `src/data/data_reporter.py`.
- Shared logging is isolated under `src/utils/logger.py`.
- Future feature, modeling, and evaluation layers have their own packages.

This keeps Sprint 1 focused on infrastructure while leaving clear extension
points for future ML development.
