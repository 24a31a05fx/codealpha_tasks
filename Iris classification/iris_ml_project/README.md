# 🌸 Iris Flower Classification

> A production-style machine learning pipeline for multi-class flower species classification using the UCI Iris dataset. Trains and compares five classifiers, evaluates them rigorously, and generates a self-contained HTML report with rich visualisations.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Output Report](#output-report)
- [Module Reference](#module-reference)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)

---

## Overview

This project implements a complete supervised classification pipeline on the classic **Iris dataset** — from raw data ingestion through exploratory analysis, model training, evaluation, and automated report generation.

The codebase is structured as a proper Python package with separated concerns: configuration, data loading, model training, visualisation, and report building each live in their own module. The goal is to demonstrate ML engineering practices beyond a typical notebook — clean separation of logic, reproducible experiments, and a polished deliverable.

**Key objectives:**

- Preprocess and validate input data programmatically
- Train five classifiers with consistent train/test splits and stratification
- Evaluate using test accuracy, 5-fold cross-validation, precision, recall, and F1
- Generate a self-contained HTML report with 9 embedded visualisations
- Keep all configuration in one place (`config.py`) for easy tuning

---

## Project Structure

```
iris_ml_project/
│
├── data/
│   └── iris.csv                  # UCI Iris dataset (150 samples)
│
├── src/
│   ├── config.py                 # All paths, hyperparameters, colour palette
│   ├── data_loader.py            # Ingestion, validation, train/test split
│   ├── models.py                 # Model registry + training pipeline
│   ├── visualiser.py             # All matplotlib/seaborn chart functions
│   ├── report.py                 # HTML report assembler
│   └── main.py                   # Orchestration entry point
│
├── outputs/
│   └── iris_classification_report.html   # Generated report (auto-created)
│
├── notebooks/                    # Jupyter notebooks (exploratory work)
├── requirements.txt
└── README.md
```

---

## Dataset

The **UCI Iris dataset** is a multivariate dataset introduced by Ronald Fisher in 1936. It is one of the most widely used benchmarks in pattern recognition and machine learning.

| Property        | Value                              |
|-----------------|------------------------------------|
| Samples         | 150 (50 per class)                 |
| Features        | 4 (sepal length, sepal width, petal length, petal width) |
| Target classes  | 3 (*Iris setosa*, *Iris versicolor*, *Iris virginica*)   |
| Missing values  | None                               |
| Feature type    | Continuous (cm)                    |

### Feature Summary

| Feature          | Mean   | Std    | Min    | Max    |
|------------------|--------|--------|--------|--------|
| Sepal Length (cm)| 5.843  | 0.828  | 4.300  | 7.900  |
| Sepal Width (cm) | 3.054  | 0.434  | 2.000  | 4.400  |
| Petal Length (cm)| 3.759  | 1.765  | 1.000  | 6.900  |
| Petal Width (cm) | 1.199  | 0.763  | 0.100  | 2.500  |

---

## Models

Five classifiers are trained and compared. Each model is drawn from `MODEL_REGISTRY` in `models.py` — adding or removing a model requires only editing that dictionary.

| Model                   | Algorithm type        | Key hyperparameters             |
|-------------------------|-----------------------|---------------------------------|
| K-Nearest Neighbors     | Instance-based        | k = 5, uniform weights          |
| Decision Tree           | Tree-based            | Gini impurity, random_state=42  |
| Random Forest           | Ensemble (bagging)    | 100 estimators, random_state=42 |
| Support Vector Machine  | Kernel method         | RBF kernel, C=1.0, γ=scale     |
| Logistic Regression     | Linear discriminant   | max_iter=200, L2 regularisation |

**Preprocessing applied to all models:**
- `StandardScaler` fitted on training data only, applied to test data (no leakage)
- Stratified 75/25 train/test split to preserve class balance
- 5-fold stratified cross-validation on the full dataset

---

## Results

> Evaluated on a held-out test set of 38 samples (25% stratified split, `random_state=42`)

| Rank | Model                   | Test Accuracy | CV Mean  | CV Std  | Weighted F1 |
|------|-------------------------|---------------|----------|---------|-------------|
| 🥇 1  | Support Vector Machine  | **94.7%**     | 96.7%    | ±2.1%   | 0.947       |
| 🥈 2  | K-Nearest Neighbors     | 92.1%         | 96.0%    | ±2.5%   | 0.921       |
| 🥈 2  | Random Forest           | 92.1%         | 96.7%    | ±2.1%   | 0.921       |
| 🥈 2  | Logistic Regression     | 92.1%         | 96.0%    | ±3.9%   | 0.921       |
| 5    | Decision Tree           | 89.5%         | 95.3%    | ±3.4%   | 0.894       |

**Winner: Support Vector Machine (RBF kernel)** — highest test accuracy and joint-best cross-validation score, with lower variance than Logistic Regression.

### Feature Importance (Random Forest)

| Rank | Feature           | Importance |
|------|-------------------|------------|
| 1    | Petal Length (cm) | ~0.44      |
| 2    | Petal Width (cm)  | ~0.43      |
| 3    | Sepal Length (cm) | ~0.09      |
| 4    | Sepal Width (cm)  | ~0.04      |

Petal measurements account for ~87% of the discriminative signal. Sepal width is nearly uninformative in isolation.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone or download the project
git clone https://github.com/your-org/iris-classification.git
cd iris-classification

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
```

---

## Usage

```bash
# From the project root
cd src
python main.py
```

The pipeline prints a live progress log:

```
  🌸  Iris Flower Classification Pipeline
    ──────────────────────────────────────────────────────────

  [1/4] Loading & preprocessing data …
         150 samples, 4 features, 3 classes  | missing: 0

  [2/4] Training models …
  ✓ K-Nearest Neighbors           acc=92.1%  cv=96.0% ± 2.5%
  ✓ Decision Tree                 acc=89.5%  cv=95.3% ± 3.4%
  ✓ Random Forest                 acc=92.1%  cv=96.7% ± 2.1%
  ✓ Support Vector Machine        acc=94.7%  cv=96.7% ± 2.1%
  ✓ Logistic Regression           acc=92.1%  cv=96.0% ± 3.9%

         🏆  Best model: Support Vector Machine  (94.7% test accuracy)

  [3/4] Generating visualisations …
         9 charts rendered

  [4/4] Building HTML report …
  ──────────────────────────────────────────────────────────
  ✅  Done in 4.7s
  📄  Report: ../outputs/iris_classification_report.html
```

Open `outputs/iris_classification_report.html` in any browser to view the full report — no server required.

### Customising the experiment

All tuneable parameters are in `src/config.py`:

```python
TEST_SIZE    = 0.25        # fraction held out for testing
RANDOM_STATE = 42          # global seed for reproducibility
CV_FOLDS     = 5           # number of cross-validation folds
```

To add a new model, edit `MODEL_REGISTRY` in `src/models.py`:

```python
from sklearn.naive_bayes import GaussianNB

MODEL_REGISTRY["Naive Bayes"] = GaussianNB()
```

No other file needs changing.

---

## Output Report

The HTML report is fully self-contained (all charts are embedded as base64 PNG) and includes:

| Section                    | Content                                                          |
|----------------------------|------------------------------------------------------------------|
| Dataset Overview           | Key stats, class distribution donut chart, descriptive stats table |
| Exploratory Data Analysis  | Histograms, box plots, pairplot scatter matrix, correlation heatmap |
| Model Comparison           | Horizontal accuracy bars, CV mean ± std grouped bars             |
| Best Model Deep-Dive       | Confusion matrix heatmap, feature importance chart + table       |
| Detailed Per-Model Metrics | Per-class precision / recall / F1 / support for all 5 models     |

---

## Module Reference

| Module            | Responsibility                                                  |
|-------------------|-----------------------------------------------------------------|
| `config.py`       | Single source of truth for paths, constants, and visual style   |
| `data_loader.py`  | CSV loading, validation, label encoding, StandardScaler splits  |
| `models.py`       | MODEL_REGISTRY, `train_all()`, `get_best()`, `get_feature_importances()` |
| `visualiser.py`   | Nine chart functions, each returning a base-64 PNG string       |
| `report.py`       | Jinja-free HTML templating; assembles the final report file     |
| `main.py`         | Orchestrates steps 1–4; the only file a user needs to run       |

---

## Key Findings

1. **Petal features dominate.** Petal length and petal width together account for ~87% of feature importance (Random Forest). Any model with access to petal measurements performs well.

2. **Setosa is trivially separable.** All five models achieve perfect recall on *Iris setosa*. The classification challenge lies entirely in distinguishing *versicolor* from *virginica*, which overlap in sepal space but are separable in petal space.

3. **SVM edges ahead.** The RBF kernel's ability to model non-linear boundaries gives it a slight advantage over the linear Logistic Regression model on the versicolor/virginica boundary.

4. **High cross-validation consistency.** All models achieve >95% mean CV accuracy, confirming the dataset is genuinely learnable and the results are not artefacts of a lucky split.

5. **Decision Tree is most variance-prone.** Its higher CV std (±3.4%) reflects sensitivity to the specific training folds — the ensemble Random Forest corrects this at the cost of interpretability.

---

## Tech Stack

| Library        | Version    | Purpose                              |
|----------------|------------|--------------------------------------|
| Python         | ≥ 3.9      | Core language                        |
| scikit-learn   | ≥ 1.3      | ML models, preprocessing, evaluation |
| pandas         | ≥ 2.0      | Data ingestion and manipulation      |
| numpy          | ≥ 1.24     | Numerical operations                 |
| matplotlib     | ≥ 3.7      | Base plotting engine                 |
| seaborn        | ≥ 0.12     | Statistical visualisation            |

---

## License

This project is released for educational purposes. The Iris dataset is public domain (UCI Machine Learning Repository).

---

*Built with scikit-learn · UCI Iris Dataset · Fisher (1936)*
