# CS156: Finding Patterns in Data with Machine Learning

This repository contains coursework materials and solutions for CS156, a 4-credit machine learning course offered in Fall 2026 at Minerva University.

## Course Overview

**Instructor**: [Your Instructor]  
**Term**: Fall 2026  
**Credits**: 4  
**Prerequisites**: CS110, CS111, CS113, CS114

This course covers core machine learning techniques including classification, neural networks, SVMs, clustering, feature selection, cross-validation, and overfitting. Students implement ML algorithms in Python and apply them to real datasets.

### Learning Objectives

- **#cs156-MLCode**: Produce working, readable, and performant Python implementations of machine learning systems
- **#cs156-MLExplaination**: Clearly articulate ML systems using descriptions, mathematical notation, and visualizations
- **#cs156-MLMath**: Evaluate problems and derive solutions in linear algebra, calculus, and Bayesian statistics
- **#cs156-MLDevelopment**: Contribute to ML learning resources for current and future students

## Repository Structure

```
.
├── src/cs156/                    # Main package with reusable utilities
│   ├── __init__.py              # Package initialization
│   ├── data.py                  # Data loading and preprocessing
│   └── plotting.py              # Visualization utilities
├── PCW/                          # Problem-solving coursework
│   ├── Session 1 - Fundamentals/
│   ├── Session 2 - Linear Algebra 1/
│   ├── Session 3 - Linear Algebra 2/
│   └── ... (more sessions)
├── .vscode/                      # VS Code workspace configuration
│   └── settings.json            # Python interpreter and linting config
├── binder/                       # Binder configuration for cloud environments
│   └── requirements.txt         # Pinned dependencies
├── pyproject.toml               # Project metadata and dependencies (uv)
└── README.md                    # This file
```

## Quick Start

### Prerequisites
- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone <repository-url>
cd cs156-katia

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"
```

### Launch Jupyter

```bash
jupyter notebook
```

## Course Topics by Unit

### Unit 0: ML Fundamentals
- Session 1: Introduction and Overview

### Unit 1: Core Mathematics
- Session 2: Linear Algebra 1 - Tensors, Classification, Regression
- Session 3: Linear Algebra 2 - Collinearity
- Session 4: Trees 1 - Classification by Partition
- Session 5: Max Likelihood 1 - Naive Bayes
- Session 6: Max Likelihood 2 - Parameter Estimation
- Session 7: Networks 1 - Feed-Forward Neural Networks
- Session 8: Gradients - Multivariate Derivatives
- Session 9: Metrics and Cross-Validation
- Session 10: Unit 1 Review & Assignment 1 Prep

### Unit 2: High-Dimensional Data
- Session 11: Data Ethics and Bias-Variance Tradeoff
- Session 12: Basis Functions 1 - Functions as Parameters
- Session 13: Projections 1 - Support Vector Machines & Kernels
- Session 14: Trees 2 - XGBoost
- Session 15: Basis Functions 2 - Ridge and Lasso
- Session 16: Dimensionality Reduction 1 - PCA
- Session 17: Projections 2 - Filters, Convolutions & Transforms
- Session 18: Dimensionality Reduction 2 - Autoencoders
- Session 19: Time Series 2 - Recurrent Neural Networks
- Session 20: Networks 2 - Transfer Learning
- Session 21: Unit 2 Review & Assignment 2 Prep

### Unit 3: Generative Models
- Session 22: Time Series 1 - Hidden Markov Models
- Session 23: Attention
- Session 24: Transformer-Based Large Language Models
- Session 25: Course Synthesis & Final Project Prep

## Assignments & Grading

| Assignment | Weight | Due Date | Points |
|-----------|--------|----------|--------|
| Repository & Summary | 30% | Week 1 (Fri) | 12x |
| First Pipeline | 10% | Week 6 (Thu) | 4x |
| Second Pipeline | 15% | Week 13 (Mon) | 6x |
| Final Pipeline | 20% | Week 15 (Thu) | 8x |
| **Pre-Class Work** | **15%** | Weekly | — |
| **Poll Responses** | **10%** | As given | — |

**Total**: 75% Assignments + 25% Classroom Scores

## Using the Package

### Data Utilities
```python
from cs156.data import load_data, preprocess_data

df = load_data("path/to/data.csv")
df_clean = preprocess_data(df)
```

### Plotting Utilities
```python
from cs156.plotting import plot_distribution, plot_scatter

plot_distribution(df["column"], title="Distribution")
plot_scatter(df["x"], df["y"], title="Relationship")
```

## Key Resources

**Main Text**: Murphy, K. P. (2022). *Probabilistic Machine Learning* (Creative Commons)  
https://probml.github.io/pml-book/book1.html

**Supplemental Texts**:
- Barber, D. (2012). *Bayesian Reasoning and Machine Learning*
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*
- Downey, A. (2013). *Think Bayes* (Creative Commons)

## Environment Configuration

The `.vscode/settings.json` automatically configures:
- Python interpreter path to project's venv
- Black code formatting (line length: 88)
- Flake8 linting
- Import organization

## Dependencies

**Core ML Stack**:
- numpy, pandas, matplotlib, scikit-learn, jupyter

**Development**:
- pytest, black, ruff, mypy

## Academic Integrity

All work submitted must be your own. Refer to Minerva's academic integrity policies.

---

**Note**: This repository is actively used for Fall 2026 coursework. Check the syllabus regularly for updates.
