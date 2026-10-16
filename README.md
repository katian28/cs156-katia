# CS156: Finding Patterns in Data with Machine Learning

This repository contains coursework for CS156, a 4-credit machine learning course at Minerva University, Fall 2026.

## Course Overview

**Instructor**: Watson (CS156, TTH @ 3:00PM UTC)
**Term**: Fall 2026
**Credits**: 4

This course covers core machine learning techniques including classification, neural networks, SVMs, clustering, feature selection, cross-validation, and overfitting. Work is implemented in Python and applied to real datasets, including personal data for the three pipeline assignments.

### Learning Objectives

- **#cs156-MLCode**: Produce working, readable, and performant Python implementations of machine learning systems
- **#cs156-MLExplanation**: Clearly articulate ML systems using descriptions, mathematical notation, and visualizations
- **#cs156-MLMath**: Evaluate problems and derive solutions in linear algebra, calculus, and Bayesian statistics
- **#cs156-MLFlexibility**: Reason flexibly, apply information in new contexts, produce novel work, and articulate meta-knowledge about machine learning
- **#cs156-MLParticipation**: Pre-class work and in-class participation

## How This Repo Is Organized

- **`PCW/`**: one folder per session (`Session 1 - Introduction` through `Session 11 - Data Ethics` so far), each with a `README.md` summarizing the session and a `pcw_lesson_N.ipynb` notebook, executed and containing real output.
- **`study_materials/`**: deeper conceptual notes per session, in my own words, written for review rather than submission, separate from the PCW answers themselves.
- **`Assignments/`**: the three pipeline assignments. `Assignment 1 - First Pipeline/` is current, a food-photo allergen detector built from my own personal photo archive, with a `README.md` write-up and an executed `pipeline.ipynb`.
- **`course_materials/`**: the syllabus and other instructor-provided reference material.

## Grading

| Component | Weight | Notes |
|---|---|---|
| Pre-class work | 15% | Scored per session, typically 3s |
| Participation | 15% | Scored per session, typically 3s |
| Code Repository (Summative) | 40% | Tracked continuously from a coversheet assignment linking here; scored on #MLMath, #MLExplanation, #MLCode, #MLFlexibility as they're demonstrated across the repo over the term |
| Assignments (3 pipelines) | 30% | Each scored only on #MLFlexibility: how many techniques not yet covered in class were used, and how creatively |

The thesis for doing well, per the instructor: participate and complete PCW every session, be creative on the open-ended assignments, and keep this repo well documented enough that an external reader can follow it without extra context. On the pipeline assignments specifically: low accuracy is fine, loose reasoning about *why* a result looks the way it does is not.

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
- Session 11: Data Ethics and Bias-Variance Tradeoff

### Unit 2: High-Dimensional Data
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

Each PCW session and assignment notebook can be run independently from its own folder; see that folder's `README.md` for specifics (e.g. `Assignment 1 - First Pipeline/` needs photos placed in a local `food_photos/` folder, gitignored, before `pipeline.ipynb` will run).

## Key Resources

**Main Text**: Murphy, K. P. (2022). *Probabilistic Machine Learning* (Creative Commons)
https://probml.github.io/pml-book/book1.html

**Supplemental Texts**:
- Barber, D. (2012). *Bayesian Reasoning and Machine Learning*
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*
- Downey, A. (2013). *Think Bayes* (Creative Commons)

## Dependencies

**Core ML Stack**: numpy, pandas, matplotlib, scikit-learn, jupyter, tensorflow, seaborn, pillow-heif
**Development**: pytest, black, ruff, mypy

## Academic Integrity

All work submitted must be my own. Refer to Minerva's academic integrity policies.

---

**Note**: This repository is actively used for Fall 2026 coursework and updated each session.
