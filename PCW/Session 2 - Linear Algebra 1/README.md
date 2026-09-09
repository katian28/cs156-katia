# Session 2 PCW: Linear Algebra 1 — Drawing Lines with Regression

**Date**: Fall 2026, Session 2  
**Student**: Katia Gwaneza Nkurunziza  
**Status**: ✅ Complete

---

## Overview

This session covers linear and logistic regression on the Iris dataset, implementing both models using scikit-learn and from-scratch code to understand their mechanics.

## Learning Objectives

- Fit linear regression to continuous data and interpret the fitted line
- Fit logistic regression to categorical data and visualize decision boundaries
- Implement loss functions (MSE, log-loss) and understand what they measure
- Build regression and classification models from scratch to see how they minimize loss
- Distinguish between a model's training performance and generalization

## Notebook Structure

### Setup (Cell 0)
- Import libraries: numpy, pandas, matplotlib, scikit-learn
- Load Iris dataset (150 flowers, 3 species, 4 measurements)
- Display dataset structure and species distribution

### Code Cell 1: Linear Regression
**Problem**: Predict **petal length** from **petal width** (continuous → continuous)

**Key Results**:
- Fitted equation: `petal_length = 2.2293 * petal_width + 1.0840`
- R² score: **0.9725** (model explains 97.25% of variance)
- Interpretation: Petal measurements scale proportionally; a simple linear model captures this well

**Visualization**: Scatter plot with fitted line showing how well the model fits the data

---

### Code Cell 2: Logistic Regression
**Problem**: Predict **species** from **sepal length & width** (continuous → categorical)

**Key Results**:
- Overall accuracy: **96%** (4/150 errors, mostly versicolor vs. virginica confusion)
- Per-species accuracy:
  - Setosa: 100% (perfectly separated from others)
  - Versicolor: 92% (some overlap with virginica)
  - Virginica: 100% (well-separated)

**Visualization**: Decision boundary plot showing where the model draws lines between species

**Insight**: Iris species have distinct sepal morphologies—small setosa flowers are easily distinguished from large virginica flowers—but versicolor is intermediate and harder to separate perfectly.

---

### Code Cell 3: Loss Functions — Measuring "Badness"
Implement two loss functions from scratch:

**Mean Squared Error (Regression)**:
```python
MSE = mean((y_true - y_pred)²)
RMSE = sqrt(MSE) ≈ 0.62 cm
```
Interpretation: On average, predictions are off by ±0.62 cm. Low residuals and good R² indicate the model captures the real trend.

**Log-Loss (Classification)**:
```python
Log-Loss = -mean(y*log(ŷ) + (1-y)*log(1-ŷ))
```
Interpretation: Heavily penalizes confident wrong predictions. A model predicting 0.99 probability for the wrong class is penalized much more than one predicting 0.51.

**Verification**: Implementations match scikit-learn's built-in loss calculations, confirming correctness.

**Visualizations**:
- Residual plot: Shows if errors are randomly scattered (good) or have patterns (bad model assumption)
- Residual histogram: Checks if errors are approximately normally distributed

---

### Code Cell 4: Linear Regression from Scratch
**Approach**: Closed-form normal equation (no iteration)

**Mathematics**:
```
θ = (X^T X)^-1 X^T y
```
This formula finds the exact optimal weights in one step by minimizing ||Xθ - y||².

**Key Points**:
- The fitted line *minimizes* squared errors—it's the provably best line for this data
- Implementation matches sklearn perfectly (verification included)
- The model doesn't fit perfectly because:
  1. Biological variation within species
  2. Measurement precision limits
  3. Unmodeled factors (soil, season, growing conditions)
  4. Linear relationship assumption (nature isn't perfectly linear)

**Code Insight**: By adding a bias term (column of 1s) to X, we solve for both slope and intercept simultaneously.

---

### Code Cell 5: Logistic Regression from Scratch
**Approach**: Iterative gradient descent with sigmoid activation

**Mathematics**:
```
1. Sigmoid: s(z) = 1 / (1 + e^-z)  [maps ℝ → [0,1]]
2. Prediction: ŷ = sigmoid(X @ w + b)
3. Loss: -mean(y*log(ŷ) + (1-y)*log(1-ŷ))
4. Update: w := w - learning_rate * ∇loss
```

**Binary Classification** (Setosa vs. Versicolor):
- Accuracy: 100%
- Uses only 2 sepal features (1,000 iterations, learning_rate=0.1)
- Features are normalized (StandardScaler) so gradient descent converges smoothly

**Visualizations**:
- Loss history: Shows convergence—log-loss decreases smoothly over iterations
- Decision boundary: Curved line separating the two species

**Why It Works**: 
- Gradient descent iteratively reduces loss by moving weights in the direction of steepest descent
- Convex loss function means no local optima—any minimum found is globally optimal
- Feature normalization prevents large-scale features from dominating the gradient
- The model learns a decision boundary where sigmoid output = 0.5

**Why It Doesn't Fit Perfectly**:
- Linear separability assumption: Real species differences might curve in complex ways
- Limited features: Using only 2 of 4 available measurements
- Inherent biological variation: Flowers have individual differences

---

## Questions & Reflections

### Question 1: Basic Concepts (6 Questions)
Covers: Iris dataset, visual patterns, model objectives, loss functions, metrics, residuals/accuracy/sensitivity

**Key Insights**:
- Iris represents real biological data: measurements that vary by species
- Visual patterns reveal *systematic differences*: species are clustered, petal measurements are correlated
- Models learn simplified representations: lines (regression) or boundaries (classification)
- Loss functions measure error; metrics measure performance

---

### Question 2: Core Questions (Critique & Reflection)
Critiques the two models and measures how badly they fit

**What Models Capture**:
- Linear regression: Allometric scaling (flower parts grow together proportionally)
- Logistic regression: Distinct species morphologies (systematic size differences by species)

**Limitations**:
- Linear regression assumes perfect linearity; real relationships curve slightly
- Logistic regression assumes linear separability; iris species slightly overlap in sepal space

**Badness Measures**:
- Regression: MSE, RMSE (in original units), R² (fraction of variance explained)
- Classification: Accuracy, log-loss, per-class precision/recall, confusion matrix

**Reflection on AI Responses**:
- AI correctly identified all measures but was generic about *why* certain species are harder to classify
- AI could have dug deeper into log-loss sensitivity to confidence, and residual plot interpretation
- My own understanding: High R² (0.97) and high accuracy (96%) suggest genuine learning, not memorization—verified by implementing loss functions and checking convergence

---

### Question 3: Extension (From-Scratch Models)
Implements both models without scikit-learn to understand internal mechanics

**Linear Regression Insights**:
- Closed-form solution: There's an exact formula that finds the best line instantly
- No iteration needed (unlike gradient descent)
- Why it doesn't fit perfectly: biological variation, measurement noise, unmodeled factors, linearity assumption

**Logistic Regression Insights**:
- Iterative optimization: Gradient descent nudges weights step-by-step toward lower loss
- Convex loss: Guarantees global optimum (no local minima traps)
- Feature normalization: Essential for gradient descent to work smoothly
- Why it doesn't fit perfectly: overlapping species, linear boundary assumption, limited features

**Specific Dialogue**:
- **On gradient descent**: Convex loss function ensures any local minimum is globally optimal; no risk of getting stuck
- **On learning rate**: Too high → oscillation/divergence; too low → slow convergence; tuning is necessary
- **On normalization**: Gradient descent is scale-sensitive; normalized features give balanced gradients across dimensions

---

## Mathematical Concepts

### Linear Regression
- **Model**: ŷ = w₀ + w₁x₁ + ... + wₙxₙ
- **Loss**: MSE = (1/m) Σ(ŷ - y)²
- **Solution**: θ = (X^T X)^-1 X^T y (closed-form normal equation)
- **Interpretation**: Finds the line that minimizes squared vertical distance to all points

### Logistic Regression
- **Model**: ŷ = sigmoid(w₀ + w₁x₁ + ... + wₙxₙ)
- **Sigmoid**: s(z) = 1 / (1 + e^-z) ∈ [0, 1] (squashes to probabilities)
- **Loss**: Log-loss = -(1/m) Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
- **Optimization**: Gradient descent: θ := θ - η∇loss
- **Interpretation**: Learns a decision boundary where sigmoid = 0.5 separates classes

---

## Code Implementations

### Functions Implemented

**`compute_mse(y_true, y_pred)`**: Computes Mean Squared Error from scratch
```python
residuals = y_true - y_pred
mse = mean(residuals²)
```

**`compute_log_loss(y_true, y_pred_proba)`**: Computes Log-Loss from scratch
```python
y_true_onehot = one_hot_encode(y_true)
loss = -mean(sum(y_true_onehot * log(y_pred_proba)))
```

**`LinearRegressionFromScratch`**: Closed-form solution
- `fit()`: Solves X^T X @ θ = X^T y via matrix inversion
- `predict()`: Returns Xθ

**`LogisticRegressionFromScratch`**: Gradient descent optimization
- `fit()`: Iteratively updates weights via gradient descent
- `predict_proba()`: Returns sigmoid(Xw + b) ∈ [0,1]
- `predict()`: Thresholds probabilities at 0.5

---

## Files in This Folder

```
PCW/Session 2 - Linear Algebra 1/
├── README.md                  (this file)
├── pcw_lesson_2.ipynb        (complete notebook with all cells and questions)
└── [output images from notebook execution]
```

---

## How to Run

### Option 1: Jupyter Notebook
```bash
cd "PCW/Session 2 - Linear Algebra 1"
jupyter notebook pcw_lesson_2.ipynb
```
Then run all cells from top to bottom.

### Option 2: Command Line (if converting to Python)
```bash
# Extract code cells and run as .py script
python pcw_lesson_2.py
```

---

## Dependencies

All dependencies are in `pyproject.toml` (project root). If needed, install with:
```bash
uv pip install numpy pandas matplotlib scikit-learn jupyter
```

---

## Key Results Summary

| Model | Task | Metric | Value | Interpretation |
|-------|------|--------|-------|-----------------|
| Linear Regression | Predict petal length from width | R² | 0.9725 | Model explains 97.3% of variance—excellent fit |
| Linear Regression | | RMSE | 0.62 cm | Typical prediction error ±0.62 cm |
| Logistic Regression | Predict species from sepal measurements | Accuracy | 96.0% | 144/150 predictions correct |
| Logistic Regression (Setosa vs. Versicolor) | Binary classification | Accuracy | 100% | Perfect separation for binary case |

---

## Concepts Mastered This Session

✅ Linear regression: fitting lines to minimize squared errors  
✅ Logistic regression: learning decision boundaries for classification  
✅ Loss functions: what models optimize during training  
✅ Residual analysis: checking if model assumptions are valid  
✅ From-scratch implementations: understanding internal mechanics  
✅ Generalization: why models don't fit perfectly (and that's okay)  

---

## Next Steps (Session 3 Preview)

**Topic**: Linear Algebra 2 — Collinearity and Feature Relationships

- **Collinearity**: What happens when features are correlated? (Iris sepal length & width are correlated)
- **Feature selection**: How do we choose which features to use?
- **Regularization**: How do we prevent overfitting when we have many features?
- **Principal Component Analysis**: How do we reduce dimensionality while preserving information?

---

## Reflection

This session clarified the deep difference between regression and classification:
- **Regression** learns continuous outputs; it minimizes squared errors
- **Classification** learns boundaries; it minimizes classification error (or log-loss)

Both use calculus to find optimal parameters, but the geometry is different. Linear regression finds a 1D line in feature space; logistic regression finds a hyperplane that divides space into regions.

Implementing both from scratch showed that the same data can be modeled in multiple ways, each capturing different structure. The Iris dataset's clear separation by species means both models work well, but in messier real-world problems, these models would fail differently—linear regression would have huge residuals, logistic regression would misclassify overlapping points.

---

**Last Updated**: Session 2, Fall 2026  
**Status**: ✅ Ready for Session 3
