# Session 3 PCW: Linear Algebra 2 - Collinearity and Multivariate Regression

**Date**: Fall 2026, Session 3  
**Student**: Katia Gwaneza Nkurunziza  
**Status**: Ready to complete

---

## Overview

This session digs into what happens when you have *multiple* independent variables. We move from fitting a line (Session 2) to fitting a hyperplane in 3D or higher dimensions.

The key concepts:
1. Matrices represent vector spaces
2. Matrix operations describe geometric transformations (rotating, scaling, projecting)
3. Multivariate regression finds the best-fitting hyperplane by minimizing squared residuals
4. The normal equation solves for optimal parameters via linear algebra: β = (X^T X)^-1 X^T y

## Learning Objectives

- Understand vector spaces and how matrices represent them
- Fit multivariate linear regression to real data (Titanic)
- Derive the OLS solution from calculus (setting derivative to 0)
- Implement the normal equation to solve for parameters
- Visualize regression results and interpret residuals
- Understand what collinearity means (preview of Session 4)

## What You're Doing

### Question 1: Basic Concepts (LLM Interview)
Interview an LLM about:
- What is a vector space in ML?
- How do matrix operations work?
- Why sum of squares instead of sum?
- Why set the derivative to 0?

**Action**: Have this conversation with Claude, ChatGPT, etc. Paste the full transcript in the notebook. Flag any errors or confusing answers.

### Code Cell 1: Load Titanic and Fit Model
- Clean the Titanic dataset
- Choose 3 numeric independent variables (e.g., age, pclass, sibsp)
- Fit a linear regression model predicting passenger fare
- Extract and print model parameters (intercept + 3 slopes)
- Calculate R², MSE, RMSE

This is your baseline model.

### Question 2: Deep Mathematical Dive (LLM Interview)
Continue the LLM conversation, asking for:
1. Multivariate model equations in matrix notation (y = Xβ + ε)
2. OLS loss function in matrix notation: L(β) = (y - Xβ)^T(y - Xβ)
3. First derivative: ∇L(β) = -2X^T(y - Xβ)
4. Normal equation solution: β = (X^T X)^-1 X^T y
5. Three specific data points: create a 3x3 X matrix and 3x1 y vector
6. Solve the 3x3 system by hand (show numerical work)
7. Explain operations in vector space language

**Action**: Push the LLM for specific math and code. Correct any mistakes. Paste final transcript.

### Question 3: Visualization
Create a plot showing your three data points and fitted hyperplane.

### Code Cell 2: Visualizations
- Actual vs predicted scatter plot
- Residual plot
- Residual histogram
- Summary statistics on residuals

### Question 4-5: Goodness of Fit (Extension)

**Part A**: Choose one metric:
- Sum of Squared Residuals: SSR = Σ(y_i - ŷ_i)²
- Sum of Absolute Residuals: SAR = Σ|y_i - ŷ_i|

Explain what it means geometrically in vector space.

**Part B**: Calculate by hand for your three data points.

**Part C**: Compute the gradient (partial derivatives) of the loss function with respect to all parameters. Show that at the optimum, the gradient is 0.

### Code Cell 3: Implement Goodness of Fit
- Calculate your chosen metric for the three sample points (by hand + code)
- Calculate for the full dataset
- Compute the gradient vector: ∇L(β) = -2X^T(y - Xβ)
- Verify it's close to 0 at the optimal solution

## Key Equations

### Linear Model
```
y = Xβ + ε
```
- y: target vector (shape: n_samples,)
- X: feature matrix (shape: n_samples, n_features+1 with bias column)
- β: parameter vector (shape: n_features+1,)
- ε: error vector (shape: n_samples,)

### Loss Function (OLS)
```
L(β) = (y - Xβ)^T(y - Xβ) = ||y - Xβ||²
```
Minimize this by finding where ∇L(β) = 0.

### Gradient
```
∇L(β) = -2X^T(y - Xβ)
```

### Normal Equation (Optimal Solution)
```
β* = (X^T X)^-1 X^T y
```

This is what sklearn's LinearRegression computes internally.

## Key Concepts

### Vector Space
A set of vectors where you can add vectors and scale them. The Titanic dataset is a vector space where:
- Each row (passenger) is a vector in high-dimensional space
- Each column (feature) is a dimension
- The hyperplane (regression fit) is a subspace within this space

### Matrix Operations
- X^T X: captures correlations between features (Gramian matrix)
- (X^T X)^-1: inverts correlations (this is why collinearity breaks it down)
- X^T y: correlations between features and target
- (X^T X)^-1 X^T y: combines all three to find optimal weights

### Residuals
The vertical distances from data points to the fitted hyperplane. In vector space, residuals are perpendicular to X.

### Why Sum of Squares?
- Squares penalize large errors more than small ones (outliers matter)
- Makes calculus cleaner (derivative is linear, easy to solve)
- Leads to the normal equation (closed-form solution exists)

## Files

```
PCW/Session 3 - Linear Algebra 2/
├── README.md                (this file)
├── pcw_lesson_3.ipynb       (your notebook)
└── [LLM transcripts to insert]
```

## How to Run

1. Open pcw_lesson_3.ipynb in Jupyter
2. Run the setup cell (imports + load Titanic data)
3. Do Question 1: Have LLM interview, paste transcript
4. Run Code Cell 1: Fit model, check parameters
5. Do Question 2: Deep LLM interview with math, paste transcript
6. Do Question 3: Create visualization
7. Run Code Cell 2: Generate residual plots
8. Do Question 4-5: Choose metric, calculate by hand
9. Run Code Cell 3: Compute goodness of fit measure + gradient

## Expected Output

After running all cells, you should have:
- A fitted linear regression model with 3 independent variables
- Model parameters (intercept + 3 slopes)
- R² and MSE showing model fit quality
- Visualizations of predictions vs actual and residuals
- A hand calculation of your chosen goodness-of-fit metric
- A gradient vector showing the slope of the loss function (should be close to 0)
- LLM transcripts showing your learning process

## Next Steps (Session 4 Preview)

**Topic**: Collinearity and Regularization

What happens when your independent variables are *correlated* with each other (collinear)? The matrix X^T X becomes nearly singular (hard to invert), and:
- Parameter estimates become unstable
- Small changes in data lead to huge changes in coefficients
- The normal equation breaks down

Solutions:
- Feature selection: drop correlated features
- Regularization: add a penalty term to the loss (Ridge, Lasso)
- Dimensionality reduction: PCA
- Domain knowledge: understand which features matter

---

**Last Updated**: Session 3, Fall 2026  
**Status**: Ready for Session 4
