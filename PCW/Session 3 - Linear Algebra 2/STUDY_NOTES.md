# Session 3 Study Notes: Vector Spaces and Multivariate Regression

**Purpose**: Deep explanation of the three core concepts. Read this before class.

---

## Core Concept 1: Matrices Represent Vector Spaces

### What is a Vector?
A vector is an ordered list of numbers.

```
v = [3, 5, 2]  <- a 3D vector (3 numbers)
```

Each number is a "coordinate" in a dimension. This vector points to a location in 3D space.

### What is a Vector Space?
A collection of vectors where you can add them and scale them.

**Example**: Your Titanic data with 3 features (age, pclass, sibsp).

Each passenger is a vector:
```
passenger_1 = [25, 1, 1]    <- 25 years old, class 1, 1 sibling
passenger_2 = [34, 3, 0]    <- 34 years old, class 3, 0 siblings
passenger_3 = [18, 2, 2]    <- 18 years old, class 2, 2 siblings
```

Stack all passengers together and you get a matrix X (1000 × 3 if you have 1000 passengers).

```
X = [[25, 1, 1],
     [34, 3, 0],
     [18, 2, 2],
     ...]
```

**This matrix IS your vector space.** Each row is a point. All 1000 points together form a cloud in 3D space.

### What Does a Matrix Do?

A matrix is not just data storage. It's a *geometric transformation*.

When you multiply a vector by a matrix, you transform it:

```
v_new = A @ v_old
```

This can:
- Rotate the vector
- Scale it (make it bigger or smaller)
- Project it onto a subspace
- Translate it

**Key insight for regression**: The matrix X in y = Xβ transforms your feature vectors (the β weights) into predictions. Different β values rotate/scale/shift the hyperplane through your data.

---

## Core Concept 2: Multivariate Regression Fits a Hyperplane

### One Variable: A Line

Session 2 was fitting a line:
```
y = β₀ + β₁*x
```

This is 1D input, 1D output. Geometrically, you're fitting a line through 2D scatter plot.

### Three Variables: A Plane

Now we have:
```
y = β₀ + β₁*x₁ + β₂*x₂ + β₃*x₃
```

This is 3D input, 1D output. Geometrically, you're fitting a plane through a 3D cloud of points.

**Visualization**:
- Your data cloud: 1000 points in 3D space (each point is a passenger with 3 features)
- The regression model: a 2D plane in 3D space (the best-fit surface)
- Residuals: vertical distances from each point to the plane

**Why a plane?** Because with 3 independent variables, the model lives in 3D space. Adding a 4th variable makes it a 3D hyperplane in 4D space. And so on.

### Matrix Form: y = Xβ + ε

Write the entire dataset as matrices:

```
y = [y₁, y₂, ..., y₁₀₀₀]ᵀ          <- 1000x1 vector of fares (targets)

X = [[1, x₁₁, x₁₂, x₁₃],           <- 1000x4 matrix (bias column + 3 features)
     [1, x₂₁, x₂₂, x₂₃],
     ...]

β = [β₀, β₁, β₂, β₃]ᵀ              <- 4x1 vector (intercept + 3 slopes)

ε = [ε₁, ε₂, ..., ε₁₀₀₀]ᵀ          <- 1000x1 vector of errors (residuals)
```

The equation y = Xβ + ε means:
- Multiply matrix X (1000×4) by vector β (4×1) → get predictions (1000×1)
- Each prediction is: 1*β₀ + x₁*β₁ + x₂*β₂ + x₃*β₃
- Add errors ε to get actual y values

**Key shapes to remember**:
```
X: (n_samples, n_features+1)           = (1000, 4)
β: (n_features+1, 1)                   = (4, 1)
Xβ: (n_samples, 1)                     = (1000, 1)  <- predictions
y: (n_samples, 1)                      = (1000, 1)  <- actual targets
ε = y - Xβ: (n_samples, 1)             = (1000, 1)  <- residuals
```

---

## Core Concept 3: Finding the Best Hyperplane via Calculus

### The Loss Function

We want to minimize squared residuals:

```
L(β) = (y - Xβ)ᵀ(y - Xβ)
```

This is the sum of squared residuals in matrix form. Let me expand it:

```
L(β) = Σᵢ (yᵢ - ŷᵢ)²
```

Where ŷᵢ = β₀ + β₁*x₁ᵢ + β₂*x₂ᵢ + β₃*x₃ᵢ (the i-th prediction).

**Why squared?** 
- Penalizes large errors more than small ones (outliers matter)
- Makes the math cleaner (the derivative is polynomial, easy to solve)
- Leads to a closed-form solution (the normal equation)

### Taking the Derivative

To minimize L, we take the derivative with respect to β and set it to 0:

```
dL/dβ = -2Xᵀ(y - Xβ) = 0
```

This is a **gradient**: how fast does the loss change as we change β?

When gradient = 0, we're at a minimum (or maximum). For regression, it's a minimum.

### Solving for Optimal β

Set the gradient to 0:

```
-2Xᵀ(y - Xβ) = 0
Xᵀ(y - Xβ) = 0
Xᵀy - XᵀXβ = 0
XᵀXβ = Xᵀy
```

Multiply both sides by (XᵀX)⁻¹ (the inverse of XᵀX):

```
β = (XᵀX)⁻¹Xᵀy
```

**This is the normal equation.** It gives us the optimal β in one step (closed-form solution).

### Matrix Dimensions (The Key Question)

Let's trace through:

```
X:              (1000, 4)
Xᵀ:             (4, 1000)
XᵀX:            (4, 1000) × (1000, 4) = (4, 4)
(XᵀX)⁻¹:        (4, 4)
Xᵀy:            (4, 1000) × (1000, 1) = (4, 1)
β = (XᵀX)⁻¹Xᵀy: (4, 4) × (4, 1) = (4, 1)
```

So β is a (4, 1) vector: intercept + 3 slopes.

**Geometric insight**:
- XᵀX captures correlations between features (how features interact)
- (XᵀX)⁻¹ inverts those correlations
- Xᵀy captures correlations between features and target
- Together: (XᵀX)⁻¹Xᵀy solves for weights that best match the target while accounting for feature correlations

---

## Why This Matters: Collinearity (Preview)

What if two features are almost identical (perfectly correlated)?

Example: age_in_years and age_in_months. They say the same thing.

Then XᵀX has two rows that are nearly identical → the matrix is nearly singular (non-invertible) → (XᵀX)⁻¹ blows up → β estimates become unstable and huge.

**Session 4 will solve this** with regularization and feature selection.

---

## Summary: The Big Picture

1. **Matrices represent vector spaces**: Each row of X is a data point in high-dimensional space.

2. **Regression finds the best hyperplane**: β defines the shape and position of the hyperplane that minimizes squared residuals.

3. **Calculus finds the optimum**: Take the derivative of the loss function, set it to 0, and solve for β algebraically → normal equation β = (XᵀX)⁻¹Xᵀy.

4. **Matrix algebra does the heavy lifting**: Matrix multiplication, inversion, and transpose make the math work.

---

## Key Equations to Know

```
Model:                  y = Xβ + ε
Loss function:          L(β) = (y - Xβ)ᵀ(y - Xβ)
Gradient:               ∇L(β) = -2Xᵀ(y - Xβ)
Set gradient to 0:      XᵀXβ = Xᵀy
Normal equation:        β = (XᵀX)⁻¹Xᵀy
```

---

## Common Pitfalls

1. **Forgetting the bias column**: X should have a column of 1s for the intercept β₀.

2. **Transposing X**: X is (n_samples, n_features), not (n_features, n_samples).

3. **Thinking (XᵀX)⁻¹ always exists**: If features are collinear, XᵀX is singular (non-invertible). The normal equation breaks.

4. **Confusing matrix shapes**: Always check: is this (4, 1000) or (1000, 4)? Matters for multiplication order.

5. **Thinking residuals are random**: Residuals ε should be random noise, but if they're not (they have a pattern), your model is missing something.

---

## What to Focus On in the PCW

1. **Question 1**: Understand what vector spaces and matrix operations mean conceptually.

2. **Code Cell 1**: Fit a real model to Titanic data. See that sklearn's LinearRegression produces (intercept, slopes).

3. **Question 2**: Derive the normal equation from first principles. Push the LLM to show the math step-by-step. Correct any mistakes.

4. **Code Cell 2**: Look at residual plots. Are they randomly scattered? If not, what's the model missing?

5. **Question 4-5**: Calculate goodness-of-fit metrics by hand. Verify the gradient is 0 at the optimum.

---

## Next Class Preview (Session 4)

What happens when features are correlated (collinear)?

- XᵀX becomes nearly singular
- (XᵀX)⁻¹ explodes (huge numbers)
- β estimates become unstable
- Small changes in data cause huge changes in coefficients

**Solutions**:
- Feature selection: drop redundant features
- Regularization: add penalty term to loss (Ridge, Lasso)
- PCA: reduce dimensionality

This is why understanding the normal equation matters: you see exactly where collinearity breaks things.

---

**Study tip**: Work through the matrix dimensions yourself. Trace through β = (XᵀX)⁻¹Xᵀy with real shapes. Don't just memorize the equation.
