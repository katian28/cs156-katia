# Session 6 PCW: Max Likelihood 2 - Parameter Estimation for Logistic Regression

**Date**: Fall 2026, Session 6
**Student**: Katia Gwaneza Nkurunziza
**Status**: Complete

---

## Overview

Maximum likelihood estimation (MLE) is the principle behind fitting nearly every model in this course. This session applies it directly to logistic regression: each observation is treated as a weighted coin flip, and MLE finds the parameters that make the observed data most probable.

## Important Note: Bug Caught and Fixed

The PCW's own starter code for `log_likelihood_per_point` used `if y is True:`. Since `y` comes from `np.random.uniform() < prob`, it's a `numpy.bool_`, not Python's built-in `True` singleton, and `is True` silently never matches a `numpy.bool_`. Every point fell into the `else` branch regardless of its real label, producing wrong log-likelihoods (I first computed -715.01 for Aishwarya's guess before catching this, when the correct value is -2305.95). Fixed by using plain truthiness (`if y:`) instead of identity comparison. Full explanation and fix are in the notebook, in the Interpretation cell right after Code Cell 3.

## Learning Objectives

- Distinguish probability (function of data) from likelihood (function of parameters)
- Understand logistic regression as fitting a Bernoulli (coin-flip) probability per observation
- Derive the likelihood and log-likelihood for logistic regression by analogy to repeated coin flips
- Compute and compare log-likelihoods to judge which of two wrong parameter guesses is "more wrong"
- Compute gradients of the log-likelihood with JAX, and understand what they do and don't tell you

## Results Summary

| Metric | Value |
|---|---|
| True parameters | beta_0 = -5, beta_1 = 3 |
| Aishwarya's guess | beta_0 = 3, beta_1 = -3, log-likelihood = -2305.95 |
| Irhum's guess | beta_0 = 4, beta_1 = -1, log-likelihood = -1823.41 |
| True parameters' log-likelihood | -346.82 |
| More wrong guess | Aishwarya |
| Gradient at Aishwarya's guess | [95.09, 751.65] |
| Gradient at Irhum's guess | [-456.39, -291.62] |

## What's in the Notebook

### Readings
Five sources on probability vs. likelihood, maximum likelihood, MLE under model misspecification, logistic regression's Bernoulli interpretation, and Murphy Ch. 4's derivation of MLE for the mean and linear regression.

### Question 1: Logistic Regression Setup
What `x`, `beta_0`/`beta_1`, and `y` mean in `y = sigma(beta_1*x + beta_0)`. Key distinction: `x` varies per observation, parameters are fixed across the whole dataset, `y` is a conditional probability, not a hard prediction.

### Question 2: Likelihood by Analogy
Derives the likelihood and log-likelihood for logistic regression from the coin-flip analogy: each observation contributes `P(True|x)` or `P(False|x)` to a running product, exactly like a sequence of coin flips with a different bias per flip.

### Core Question: Likelihood Calculation
- Code Cell 1: generates synthetic data from known true parameters (beta_0=-5, beta_1=3), commented line by line
- Code Cell 2: visualizes the generated data
- Code Cell 3: implements `log_likelihood_per_point` and `total_log_likelihood`, compares Aishwarya's and Irhum's guesses. **Aishwarya is more wrong.**

### Extension: Gradient
- Code Cell 4: uses JAX's `jax.grad` to compute the exact gradient of the log-likelihood with respect to both parameters, vectorized (no Python for-loop), commented line by line
- Interpretation flags a subtlety: the gradient at Aishwarya's guess points toward *increasing* beta_0, even though the true beta_0 is much lower, gradient ascent only gives the locally steepest direction, not a straight line to the true optimum

## Key Concepts

### Probability vs. Likelihood
Same formula, different fixed variable:
- **Probability**: `P(data | fixed parameters)`, a function of the data
- **Likelihood**: `L(parameters | fixed data)`, a function of the parameters

### Logistic Regression's Likelihood
```
L(beta_0, beta_1) = product over all i of:
                       sigma(beta_1*x_i + beta_0)        if y_i is True
                       1 - sigma(beta_1*x_i + beta_0)    if y_i is False

ln L(beta_0, beta_1) = sum over all i of:
                          ln(sigma(beta_1*x_i + beta_0))       if y_i is True
                          ln(1 - sigma(beta_1*x_i + beta_0))   if y_i is False
```

### Why Log-Likelihood, Not Raw Likelihood
Multiplying 1000+ probabilities (each less than 1) underflows to numerically indistinguishable-from-zero very fast. Taking logs turns the product into a sum, avoiding underflow, and since log is monotonic, the parameters that maximize likelihood also maximize log-likelihood.

### Gradient as Local Direction, Not a Compass to the Truth
`jax.grad` computes the exact local slope of the log-likelihood surface at a given point. It tells you which way to nudge parameters to locally increase log-likelihood right now, not the direction of the true, possibly far-away, optimum. Repeated small steps (gradient ascent) eventually get there; a single gradient does not point straight at the answer.

## Files

```
PCW/Session 6 - Max Likelihood 2/
├── README.md              (this file)
└── pcw_lesson_6.ipynb    (complete notebook, all questions answered, code verified, bug documented)
```

## How to Run

```bash
cd "PCW/Session 6 - Max Likelihood 2"
jupyter notebook pcw_lesson_6.ipynb
```
Run all cells top to bottom. Requires `jax[cpu]` installed (`pip install "jax[cpu]"`).

## Next Steps

**Preview**: Unit 1 Review & Assignment 1 Prep. This session's MLE-for-logistic-regression machinery connects directly to Session 2's from-scratch logistic regression (gradient descent) and Session 5's Naive Bayes (also fit via maximum likelihood, just with a different, closed-form estimator instead of gradient-based search).

---

**Last Updated**: Session 6, Fall 2026
**Status**: Ready for submission
