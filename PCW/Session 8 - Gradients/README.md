# Session 8 PCW: Gradients - Multivariate Derivatives

**Date**: Fall 2026, Session 8
**Student**: Katia Gwaneza Nkurunziza
**Status**: Complete

---

## Overview

Gradients are fundamental to modern machine learning because most models don't permit a closed-form analytical solution for the optimal parameters. This session dives into how backpropagation actually computes gradients: the chain rule applied gate by gate, backward through a computation graph.

See `study_materials/session_8_study_notes.md` for deeper notes on all readings.

## Before Class: Readings

1. **CS231n. *Backpropagation, Intuitions*.** Computation graphs and the chain rule, read up to "Modularity: Sigmoid example."
2. **Smilkov & Carter. *A Neural Network Playground*.** Interactive browser neural network.
3. **[Review] Murphy (2022), 13.3.1, 13.3.2 and 13.3.4 (new).** How backpropagation relates to reverse-mode autodiff.
4. **[Optional] Shafkat. *Intro to Machine Learning with JAX*.**
5. **[Optional] RitvikMath. *Jacobians*.**

## Learning Objectives

- Compute derivatives through a computation graph by hand, using only local gate rules
- Understand why multiply gates swap their inputs and add gates pass gradients through unchanged
- Generalize scalar backprop to vectors (`d(w^T x)/dw = x`, `d(w^T x)/dx = w`)
- Distinguish gradients (scalar-valued functions) from Jacobians (vector-valued functions), and understand why backprop avoids materializing full Jacobians
- Use JAX's automatic differentiation and verify it matches hand-derived gradients exactly
- Run gradient descent to optimize toward a target output

## Results Summary

| Computation | Result |
|---|---|
| Forward pass output | 0.7311 |
| Manual backward pass | d_w0=-0.1966, d_w1=-0.3932, d_x0=0.3932, d_x1=-0.5898 |
| Vectorized backward pass | d_w=[-0.1966, -0.3932], d_x=[0.3932, -0.5898], d_b=0.1966 |
| JAX autodiff | Matches exactly: d_w=[-0.1966, -0.3932], d_b=0.1966 |
| Gradient descent (1000 iters) | Output: 0.731 → 0.0017 |

## Important Note: JAX Version

The PCW pins an old JAX release (`0.4.13`) via a legacy Google Cloud Storage URL. Since JAX was already installed and working in this project (Sessions 6-7 verification, and this exercise only needs `jax.grad`, no version-specific feature), I used the already-installed modern version instead of the old pinned release.

## What's in the Notebook

### Question 1: LLM Interview on Gradients and Jacobians
Answered 3 quiz questions I had an LLM generate from the CS231n reading: what gradient descent is and why it's needed, why the chain rule makes backprop efficient, and the required question on the gradient-vs-Jacobian relationship. Includes reflection on where the LLM's first-pass answer was generic (defining Jacobians without really defining them) versus after being pushed for specifics.

### Q2A: Manual Derivatives (Scalar)
Full step-by-step backward pass through the classic CS231n sigmoid computation graph, computed by hand and verified in code. Key pattern: multiply gates swap their two inputs' derivatives; add gates pass gradients through unchanged.

### Q2B: Calculus Derivation
Derives `df/dw = x` for `f(w,x) = w^T x`, by symmetry with the given `df/dx = w`.

### Core Questions (3): Vectorized Sigmoid Example
Same computation graph, now with `w` and `x` as vectors instead of individual scalars, using the Q2B result directly in the backward pass.

### Preview: Auto-Magic Gradients (JAX)
Defines the forward pass once, lets `jax.grad` compute the backward pass automatically. Verified to match the hand-derived and NumPy-vectorized gradients exactly.

### Extension: Training
Runs 1000 iterations of gradient descent using the JAX-computed gradient, driving the sigmoid's output from 0.731 down to 0.0017.

## Key Concepts

### The Two Local Gate Rules
```
Multiply gate:  d(a*b)/da = b,  d(a*b)/db = a     (swaps its two inputs)
Add gate:       d(a+b)/da = 1,  d(a+b)/db = 1     (passes gradient through unchanged)
```
These two rules are the entire local computation needed at every step of backpropagation.

### Gradient vs. Jacobian
- **Gradient**: derivative of a scalar-valued function. One number out, a vector of partial derivatives.
- **Jacobian**: derivative of a vector-valued function. Many numbers out, a full matrix of partial derivatives.
- The network's final loss is scalar, so its overall derivative is a gradient. But the relationship between one layer's input vector and output vector is vector-to-vector, a Jacobian. Backprop never materializes that full Jacobian, it only ever computes Jacobian-vector products, collapsing back to a vector at every step, which is exactly what makes it efficient.

### Vector Generalization
```
f(w, x) = w^T x
df/dx = w
df/dw = x
```
Each variable's derivative equals the *other* variable's value, the vector version of the scalar multiply-gate swap rule.

## Files

```
PCW/Session 8 - Gradients/
├── README.md              (this file)
└── pcw_lesson_8.ipynb    (complete notebook, all questions answered, code verified)
```

## How to Run

```bash
cd "PCW/Session 8 - Gradients"
jupyter notebook pcw_lesson_8.ipynb
```
Requires `jax` (already in `pyproject.toml` per environment; installed via `pip install "jax[cpu]"` if needed elsewhere).

## Next Steps

**Preview**: Session 9, Metrics and Cross-Validation. This session's gradient machinery underlies how every model in Unit 1 was actually fit; Session 9 shifts to evaluating whether a fitted model generalizes, not just whether training loss went down.

---

**Last Updated**: Session 8, Fall 2026
**Status**: Ready for submission
