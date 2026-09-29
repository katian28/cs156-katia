# Session 7 PCW: Networks 1 - Feed-Forward Neural Networks

**Date**: Fall 2026, Session 7
**Student**: Katia Gwaneza Nkurunziza
**Status**: Complete

---

## Overview

This session (and the next) introduces feed-forward neural networks: drawing basic architectures, writing them as nested functions, describing them with matrices, and running the forward pass. Backpropagation is introduced conceptually here, covered in depth next session.

See `study_materials/session_7_study_notes.md` for deeper notes on layers, units, activation functions, and the matrix view of a forward pass.

## Before Class: Readings

1. **Murphy, K. (2022), Chapter 13: sections 13.0, 13.1, 13.2, 13.3.1, 13.3.2** - basic matrix algebra of neural networks.
2. **3Blue1Brown (2017). *But what is a neural network?*** - visual intro to neurons, layers, weights, biases.
3. **3Blue1Brown (2017). *Gradient descent, how neural networks learn*** - intuition for the learning procedure, previewed here, covered fully next session.

## Important Notes: Bugs and Fixes

**Bug 1 in the given Keras/MNIST code**: uses `BinaryCrossentropy` loss with one-hot 10-class labels and no activation on the final `Dense(10)` layer. Result: accuracy stuck around 9-10%, the model essentially never learns. Fixed with `CategoricalCrossentropy` + `activation='softmax'` on the output layer, accuracy then reaches ~84% train / ~86% validation after 5 epochs.

**Bug 2 in the given Keras/MNIST code**: the validation call (`model.fit(..., validation_data=(x_val, y_val))`) crashes outright, `x_val` was never normalized and `y_val` was never one-hot encoded, so the target shape doesn't match the model's output shape. Fixed by preprocessing `x_val`/`y_val` exactly the same way as the training data.

**Missing dataset in the given perceptron code**: references a local `sonar.all-data.csv` that isn't provided. Fetched the same classic Sonar dataset at runtime from Jason Brownlee's own dataset repository (same author as the algorithm) instead.

## Learning Objectives

- Describe a neural network in terms of layers, units, activations, and connection weights
- Represent a forward pass as a sequence of matrix multiplications
- Understand the perceptron as the simplest possible neural network (one unit, one layer)
- Use PCA to make a simple model perform well on high-dimensional image data
- Practice catching bugs by actually running code, not just reading it

## Results Summary

| Network | Task | Result |
|---|---|---|
| Keras MNIST (broken, as given) | Classify digits 0-9 | ~9-10% accuracy (not learning) |
| Keras MNIST (fixed) | Classify digits 0-9 | ~84% train / ~86% validation accuracy |
| Perceptron on Sonar | Mine (M) vs. Rock (R) | 71.0% mean accuracy (3-fold CV) |
| Perceptron on FashionMNIST + PCA-10 | Top vs. Trouser | 97.5% test accuracy |

## What's in the Notebook

### Question 1: Basic Questions (Keras Network)
Answers a-k about the given MNIST network: data, task, layers, units, activation functions, weight initialization, connection weights, output, and performance, the last of which required actually running the code and finding the two bugs above.

### Code Cell 1: The Keras Network
Runs the code exactly as given (documenting the ~9-10% accuracy failure), then a corrected version reaching ~84-86% accuracy.

### Question 2: Draw the Network + Matrix Multiplication
A generated diagram (rectangles for each layer, arrows for connections) plus the full forward-pass equations: `h1 = sigmoid(x @ W1 + b1)`, `h2 = sigmoid(h1 @ W2 + b2)`, `y = softmax(h2 @ W3 + b3)`.

### Question 3: Core Questions (Perceptron)
Answers a-g about Jason Brownlee's from-scratch perceptron: task, single-unit architecture, step-function activation, zero-initialized weights, and what `predict()`/`train_weights()` actually do (same `error = actual - predicted` update rule as Session 2's from-scratch logistic regression).

### Code Cell 2: The Perceptron on Sonar Data
Runs end to end (after fixing the missing dataset), 71.0% mean accuracy across 3-fold cross-validation.

### Question 4: Draw the Perceptron + Matrix Multiplication
Diagram plus equations: `activation = x @ w + b`, `output = step(activation)`.

### Question 5: Extension - FashionMNIST + PCA
Filters FashionMNIST to two classes (T-shirt/top vs. Trouser), reduces 784 pixels to 10 PCA components, reuses the exact same perceptron structure from Question 3. Reaches 97.5% test accuracy.

## Key Concepts

### Layer vs. Unit
A **layer** is a group of units sharing the same inputs. A **unit** computes one weighted sum of its inputs plus a bias, then applies an activation function.

### Forward Pass as Matrix Multiplication
```
h = activation(x @ W + b)
```
Applied once per layer, output of one layer becomes input to the next. A perceptron is the degenerate case: one layer, one unit, a step-function activation instead of sigmoid/softmax.

### Why PCA Helped
A perceptron initialized at all-zero weights, updated with a small learning rate, converges slowly on 784 raw, noisy pixel values. PCA compresses those pixels into 10 components capturing 73.5% of the meaningful variance, letting the same simple update rule converge to a much better decision boundary, faster.

## Files

```
PCW/Session 7 - Networks 1/
├── README.md              (this file)
└── pcw_lesson_7.ipynb    (complete notebook, all questions answered, code verified, bugs documented)
```

## How to Run

```bash
cd "PCW/Session 7 - Networks 1"
jupyter notebook pcw_lesson_7.ipynb
```
Requires `tensorflow` (added to `pyproject.toml`). First run downloads MNIST and FashionMNIST (~50MB combined) and the Sonar CSV.

## Next Steps

**Preview**: Session 8, Gradients - Multivariate Derivatives. Backpropagation (how the weights in a multi-layer network actually get updated) builds directly on this session's matrix representation of the forward pass.

---

**Last Updated**: Session 7, Fall 2026
**Status**: Ready for submission
