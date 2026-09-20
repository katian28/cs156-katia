# Session 4 PCW: Trees 1 - Decision Trees and Gini Impurity

**Date**: Fall 2026, Session 4
**Student**: Katia Gwaneza Nkurunziza
**Status**: Complete

---

## Overview

Most real-world machine learning runs on tabular data, and decision trees are still the most common and often best-performing model for that kind of data. This session covers how trees choose splits using Gini impurity, why trees can be unstable, and why splits are always axis-aligned.

## Learning Objectives

- Understand how Gini impurity measures node purity
- Compute the best split (axis + threshold) by brute-force search over candidate thresholds
- Understand why decision trees are sensitive to small perturbations in data
- Understand why tree splits are always axis-aligned, and what that means for curved decision boundaries

## What's in the Notebook

### LLM Prompt and Summary
Tutor-style rundown of decision trees and Gini impurity, condensed to the core mechanics.

### Question 1: Basic Questions (Tree Instability)
Answers three questions based on the "Problem of Perturbations" reading:
- **1a**: Why small data changes cause big changes in splits (greedy, non-lookahead algorithm; cascading effect from root down)
- **1b**: Why perfectly fitting training data is bad (overfitting, high variance, unstable across resampling)
- **1c**: What splits look like in 2D (always axis-aligned, forming a grid of rectangles; curved boundaries need a staircase approximation)

### Question 2: Visualize the Data
Generates the two-moons dataset (`sklearn.datasets.make_moons`) and predicts by eye which axis the first split should use.

### Question 3: Find the Best First Split
Uses the provided Gini impurity functions to brute-force search every unique threshold on both axes.

**Result**: Best first split is on the **Y axis, threshold ≈ 0.147**, dropping weighted Gini impurity from 0.500 (unsplit) to 0.263.

### Question 4: Extension - Second-Level Splits
Splits the data into the two halves created by the first split, then finds the best split independently within each half.

**Results**:

| Level | Node | Axis | Threshold | Weighted Gini | Samples |
|-------|------|------|-----------|----------------|---------|
| 0 (root) | all data | Y | 0.1470 | 0.2627 | 500 |
| 1 | left (Y ≤ 0.147) | X | -0.7804 | 0.0731 | 210 |
| 1 | right (Y > 0.147) | X | 1.2375 | 0.1830 | 290 |

The root splits on Y (top/bottom), then both children split on X (left/right), tracing out a staircase approximation of the curved moon boundary.

## Key Concepts

### Gini Impurity
```
Gini(node) = 1 - sum(p_i^2)  for each class i
```
- 0 = pure node (one class only)
- 0.5 = maximally mixed (binary, 50/50)

### Weighted Gini Impurity (for evaluating a split)
```
Weighted Gini = (n_left/n_total) * Gini(left) + (n_right/n_total) * Gini(right)
```
Lower is better. The tree picks whichever axis/threshold minimizes this.

### Why Trees Are Unstable
Trees are built greedily: each split only optimizes the current node, never looking ahead. When two candidate splits have similar Gini scores, small data changes can flip which one wins, and that change cascades through every split beneath it.

### Why Splits Are Always Axis-Aligned
Each split only examines one feature at a time. In 2D this means every split is a horizontal or vertical line, never diagonal or curved. Curved decision boundaries require many small axis-aligned splits to approximate.

## Files

```
PCW/Session 4 - Trees 1/
├── README.md              (this file)
└── pcw_lesson_4.ipynb    (complete notebook, all questions answered, code verified)
```

## How to Run

```bash
cd "PCW/Session 4 - Trees 1"
jupyter notebook pcw_lesson_4.ipynb
```
Run all cells top to bottom. Uses `random_state=42` for reproducibility, so re-running gives identical splits.

## Next Steps (Session 5 Preview)

**Topic**: Max Likelihood 1 - Naive Bayes

- Moving from tree-based partitioning to probabilistic classification
- Connects back to Session 1's Bayesian probability extension problems

---

**Last Updated**: Session 4, Fall 2026
**Status**: Ready for submission
