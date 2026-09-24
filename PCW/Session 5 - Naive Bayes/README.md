# Session 5 PCW: Naive Bayes - SMS Spam Classification

**Date**: Fall 2026, Session 5
**Student**: Katia Gwaneza Nkurunziza
**Status**: Complete

---

## Overview

Naive Bayes is a generative classifier built on Bayes' theorem plus one simplifying assumption: every feature is independent of every other feature, given the class. This session applies it to a real, messy dataset: classifying SMS text messages as spam or ham (legitimate).

## Important Note: Dataset URL Fix

The original PCW's dataset URL (`milindsoorya/SpamClassifier-in-python`) returns a 404, that GitHub repo no longer exists. I found a working mirror of the same classic SMS Spam Collection dataset (`mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-`). Confirmed it's the intended dataset because vectorizing it produces exactly **8672 features**, matching the number stated in the assignment's Question 4.

The ham/spam counts differ slightly from the PCW's stated "615 spam and 4957 ham" (this mirror has 747 spam and 4825 ham, same total of 5572 rows), likely a different dedup pass of the same underlying UCI SMS Spam Collection dataset. All numbers in this notebook use the actual counts from the mirror I used.

## Before Class: Readings and Study Guide

Work through these before starting the PCW. See also `study_materials/session_5_study_notes.md` for deeper explanation.

### Starmer, J. (2020). *Naive Bayes, Clearly Explained!!!* (StatQuest)

Covers Naive Bayes with multiple, discrete classes. Focus specifically on the **smoothing term** (Laplace / additive smoothing) added near the end, this exists to prevent multiplying by a zero probability when a word never appeared in a class during training.

### Starmer, J. (2020). *Gaussian Naive Bayes, Clearly Explained!!!* (StatQuest)

Covers the variant of Naive Bayes used for **continuous** features (not word counts). Instead of counting word frequencies per class, it fits a normal (Gaussian) distribution to each feature within each class, then uses that distribution's probability density as the likelihood term.

### VanderPlas, J. (2016). *In Depth: Naive Bayes Classification*

From the Python Data Science Handbook. Walks through Gaussian, Multinomial, and other Naive Bayes variants with runnable code, and discusses when each variant is appropriate based on the feature type (continuous vs. count vs. binary).

**Why all three matter for this PCW**: the SMS classifier in this notebook uses `MultinomialNB` (word counts / TF-IDF, discrete-ish features). The Starmer Gaussian video and VanderPlas reading explain the sibling approach for continuous features, useful context for why sklearn has multiple `naive_bayes` classes (`GaussianNB`, `MultinomialNB`, `BernoulliNB`) and how to pick the right one for a given dataset.

## Learning Objectives

- Understand what makes Naive Bayes "naive" (conditional independence assumption)
- Distinguish joint probability from marginal probability
- Convert text into numeric features using TF-IDF
- Train and evaluate a MultinomialNB spam classifier
- Count model parameters directly from the independence assumption
- Compare Naive Bayes's decision boundary (linear) to decision trees' (axis-aligned boxes)

## Results Summary

| Metric | Value |
|---|---|
| Dataset size | 5,572 messages (4,825 ham, 747 spam) |
| TF-IDF features | 8,672 |
| Model parameters | 17,346 (8672 words x 2 classes + 2 priors) |
| Overall accuracy | 97.63% |
| Ham recall | 100% (no real messages ever blocked) |
| Spam recall | 82.3% (about 1 in 6 spam messages slips through) |
| Spam precision | 100% (never falsely flags ham as spam) |

## What's in the Notebook

### Question 1: Basic Concepts
Five conceptual questions on: what makes Naive Bayes naive, latent variables, where the independence assumption breaks (e.g. "not bad"), joint vs marginal probability, and when joint probability differs from the product of marginals.

### Code Cell 1 + Question 2: TF-IDF
Loads the SMS dataset and vectorizes it with `TfidfVectorizer`. Question 2 explains TF-IDF in ~100 words: term frequency (how often a word appears in *this* message) times inverse document frequency (how rare that word is *overall*), so common filler words get down-weighted and distinctive words get up-weighted.

### Code Cell 2 + Question 3: Train and Evaluate
Trains `MultinomialNB`, plots a confusion matrix heatmap. Question 3 reads the confusion matrix to explain the model's real-world behavior: conservative, blocks zero real messages, but lets some spam through.

### Question 4: Parameter Counting
Derives the total parameter count from first principles (per-word, per-class likelihoods + class priors = 17,346), then verifies it numerically against `model.feature_log_prob_.shape` and `model.class_log_prior_.shape`.

### Question 5: Geometric Interpretation
Explains that Naive Bayes, unlike trees' axis-aligned rectangular splits, produces a single **linear decision boundary** in high-dimensional word-space, because taking the log of the naive product-of-probabilities turns it into a weighted sum, the same mathematical shape as logistic regression.

### Extension: Top/Bottom 30 Tokens
Uses `model.feature_log_prob_` and `vectorizer.vocabulary_` to extract and plot the 30 words most and least associated with spam.

### Question 6: Pattern Interpretation
Top tokens are classic spam vocabulary (call, free, txt, now, mobile). Bottom tokens are rare, idiosyncratic words sitting at the Laplace smoothing floor, not because they're distinctly "hammy," but because the model almost never saw them in a spam context. Connects back to Question 1: the model can't detect that several top spam words appearing *together* is an even stronger signal, since it treats them as independent.

## Key Concepts

### Naive Bayes Classification Rule
```
P(class | words) is proportional to P(class) * product over all words of P(word | class)
```
Predict whichever class has the higher score.

### Why It's "Naive"
Assumes `P(word_1, word_2, ... | class) = P(word_1|class) * P(word_2|class) * ...`, treating all words as conditionally independent given the class. Real language has dependencies (grammar, co-occurring phrases) that this ignores.

### Parameter Count Formula
```
total_parameters = (n_features * n_classes) + n_classes
                  = (8672 * 2) + 2
                  = 17346
```

### Why the Decision Boundary Is Linear
Taking the log of the naive product turns multiplication into addition:
```
log P(spam | words) = log P(spam) + sum_i [ log P(word_i | spam) ]
```
This is a weighted sum of features, same shape as `w*x + b`, hence a linear boundary in high-dimensional space, not a staircase of boxes like a decision tree.

## Files

```
PCW/Session 5 - Naive Bayes/
├── README.md              (this file)
└── pcw_lesson_5.ipynb    (complete notebook, all questions answered, code verified)
```

## How to Run

```bash
cd "PCW/Session 5 - Naive Bayes"
jupyter notebook pcw_lesson_5.ipynb
```
Run all cells top to bottom. Requires internet access to fetch the dataset CSV.

## Next Steps (Session 6 Preview)

**Topic**: Max Likelihood 2 - Parameter Estimation

Digs deeper into how the probabilities Naive Bayes uses (like P(word | spam)) are actually estimated from data via maximum likelihood, and what smoothing (like the Laplace floor seen in the bottom-30 tokens plot) is doing mathematically.

---

**Last Updated**: Session 5, Fall 2026
**Status**: Ready for submission
