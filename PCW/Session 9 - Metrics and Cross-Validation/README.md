# Session 9 PCW: Metrics and Cross-Validation

**Date**: Fall 2026, Session 9
**Student**: Katia Gwaneza Nkurunziza
**Status**: Complete (Assignment Prep section intentionally left as personal placeholders)

---

## Overview

Training a model is only part of the story, evaluating whether it generalizes is the other half. This session covers train/test/validation splits, why accuracy alone is a weak metric, ROC-AUC as a threshold-independent evaluation tool, and using a validation set to tune hyperparameters without biasing the final test score.

See `study_materials/session_9_study_notes.md` for deeper notes.

## Note on Assignment Prep Section

The three "Assignment Prep" questions ask about the specific personal dataset planned for Assignment 1. These are intentionally left as placeholders in the notebook, they require genuinely personal context (what data, what patterns, what could go wrong) that shouldn't be fabricated. Fill these in before submitting.

## Before Class: Readings

1. **Wilber & Werness. *The Importance of Data Splitting*.**
2. **Wilber. *Precision & Recall*.**
3. **Wilber. *ROC & AUC*.**
4. **Wilber & Croome. *Cross Validation*.**
5. **Brownlee. *A Gentle Introduction to k-fold Cross Validation*.**
6. **scikit-learn cross-validation documentation.**
7. **[Optional] Murphy (2022), 4.5.1 and 4.5.5.**

## Learning Objectives

- Understand why a random train/test split matters, and what goes wrong with a feature-based split
- See concretely why accuracy alone is misleading (a model that's actively bad at a task can still show moderate accuracy)
- Compute and interpret ROC-AUC as a threshold-independent metric
- Derive precision at a specific recall target from an ROC curve's FPR/TPR
- Use a validation set to tune hyperparameters without leaking information into the final test score

## Results Summary

| Metric | Logistic Regression | SVM (gamma=2, C=1) |
|---|---|---|
| Test accuracy (threshold=0.5) | 38.3% | 93.3% |
| Test ROC-AUC | 0.517 | 0.989 |
| Precision at TPR=1.0 | 64.2% | 89.5% |

| Hyperparameter Search | Result |
|---|---|
| Best params (validation set) | gamma=0.5, C=10 (tied with gamma=1, C=5) |
| Validation ROC-AUC | 0.962 |
| Test ROC-AUC (tuned) | 0.988 |
| Test ROC-AUC (original arbitrary params) | 0.982 |

## What's in the Notebook

### Assignment Prep
Three questions about the personal dataset for Assignment 1, left as placeholders (see note above).

### Code Cell 1-2: Data and Train/Test Split
Generate the two-circles dataset, split 80/20.

### Question 1 of 3 (Q2): Split Type
Explains why the default split is random (not stratified), and why splitting by a feature's own value risks distribution shift between train and test, even though this specific symmetric dataset masks the effect.

### Core Questions: ROC-AUC
Fits logistic regression and an SVM, shows logistic's straight-line decision boundary fails on the circular data (38.3% accuracy, worse than chance), while the SVM wraps around the inner circle (93.3% accuracy). Computes and plots ROC curves for both.

### Question 2 of 3: ROC Interpretation
Three sub-answers, all backed by computed numbers: which model is better and why (SVM, AUC 0.989 vs 0.517), whether a higher-AUC model can lose at some single threshold (yes in general, though not in this specific run, verified by checking if the curves cross), and precision at TPR=1 for both models (64.2% vs 89.5%), derived from FPR and the test set's known class counts.

### Extension: Train/Val/Test Split
Explains why hyperparameters can't be tuned on the test set (it would bias the final score), then implements a grid search over SVM's gamma/C on a held-out validation set, reporting the final tuned model's score on the untouched test set exactly once.

## Key Concepts

### Why Random Splits Matter
`train_test_split`'s default random shuffling keeps train and test as independent samples of the same underlying distribution. Splitting on a feature's value instead creates systematically different train/test regions, risking distribution shift if the dataset has any structure correlated with that feature.

### Accuracy vs. ROC-AUC
Accuracy at one fixed threshold (0.5) can be misleading. ROC-AUC summarizes performance across *every* threshold simultaneously:
- AUC = 0.5: no better than random guessing (logistic regression here)
- AUC = 1.0: perfect ranking of positives above negatives

### Precision at a Recall Target
```
At TPR (recall) = 1.0:
TP = all actual positives
FP = FPR * (number of actual negatives)
Precision = TP / (TP + FP)
```
Lets you answer "if I insist on catching everything, how many false alarms do I get?" directly from the ROC curve's FPR axis.

### Validation Set Protects the Test Set
Tuning hyperparameters directly on the test set selects whichever values happen to score best on that one sample, biasing the reported score optimistically. A validation set absorbs that search, keeping the test set untouched for one honest, final evaluation.

## Important Note: sklearn Deprecation Warning

`svm.SVC(..., probability=True)` triggers a `FutureWarning` in the installed sklearn version: the `probability` parameter is deprecated as of 1.9 and will be removed in 1.11, replaced by `CalibratedClassifierCV(SVC(), ensemble=False)`. The PCW's own code uses this exact pattern, so it's kept as-given (it still works correctly, just emits a warning), not treated as a bug to fix, since it's forward-compatibility guidance, not an error.

## Files

```
PCW/Session 9 - Metrics and Cross-Validation/
├── README.md              (this file)
└── pcw_lesson_9.ipynb    (complete notebook, code verified, assignment-prep left as personal placeholders)
```

## How to Run

```bash
cd "PCW/Session 9 - Metrics and Cross-Validation"
jupyter notebook pcw_lesson_9.ipynb
```

## Next Steps

**Preview**: Session 10, Unit 1 Review & Assignment 1 Prep. This session's evaluation tools (train/val/test splitting, ROC-AUC, hyperparameter tuning) are exactly what Assignment 1 requires applying to a personal dataset.

---

**Last Updated**: Session 9, Fall 2026
**Status**: Ready for submission (pending personal Assignment Prep answers)
