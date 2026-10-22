# Session 13: Support Vector Classifiers and The Kernel Trick

## Overview

Prep for the in-class deep dive on SVCs and the kernel trick: why the nullspace of the data matrix doesn't affect optimal weights, why those weights can always be written as a combination of data points (the representer-theorem idea that makes kernels possible), hand-derived kernel feature maps, a real 3-kernel SVM comparison on the UCI banknote authentication dataset, and two extension experiments applying SVC/SVR to the Yosemite temperature data from Session 12.

## Question 1: Basic Prep

**Nullspace of X.** Every weight direction $v$ with $Xv = 0$ predicts zero change for every training point, so adding it to an optimal $w^*$ doesn't touch the loss. It's invisible to the optimization and gets ignored (set to zero in minimum-norm solutions).

**Why optimal weights are a combination of data points.** Quick version: since the nullspace component of $w$ doesn't matter, the useful part of $w$ lives entirely in the row space of $X$, so $w = \sum_i \alpha_i x_i$. The rigorous version, checked against Murphy §14.4.3 (kernelized ridge regression): starting from the regularized primal $w = (X^TX+\lambda I)^{-1}X^Ty$, the matrix inversion lemma rewrites this as $w = X^T(XX^T+\lambda I)^{-1}y = \sum_i \alpha_i x_i$, an identity that holds even with no nullspace at all. Either way, predictions only ever need dot products between data points, which is what the kernel trick replaces with a kernel function, provided that kernel is a **Mercer kernel** (positive semi-definite), a condition the quick version glosses over.

**Generated SVC functions.** Two prompts and functions (simple linear SVC, RBF-kernel SVC), full text and code in the notebook, ready to bring to class.

**Deep dive prep, checked against the textbook (Murphy §14.4 and §14.5).** §14.4 is the kernel trick itself, generic, not SVM-specific, worked in detail for kernelized ridge regression. §14.5 is what SVMs specifically add on top: pairing the (already-kernelizable) regression/classification objective with hinge loss (classification) or $\epsilon$-insensitive loss (regression), specifically because those losses make the weight combination **sparse** (only support vectors get nonzero $\alpha$), unlike plain kernelized ridge regression which is dense. Full soft-margin primal, the sparse solution form, the kernelized prediction rule, and the general vs. isotropic radial basis kernel equation are all derived in the notebook, matched against the book's actual equations (14.58, 14.59, 14.61).

## Question 2: Core Questions

**Kernel parameterizations.** Linear kernel: $\phi(x) = x$ (no transform). Quadratic kernel $(a\langle x,x'\rangle + b)^2$, the general form that matches `SVC(kernel='poly')`'s actual `gamma`/`coef0` parameters: $\phi(x) = [a\,x_i^2,\ \sqrt{2}a\,x_ix_j \text{ for } i<j,\ \sqrt{2ab}\,x_i,\ b]$, every pairwise product of the original features, worked out explicitly for $d=2$ in the notebook.

**Dataset.** The guide links to [StackAbuse's SVM/kernel SVM tutorial](https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/), which uses the UCI banknote authentication dataset: 1372 banknote images reduced to 4 wavelet-transform features (variance, skewness, curtosis, entropy), predicting forged (0) vs. real (1).

**3-kernel comparison (linear / quadratic / radial basis):**

| Kernel | Accuracy |
|---|---|
| Linear | 0.9855 |
| Quadratic (poly, degree 2) | 0.9745 |
| Radial basis | **1.0000** |

Radial basis wins. Quadratic actually underperforms plain linear here, a reminder that a fancier kernel isn't automatically better, it has to match the real shape of the decision boundary.

## Question 3: Extension Questions

**1. Classify month from raw temperature (Yosemite data).** A single-feature, 12-class RBF-kernel SVC gets **19.1% accuracy**, well above the 8.2% majority-class baseline, but far from reliable: one temperature reading (say 15°C) is genuinely ambiguous between several months. Matches the Session 12 finding, temperature alone just doesn't carry enough information; it needs time-of-day and date alongside it.

**2. SVR vs. the Session 12 Ridge+RBF model.**

| Model | Train R² | Test R² |
|---|---|---|
| Ridge + explicit RBF features (Session 12, 120k training rows) | 0.839 | **0.829** |
| SVR, rbf kernel (this session, 8k training rows) | 0.937 | 0.728 |

Ridge wins. The gap is mostly about scale: SVR's training cost grows much faster with dataset size than a fixed-feature linear fit, so it had to be subsampled to 8,000 rows versus Ridge's 120,000, and it shows a wider train/test gap (more overfitting on the smaller set). Same underlying kernel idea, very different cost at scale, a practical lesson about the kernel trick: elegant, but not free.

## Readings for this session

- Rohrer, B. (2017). *How Support Vector Machines work / How to open a black box* (video) — SVM intuition and the kernel trick.
- 3Blue1Brown (2016). *Inverse matrices, column space and null space* (video) — null space background for Question 1.
- Murphy, K. (2012). *Machine Learning: A Probabilistic Perspective*, §14.4-14.5 — the kernel trick.
- Ritvikmath (2019). *SVM Kernels* (video) — kernel math for SVMs.
- Ranjan, C. (2017). *Data Science for Mortals: The Kernel Trick* (blog) — computational-savings intuition.
- [StackAbuse: Implementing SVM and Kernel SVM with Python's Scikit-Learn](https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/) — Core Questions code pattern and dataset.

## Files

```
PCW/Session 13 - SVM Kernel Trick/
├── README.md
├── pcw_lesson_13.ipynb
└── data/
    ├── data_banknote_authentication.txt
    └── CRNS0101-05-{2011..2016}-CA_Yosemite_Village_12_W.txt
```

## How to Run

```bash
cd "PCW/Session 13 - SVM Kernel Trick"
jupyter nbconvert --to notebook --execute --inplace pcw_lesson_13.ipynb
```

If `data/data_banknote_authentication.txt` is missing, re-download from `https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt`. The Yosemite files are the same ones used in Session 12.
