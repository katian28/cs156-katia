# Session 18: Autoencoders

## Overview

Question 1 (LLM dialog on nonlinear dimensionality reduction vs. PCA) was done directly in the
live workbook. The Core Question (autoencoding handwritten digits, 64-dim -> 2-dim bottleneck
-> 64-dim reconstruction) and Questions 2a-4 all run end to end in the notebook, zero errors,
every quoted number from real executed output. Extension Question (3-layer encoder/decoder) is
pending, template cells are in place, to be filled in next.

## Question 2a

5 layers total. `layer_3` is the bottleneck (2-dim output). Encoder = `layer_1`, `layer_2` (2
layers, before the bottleneck). Decoder = `layer_4`, `layer_5` (2 layers, after). Named by
analogy to compressing/decompressing a signal. Returns both `reconstruct` (needed for the
loss) and `bottleneck` (the actual useful compressed representation, small enough to plot
directly).

## Core Question: Autoencoding Handwritten Digits

Trained for 3000 steps (Adam, `1e-3`), verified final loss **0.378** (full loss curve in the
notebook). `sklearn.datasets.load_digits`, no external data needed.

## Question 3

Reconstructions are recognizable but blurry, fine stroke detail lost. Verified per-example MSE:
mean **0.055** (std 0.019). Per-digit-class breakdown (all verified): digit 3 reconstructs best
(MSE ~0.041), digit 7 reconstructs worst (MSE ~0.076). Structural limitation: a 2-number
bottleneck keeps only the most informative directions of variation, so visually complex/
variable digits lose more detail than simple, consistent ones.

## Question 4

Bottleneck vectors cluster by digit despite the model never seeing labels during training.
Verified: mean pairwise distance within the same digit class = **2.08**, across different
classes = **3.80**, nearly double. The clustering is a side effect of reconstruction quality
correlating with visual similarity, which correlates with digit label, not the model learning
classes directly.

## Extension Question: A better autoencoder (pending)

Template cells are in the notebook (empty, matching the workbook's blank state). Plan:
extend the encoder and decoder from 2 layers to 3 layers each (add one more 8-dim hidden layer
on each side), retrain for the same 3000 steps, and compare final loss to the 2-layer
baseline's 0.378.

## Super Extension Question: teacher forcing (optional, not attempted)

## Readings for this session

- Jordan, J. (2018). *Introduction to autoencoders* (undercomplete + denoising sections).
- Hinton, G. (2017). *Lecture 15.1: From PCA to autoencoders* / *15.2: Deep autoencoders*.
- *Building Autoencoders in Keras* (up to the sparsity-constraint section).
- [Optional] Murphy, K. P. (2022). *Probabilistic Machine Learning*, Ch. 20.3.1, 20.3.2.

See `study_notes.md` for a deeper writeup tying the bottleneck-forces-compression idea
(Jordan), the linear-autoencoder-equals-PCA claim (Hinton, Jordan, Keras blog), and what
autoencoders are actually useful for (denoising, visualization, per the Keras blog) together.

## Files

```
PCW/Session 18 - Autoencoders/
├── README.md
├── study_notes.md
├── pcw_lesson_18.ipynb
├── pcw_lesson_18.html
└── pcw_lesson_18.pdf
```

`workbook_answers.md` (paste-ready for the SLE form) exists locally but is intentionally
**not** committed, per standing rule against checking paste-ready workbook text into a public
repo.

## How to Run

```bash
cd "PCW/Session 18 - Autoencoders"
jupyter nbconvert --to notebook --execute --inplace pcw_lesson_18.ipynb
```

Runs standalone, no external data; `sklearn.datasets.load_digits()` ships with scikit-learn.
Needs `jax` and `optax` (already in this repo's `.venv`).
