# Session 21: Assignment 2 Prep

## Overview

Prep for CS156 Assignment 2 (due 2026-12-01), which extends the Assignment 1 pipeline (Food-Photo
Allergen Detector) with a model/technique from Unit 2. Chosen extension: a **convolutional
autoencoder**, trained unsupervised on the same 199 food photos, used both to check reconstruction
error against Assignment 1's own Grad-CAM confound finding and to produce a third feature
representation for later comparison.

- Question 1 (two-slide presentation): `presentation.pdf`.
- Core Question (tools/datasets research + re-implementation): `pcw_lesson_21.ipynb`.
- Extension Question (Unit 2 answer key, bonus): `unit2_answer_key.md`.

## Core Question: Convolutional Autoencoder on Food Photos

Adapted from the standard Keras/TensorFlow conv-autoencoder pattern (`Conv2D` encoder,
`Conv2DTranspose` decoder, trained on reconstruction MSE), applied to 64x64 RGB versions of the
199 food photos from Assignment 1, reusing that assignment's HEIC-loading pipeline directly. No
labels used anywhere in training.

**Verified results** (full run, zero errors):
- Train/val split: 170 train, 29 validation photos.
- Final reconstruction MSE: 0.00586 (train), 0.00698 (validation).
- Reconstructions are recognizably blurry versions of the originals (see `reconstructions.png`),
  not noise, confirming the model actually learned something.

**Reconstruction error by label** (computed across all 199 photos):

| Label | Mean recon. MSE | Std | Count |
|---|---|---|---|
| allergen | 0.005414 | 0.001646 | 52 |
| safe | 0.006218 | 0.002514 | 147 |

Welch's t-test: t=2.608, **p=0.0101**, statistically significant. Safe photos reconstruct *worse*
than allergen photos, the opposite of what a naive "rare class = higher error" guess would
predict. The more likely explanation: "safe" spans a much wider variety of food types (147 photos
across salads, fruit, rice, bread, etc.) than "allergen" (52 photos concentrated around a smaller
set of egg/nut/dairy-containing dishes), so the autoencoder's limited bottleneck fits the more
visually consistent class better, not because it's rarer but because it's more homogeneous.

**Checking the three Grad-CAM example photos directly:**

| Photo | Recon. MSE | Error rank (of 199) | Note |
|---|---|---|---|
| `IMG_5760.HEIC` | 0.00659 | 64 (top 32.2%) | clean egg-detection case |
| `IMG_3679.HEIC` | 0.00624 | 76 (top 38.2%) | the confound case (heatmap missed visible egg) |
| `IMG_8426.HEIC` | 0.00454 | 144 (top 72.4%) | unverifiable attention case |

None of the three land in the extreme tail. The confound photo (`IMG_3679.HEIC`) does rank higher
than the unverifiable one, but not dramatically, so this doesn't give a strong independent
confirmation of the Grad-CAM finding. Worth reporting honestly in Assignment 2 rather than
overstating it: reconstruction error here tracks general visual unusualness, not allergen-specific
confounding, those are related but different signals.

**Reusable for Assignment 2**: `autoencoder_latent_features.npy` (199 x 4096, the flattened
encoder bottleneck for every photo) and `food_photo_autoencoder.keras` (the trained model), both
saved directly in this folder, ready to drop into a three-way feature-representation comparison
(hand-crafted vs. CNN/transfer-learned vs. autoencoder bottleneck).

## Extension Question: Unit 2 Answer Key (bonus)

Full consolidated, corrected answer key across all 10 Unit 2 sessions (11 through 20, Session 11
is where the course "introduces our second unit") in `unit2_answer_key.md`. Two real corrections
surfaced:
- **Session 11**: real grader feedback flagged that the ethics discussion's point 3 conflated my
  own hypothetical (feeding housing data into a policing model) with what Lum & Isaac (2016)
  actually demonstrated (their real algorithm uses no housing data at all, only past crime type/
  place/time, and still produces the same biased feedback loop). Fixed directly in that session's
  own `README.md`, the corrected point is the stronger one: biased crime-reporting data alone is
  enough, no extra proxy feature needed.
- **Session 16**: the PCA extension (meant to run on 10 of my own phone photos) was written up as
  complete but the photos were never actually added and the cell was never run, flagged honestly
  rather than left as an unverified claim.

## Assignment 2 Direction (not built yet, noted for later)

Per discussion: Assignment 2 is planned to combine three Unit 2 techniques together, not just the
autoencoder above:
1. **Autoencoder** (this session's Core Question).
2. **Data augmentation with a real with/without comparison** — Assignment 1 tried augmentation
   once, then pulled it back out without ever testing it properly; this time the plan is to
   actually measure the difference. Katia's own idea (layering/overlaying one food photo on top
   of another) is the mixup/cutmix family of augmentation, not plain rotation/flip.
2. **Fine-tuning MobileNetV2's later layers** — explicitly flagged as unfinished future work in
   Assignment 1's own Discussion section, now has a working transfer-learning pipeline to build on.

## Files

```
PCW/Session 21 - Assignment 2 Prep/
├── README.md
├── presentation.pdf             (2-slide deck, Question 1)
├── pcw_lesson_21.ipynb           (Core Question, executed, zero errors)
├── pcw_lesson_21.html / .pdf
├── unit2_answer_key.md           (Extension Question, bonus)
├── training_curve.png
├── reconstructions.png
├── recon_error_by_label.png
├── autoencoder_latent_features.npy   (reusable for Assignment 2)
└── food_photo_autoencoder.keras      (reusable for Assignment 2)
```

`workbook_answers.md` (paste-ready for the SLE form) exists locally but is intentionally **not**
committed, per standing rule against checking paste-ready workbook text into a public repo.

## How to Run

```bash
cd "PCW/Session 21 - Assignment 2 Prep"
jupyter nbconvert --to notebook --execute --inplace pcw_lesson_21.ipynb
```

Needs `pillow-heif`, `tensorflow`, `scipy` (already in this repo's `.venv`), and Assignment 1's
`food_photos/` + `labels.csv` present at `../../Assignments/Assignment 1 - First Pipeline/`
(gitignored, local only). Full run (load + 60 epochs + analysis) takes a few minutes on CPU.
