# Assignment 2: Second Pipeline — Food-Photo Allergen Detector, Extended

**Due**: Tuesday, December 1, 2026
**Status**: Complete, pipeline executed and verified

See `pipeline.ipynb` for the full notebook. Extends Assignment 1 (`Assignments/Assignment 1 - First
Pipeline/`), reusing its data and base model, and adds three new pieces chosen directly from real
grader feedback on Assignment 1 (scored 4/4 on #cs156-MLFlexibility) plus the Session 21 PCW.

## 1. High-Level Explanation

Assignment 1 built a food-photo allergen detector and found two real weaknesses worth returning
to: it only caught 38% of real allergen photos at the default threshold, and its Grad-CAM
interpretability check looked at just 3 cherry-picked examples, too few to generalize from. The
grader's own written feedback named two specific directions to pursue: add more images and inject
noise to create synthetic examples, and test whether cropping each photo down to just its
Grad-CAM-attended pixels improves accuracy on the pretrained model. A third note asked the
natural follow-up question Assignment 1 left open: do these heatmaps actually align with the
allergen food, in general, not just in 3 examples?

This assignment answers all three, plus reuses the Session 21 PCW's convolutional autoencoder as
a fourth, independent feature representation. Each addition is a genuine experiment with an
honestly reported result, not all four "worked" in the sense of improving accuracy, and that's
reported directly rather than reframed.

## 2. Pipeline Diagram

![Pipeline diagram: 199 food photos and the Assignment 1 train/test split feed an explicit new 85/15 fit/validation split (needed for a fair augmentation comparison) and the Session 21 PCW autoencoder. The fit/validation split feeds data augmentation (rotate/flip/Gaussian noise, oversampling the allergen class) into Model A (baseline) and Model B (augmented), producing Comparison 1. Model A also feeds a Grad-CAM-guided masking experiment, producing Comparison 2, and a 20-photo manual alignment review, producing the novel Heatmap-Allergen Alignment Rate metric. The autoencoder's latent features feed a three-way representation comparison, Comparison 3. All three comparisons feed the final discussion.](pipeline_diagram.png)

## 3. Data

Same 199 food photos and `labels.csv` as Assignment 1 (`Assignments/Assignment 1 - First
Pipeline/`), reused directly via a relative path, not copied, so there is exactly one copy of the
raw data in the repository. Same stratified 75/25 train/test split (`random_state=42`), so every
new result below is directly comparable to Assignment 1's numbers. New for this assignment: an
explicit 85/15 fit/validation split drawn from the training portion (needed so the augmentation
comparison is fair, see Section 9 below and the notebook for why), and 192 synthetic augmented
photos generated from the 126 real fit-set photos (not stored as a separate dataset, generated
on the fly in the notebook so the augmentation recipe itself stays inspectable and reproducible).

## 4. GitHub Repo

https://github.com/katian28/cs156-katia/tree/main/Assignments/Assignment%202%20-%20Second%20Pipeline,
following the same notebook + README pattern used for Assignment 1 and every PCW session.

## 5. Equations: Models Not Covered in Class

Three techniques here were not covered in class, each explained in full in the notebook itself
(not just stated as an equation), per the assignment's own request to flag what's new clearly:

- **Data augmentation** (rotation as a 2D affine transform, horizontal flip, Gaussian noise
  injection $p' = \\mathrm{clip}(p + \\epsilon, 0, 255)$, $\\epsilon \\sim \\mathcal{N}(0, 12^2)$):
  full derivation and rationale in the notebook's "NEW: Data Augmentation" section.
- **Grad-CAM-guided masking** (per-image adaptive thresholding,
  $M_{i,j} = \\mathbb{1}[\\mathrm{heatmap}_{i,j} > \\mathrm{percentile}_{50}(\\mathrm{heatmap})]$,
  $\\mathrm{masked\\_image} = \\mathrm{image} \\odot M$): full procedure and the honest framing of
  what this experiment actually tests, in the notebook's "NEW: Grad-CAM-Guided Masking" section.
- **The autoencoder itself** (Session 21 PCW, reused not retrained here): convolutional
  encoder/decoder, reconstruction-MSE training objective, full equations in that session's own
  notebook (`PCW/Session 21 - Assignment 2 Prep/pcw_lesson_21.ipynb`).

## 6. Novel Metric: Heatmap-Allergen Alignment Rate (HAAR)

$$\\mathrm{HAAR} = \\frac{1}{N}\\sum_{i=1}^{N} \\mathrm{aligned}_i, \\qquad \\mathrm{aligned}_i = \\mathbb{1}[\\text{heatmap's hottest region overlaps the visible allergen food in photo } i]$$

A human-judged, semi-quantitative metric (no ground-truth bounding boxes exist for this dataset),
computed over a fixed-seed, reproducible sample of 20 of the 52 allergen photos, replacing
Assignment 1's 3-example eyeball check. **Result: 11/18 = 61.1%** (2 of 20 photos had no single
food item clearly identifiable as the allergen source and were excluded from the denominator,
the same honest stance Assignment 1 took on its one unverifiable example; counting those 2 as
misaligned instead gives the more conservative 11/20 = 55.0%). Full per-photo table and reasoning
in the notebook's "NEW: Novel Metric" section.

## 7-8. Figures

All in `pipeline.ipynb`: training curves (printed per-epoch, both Model A and Model B), 4-example
original-vs-masked comparison grid (`gradcam_masking_examples.png`), and 20 individual Grad-CAM
overlay images (`alignment_overlays/`) used for the manual alignment review.

## 9. Comparison Conditions

**1. Data augmentation.** Model A (baseline, trained on 126 real fit photos) vs. Model B (same
architecture, trained on 126 real + 192 synthetic photos, 3 synthetic copies per allergen photo,
1 per safe photo, shifting the fit set's allergen share from 26.2% to 41.5%). Result: ROC-AUC
0.713 → 0.738, accuracy 0.660 → 0.720, allergen catch rate unchanged at 8/13 (61.5%) both times.
Augmentation improved overall ranking ability but did not fix the specific catch-rate problem
Assignment 1 flagged. Full discussion in the notebook.

**2. Grad-CAM-guided masking.** Model A, unchanged, frozen, run on full test images vs. the same
test images masked down to each photo's own top-50%-attention region. Result: ROC-AUC 0.713 →
0.593, accuracy 0.660 → 0.560, catch rate 8/13 → 7/13. Masking **hurt** performance across every
metric, a genuinely informative negative result: the background context the mask removes was
contributing real signal the frozen MobileNetV2 base relies on, not noise it was ignoring.

**3. Feature representation (extending Assignment 1's Condition 2).** Same classifier (Logistic
Regression) on three representations: hand-crafted features (Assignment 1, 26-dim, ROC-AUC
0.505), raw flattened pixels (Assignment 1, 3072-dim, ROC-AUC 0.430), and the Session 21 PCW
autoencoder's latent bottleneck (NEW, 4096-dim, ROC-AUC 0.430). The autoencoder ties with raw
pixels, both below hand-crafted features; a modest negative result, reported honestly.

**Results table, new conditions:**

| Comparison | Condition | Test ROC-AUC | Test accuracy | Allergen catch rate |
|---|---|---|---|---|
| 1: Augmentation | A: baseline | 0.713 | 0.660 | 61.5% (8/13) |
| 1: Augmentation | B: augmented | 0.738 | 0.720 | 61.5% (8/13) |
| 2: Grad-CAM masking | Full image (Model A) | 0.713 | 0.660 | 61.5% (8/13) |
| 2: Grad-CAM masking | Masked (top 50% attention) | 0.593 | 0.560 | 53.8% (7/13) |
| 3: Representation | Hand-crafted (Assignment 1) | 0.505 | — | — |
| 3: Representation | Raw pixels (Assignment 1) | 0.430 | — | — |
| 3: Representation | Autoencoder latent (NEW) | 0.430 | — | — |

Plus the novel metric: **HAAR = 61.1%** (11/18 judgeable photos).

## 10. Discussion

**Which extension(s) actually helped, and why?** Of the three new techniques, only data
augmentation improved the model's headline metric (ROC-AUC), and even then, it left the
catch-rate problem Assignment 1 identified as the pipeline's real weak point untouched. Grad-CAM
masking made things measurably worse, a useful negative result: it shows the model's prediction
depends on more of the photo than its own gradient-based attention highlights, so Grad-CAM should
be read as "where the decision was most sensitive," not "everything the decision depends on." The
autoencoder's latent features, reused directly from the Session 21 PCW with no modification,
performed no better than raw pixels on this specific classification task, a reminder that a
representation learned for one objective (reconstruction) does not automatically transfer to a
different one (classification) just because it is lower-dimensional or learned by a neural net.

**What the novel metric adds.** The 61.1% Heatmap-Allergen Alignment Rate, computed over a
reproducible 20-photo sample rather than 3 cherry-picked examples, confirms Assignment 1's own
finding generalizes: the model's attention often does land on the real allergen food, but a
substantial minority of the time (39% of judgeable cases, and 2 more entirely unverifiable) it
clearly does not, landing on background, side dishes, or even a person's clothing. The two
clearest total misses (`IMG_9900.HEIC`, `IMG_7172.HEIC`) both coincided with the model's lowest
confidence or most clearly wrong predictions, a pattern worth testing formally (does alignment
correlate with correctness?) with a larger sample in a future assignment, not something this
20-photo sample has the power to confirm on its own.

**Honest limitations.** The augmentation comparison is not a perfectly isolated ablation (Model B
trains on more than twice the data over the same epoch budget, so it also receives more total
gradient updates, a realistic rather than a controlled-experiment comparison). The masking
experiment's mask is derived from the same model being re-tested on it, so it measures whether
background context helps that specific model, not a universal claim about Grad-CAM masking. The
alignment metric is human-judged, not pixel-precise, since no ground-truth bounding boxes exist
for this dataset. All three limitations are stated directly in the notebook alongside the results
they qualify, not buried in a single disclaimers section.

## 11. References

- Everything listed in Assignment 1's own References section (`Assignments/Assignment 1 - First
  Pipeline/README.md`), reused in full for the base model, Grad-CAM mechanism, and metrics.
- Session 21 PCW (`PCW/Session 21 - Assignment 2 Prep/`) for the autoencoder, trained there and
  reused here unmodified.
- Grader feedback on Assignment 1 (via the course forum, 2026-11-19), the direct source of this
  assignment's two main new techniques (data augmentation, Grad-CAM-guided masking) and the
  novel metric's motivating question.

## Files

```
Assignments/Assignment 2 - Second Pipeline/
├── README.md                      (this file)
├── pipeline_diagram.png           (pipeline diagram, Section 2)
├── pipeline.ipynb                 (full pipeline, executed, zero errors)
├── pipeline.html / .pdf
├── gradcam_masking_examples.png   (4-example original-vs-masked grid)
└── alignment_overlays/            (20 Grad-CAM overlay images, novel-metric review)
```

`workbook_answers.md` (if produced for the SLE submission form) would follow the same
not-committed rule as every other PCW/Assignment in this repo.

## How to Run

```bash
cd "Assignments/Assignment 2 - Second Pipeline"
jupyter nbconvert --to notebook --execute --inplace pipeline.ipynb
```

Needs `pillow-heif`, `tensorflow`, `scikit-learn`, `scipy` (already in this repo's `.venv`), and
both `Assignments/Assignment 1 - First Pipeline/` (for `food_photos/` + `labels.csv`, gitignored,
local only) and `PCW/Session 21 - Assignment 2 Prep/` (for the saved autoencoder latent features)
present relative to this folder. Full run (two TL training runs, Grad-CAM over ~70 images) takes
roughly 15-20 minutes on CPU.
