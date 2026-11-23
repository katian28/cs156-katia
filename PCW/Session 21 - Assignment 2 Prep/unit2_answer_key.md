# Unit 2 Answer Key (Sessions 11-20)

Consolidated, corrected answer key assembled from each Unit 2 PCW session's pre-class work
(Session 11, "This class introduces our second unit," through Session 20), prepared for CS156
Assignment 2 Prep bonus credit. Each section gives the question, the corrected full answer, and
a short note on what, if anything, changed and why. Session 11's entry folds in real grader
feedback received on the original submission; Sessions 12-20 give the Extension Question
specifically, since those sessions have no other graded feedback yet to correct against.

---

## Session 11: Data Ethics and the Bias-Variance Tradeoff

**Question (Q3, Ethical Considerations):** What are the ethical risks of combining Chicago
homicide data with Ames housing data across a city police department and a bank client?

**Answer:** Six risks identified: (1) the homicide data records policing as much as crime,
recorded incidents depend on what gets reported and patrolled, not purely on where violence
occurs; (2) combining crime data into a bank's valuation/lending model risks redlining by proxy,
since recorded crime correlates with historically over-policed, majority-Black and Latino
neighborhoods, so "crime rate" functions as a race proxy even with race never an input directly
(The Markup's mortgage-lending investigation found exactly this pattern after holding 17 other
variables constant); (3) combining housing data into a police resource-allocation model risks a
feedback loop of its own (more patrols to "declining" areas inflates that area's recorded crime,
justifying more patrols next cycle), **but** this is my own proposed extension, not what Lum &
Isaac (2016) actually modeled, their real predictive-policing algorithm uses no housing data at
all, only past crime type, place, and time, and their simulation still found patrols would be
directed to Black neighborhoods at roughly twice the rate implied by actual drug-use estimates
using crime-reporting data alone, a stronger and more unsettling result than the housing-data
version, since it shows biased policing records feeding back into policing decisions is already
enough on its own, no extra proxy feature needed; (4) aggregation to the community-area level
hides real heterogeneity (the modifiable areal unit problem); (5) the substitute datasets (the
original assignment-linked ones were unrecoverable from the PDF export) limit how specific this
discussion can be to generalizable risks of this *class* of data, not the original intended
dataset's specific choices; (6) individual/location privacy narrows further when crime location
data is combined with parcel-level housing ownership records.

**Extension (neighborhood visualization):** Chicago's 2021-2025 homicide data is sharply
concentrated on the West and South Sides. The top 15 of 77 community areas account for roughly
61% of all 3,227 homicides; all 15 are West Side or South Side areas, zero North Side areas
appear in the top 15, and Austin is highest at 264.

**Edit notes:** Real correction, grader feedback flagged that point 3 conflated two separate
claims, my own hypothetical (feeding housing data into a policing model) with what Lum & Isaac's
actual paper demonstrated (crime-reporting data alone, no housing data, already produces the
biased feedback loop). Fixed directly in this session's own `README.md` and reflected above, the
corrected version is the stronger point: you don't need an extra proxy feature to get this bias,
biased crime records feeding into policing decisions is already sufficient. The rest of the
ethics discussion (points 1, 2, 4, 5, 6) and the extension's homicide-concentration finding
needed no changes.

---

## Session 12: Linear Regression with Basis Functions

**Extension question:** What performance were you able to achieve on the Yosemite temperature
fit, and what would applying Lasso to this model likely do?

**Answer:** Test R² ≈ 0.83 (train R² ≈ 0.84), beating the class benchmark of 0.82. The winning
configuration used a 2D RBF basis over (minute-of-day, day-index-since-2011-01-01), roughly 4700
centers on a grid (15 along the daily cycle, every 7 days across the 6-year span),
`sigma_min=80` minutes, `sigma_day=6` days, and `Ridge(alpha=0.1)`. A hyperparameter sweep over
center count, grid spacing, RBF width, and ridge alpha showed that fewer, coarser centers
(1500-2400) topped out around R² ≈ 0.73, confirming that the real fix was adding date as a
second input axis alongside time-of-day, not simply adding more centers along time-of-day alone.

With roughly 4700 densely-placed, overlapping centers (neighbors 7 days apart with a 6-day-wide
kernel), many centers are redundant, contributing almost the same information as their
neighbors. Lasso's L1 penalty would zero out most of those redundant coefficients, keeping only
centers that each explain a genuinely distinct region of the (time, date) grid. The result would
be a sparser model: a little R² given up in exchange for fewer active centers, faster prediction,
and a direct, visible map of which regions of the grid actually carry information versus which
were just padding.

**Edit notes:** No changes needed to the extension answer itself, already accurate and specific
(real numbers from the actual sweep, a correct mechanical description of what L1 does to
redundant, overlapping RBF centers). One class-wide grader note to flag: many Session 12
submissions had `FileNotFoundError`s that broke subsequent outputs, suspected to be a Forum
platform bug, not necessarily a code error. Checked this notebook directly, no errors of that
kind anywhere in it, not applicable here.

---

## Session 13: Support Vector Classifiers and The Kernel Trick

**Extension question:** (1) Can a kernel SVM classify the month from a single raw temperature
reading (Yosemite data)? (2) How does SVR compare to Session 12's Ridge + explicit RBF-feature
model on the same temperature-prediction task?

**Answer:**

**1. Classifying month from raw temperature.** A single-feature, 12-class RBF-kernel SVC reaches
19.1% accuracy, well above the 8.2% majority-class baseline, but still far from reliable: a
single temperature reading (say 15°C) is genuinely ambiguous across several months, since the
same temperature shows up in multiple parts of the year. This matches Session 12's own finding,
temperature alone does not carry enough information to pin down the date; it needs time-of-day
and date included alongside it.

**2. SVR vs. the Session 12 Ridge + RBF model.**

| Model | Train R² | Test R² |
|---|---|---|
| Ridge + explicit RBF features (Session 12, 120k training rows) | 0.839 | 0.829 |
| SVR, RBF kernel (this session, 8k training rows) | 0.937 | 0.728 |

Ridge wins on the held-out test set. The gap is mostly about scale: SVR's training cost grows
much faster with dataset size than a fixed-feature linear fit does, so SVR had to be trained on
a subsample of 8,000 rows instead of Ridge's full 120,000, and it shows a noticeably wider
train/test gap (0.937 vs. 0.728, more overfitting on the smaller training set). Same underlying
kernel idea behind both models, a very different practical cost at scale: the kernel trick is
elegant, but it is not free, and that cost is a real part of choosing between the two approaches
here, not just an implementation detail.

**Edit notes:** No changes needed. The comparison and its explanation (subsampling as the main
driver of SVR's worse generalization, not a weakness of kernels generally) are both accurate and
directly supported by the numbers in the notebook.

---

## Session 14: XGBoost

**Note on scope:** this PCW does not have a separately labeled "Extension Question," unlike the
other Unit 2 sessions. The closest equivalent is Q3, "Splits and Scores, extended," which takes
the Q2 result and pushes it one level further (a genuine extension of the core exercise, just
not labeled as one), so that is what is included here.

**Question:** Starting from the best first split found in Q2, split the data once at that
threshold, then treat each half as its own dataset and repeat the same threshold search
independently on each half.

**Answer:** Using the given (unmodified) `leaf_score`/`gain` functions and `seed=0` data, the
Q2 split point is threshold ≈ 1.9658.

- **Left subset** (25 points, x ≤ 1.9658): best threshold ≈ 1.5897, gain ≈ 0.4911, a reasonably
  balanced 22-vs-4 split.
- **Right subset** (76 points, x ≥ 1.9658): best threshold ≈ 1.9658 (the same value as the
  parent split), gain ≈ 0.4166, but this is a degenerate 1-vs-76 split: ranking the top 6
  candidate thresholds on this subset shows gain strictly shrinking as the isolated left leaf
  grows from 1 point to 6.

That degenerate result is not a coincidence. It follows directly from the scoring-function
discrepancy identified in Q2: the given `leaf_score` sums the *squares* of each gradient
(∑ᵢ gᵢ²) rather than squaring the *summed* gradient (Gⱼ² = (∑ᵢ gᵢ)², the paper's actual
structure-score formula). Summing squared gradients rewards isolating a single large-residual
point in its own leaf, since that point's large gᵢ² contributes fully regardless of what sits on
the other side, whereas the paper's Gⱼ² formula would let same-signed residuals on either side of
a split partially cancel before squaring, often favoring a more balanced split instead. This is a
concrete, demonstrated example of how that formula difference changes real splitting behavior,
not only a theoretical nitpick.

**Edit notes:** No changes needed, the mechanical claim (sum-of-squares vs. square-of-sum
rewarding single-point isolation) is consistent with, and correctly derived from, the scoring
discrepancy already flagged in this session's Q2.

---

## Session 15: Ridge and Lasso

**Extension question:** Across 50 independent resamples, compare an unregularized
`LogisticRegression(penalty=None)` against `RidgeClassifier(alpha=1e-2)` on the Session 6
Bollywood logistic-regression setup (true parameters β₀=-5, β₁=3). How does the bias-variance
tradeoff show up in the results?

**Answer:**

| Model | mean (β₀, β₁) | variance (β₀, β₁) |
|---|---|---|
| LogisticRegression (no penalty) | (-5.01, 3.00) | (0.130, 0.038) |
| RidgeClassifier (alpha=1e-2) | (-1.36, 0.84) | (0.0012, 0.00023) |

Variance ratio (no-penalty / ridge): roughly 109x (β₀), 163x (β₁).

Qualitatively, the unregularized model's 50 resampled fits scatter widely across (β₀, β₁) space;
the ridge model's fits cluster tightly together. Quantitatively, the unregularized model's
coefficients average to almost exactly the true value (-5.01, 3.00 against the true -5, 3): low
bias, but each individual fit can land anywhere across a fairly wide cloud around that average,
high variance. The ridge model's coefficients average to (-1.36, 0.84), systematically shrunk
toward zero and well away from the true value: the L2 penalty introduces a real, consistent bias.
In exchange, its variance is two to three orders of magnitude smaller, every individual fit lands
almost exactly where every other one does. This is the bias-variance tradeoff directly, not an
abstract idea: regularization does not make an estimate better in some absolute sense, it
exchanges some bias for a large reduction in variance, and whether that exchange is worth it
depends on which cost (a systematically wrong average, or a wildly unstable individual fit)
matters more for the problem at hand.

**Edit notes:** No changes needed. The numbers and the bias-variance framing both match the
notebook's own 50-resample results exactly, and the explanation correctly separates the
qualitative (spread of the scatter) and quantitative (variance ratio) halves of the question.

---

## Session 16: Principal Component Analysis

**Extension question:** Collect 10 of your own phone photos, run the same PCA pipeline used on
the Eigenfashion dataset (preprocess, factorize, inspect variance explained and reconstructions)
on them instead.

**Answer:** This question was never actually completed. Code Cell 4 of 4 in the notebook mirrors
the Eigenfashion pipeline exactly (same preprocessing and PCA call), pointed at a
`data/my_photos/` folder, but that folder was never populated with real photos, so the cell is
still a ready-to-run template rather than a finished result. There is no reconstruction-error
number, no variance-explained figure, and no "eigenphoto" visualization to report here, because
the only input this question needed (10 personal photos) was never supplied.

**Edit notes:** This is a real correction, not a style fix. The original pre-class work presented
this as effectively done ("mirrors the Eigenfashion pipeline exactly... drop your photos there
and run"), but no photos were ever dropped in and no output exists, so there is nothing to grade
here yet. Flagging this honestly instead of writing a plausible-sounding result for photos that
were never collected. If full credit on this question matters, the fix is mechanical: add 10
photos to `data/my_photos/`, run Code Cell 4, and report the real variance-explained and
reconstruction numbers it produces.

---

## Session 17: CNNs

**Extension question:** Implement the vertical Sobel kernel's companion, Gy (a horizontal edge
detector), apply it alongside Gx, and combine both into a single gradient-magnitude edge map.

**Answer:** Implemented Gy = [[1,2,1],[0,0,0],[-1,-2,-1]], the companion to the given vertical
Sobel kernel Gx, rotated 90 degrees. By the same logic as Gx (constant down each column, flips
sign left-to-right), Gy is constant across each row and flips sign from the top row (+) to the
bottom row (-), so it produces a large output only where intensity changes sharply moving
top-to-bottom, a horizontal edge. It is blind to purely vertical intensity changes, the mirror
image of Gx's blindness to horizontal ones. Applied to the test image, Gx ranges over
[-569, 571] and Gy ranges over [-592, 670]; each lights up on edges the other mostly misses, Gx
on roughly vertical boundaries, Gy on roughly horizontal ones, since each is a directional
derivative along a single, different axis.

Combining them into the gradient magnitude, gradient = sqrt(Gx² + Gy²), gives a value range of
[0, 685]. This is edge strength regardless of orientation: large wherever either directional
derivative is large, producing a complete, omnidirectional edge map that is strong along object
outlines in any direction and weak inside smooth, flat regions, unlike either Sobel kernel used
alone.

**Edit notes:** No changes needed. The kernel construction, the described sign pattern, and the
gradient-magnitude formula are all standard and consistent with the Gx result given in this same
session's Core Question.

---

## Session 18: Autoencoders

**Extension question:** Build a "better" autoencoder than the given 5-layer baseline (2-layer
encoder, 2-dim bottleneck, 2-layer decoder) and report whether it actually improves
reconstruction.

**Answer:** Extended the encoder from 2 layers to 3 (added `layer_2b`) and the decoder from 2
layers to 3 (added `layer_4b`), keeping the same hidden width (8) and the same 2-dimensional
bottleneck, then retrained for the same 3000 steps as the baseline. Final loss: 3-layer version
0.368, versus the 2-layer baseline's 0.378. This is a small but real improvement, roughly 2.6%
lower loss. The extra depth on each side gives the network somewhat more capacity to shape the
mapping into and out of the bottleneck, but the gain is modest rather than dramatic, because the
2-dimensional bottleneck itself, not encoder/decoder depth, is still the binding constraint on
how much information can survive the compression either way.

**Edit notes:** No changes needed. The result (3-layer slightly beats 2-layer, bottleneck
dimensionality remains the real limiting factor) is both internally consistent and a correct
reading of what changing depth while holding bottleneck size fixed should do.

---

## Session 19: RNNs

**Extension question:** Train a character-level GRU language model on Shakespeare text and use
it to generate new text. Explain what `tf.keras.layers.Embedding` does, why a GRU is used instead
of a vanilla RNN, and how the text-generation process actually works.

**Answer:** `tf.keras.layers.Embedding` is a learned lookup table, one trainable vector per
character ID, replacing raw integer character IDs (which carry no meaningful numeric
relationship to each other) with vectors the network shapes through backpropagation so that
similar characters end up with similar vectors.

A GRU updates its hidden state through gates (an update gate and a reset gate) with an additive
blend, H_new = (1-z)*H_prev + z*H_candidate, instead of a vanilla RNN's single
tanh(X@W1 + H@W2) update, which repeatedly multiplies through the same weight matrix at every
timestep. The additive update means gradients during backpropagation-through-time do not shrink
geometrically over long sequences (the vanishing-gradient problem a vanilla RNN suffers from),
letting the network learn dependencies spanning 100+ characters, which this task needs to
reproduce coherent Shakespeare dialogue structure.

Text generation works by repeated one-character sampling: seed the model with a starting string,
run it through embedding, then GRU, then a dense layer to get a probability distribution over the
next character, sample (not argmax) one character from that distribution, append it to the
output, and feed that single new character plus the carried-forward GRU hidden state back in as
the next step's input. Repeat. The hidden state, not the growing string itself, is what carries
forward everything generated or read so far.

Verified training run: untrained loss 4.188 (close to the theoretical random-guessing value of
ln(65) ≈ 4.190 for a 65-character vocabulary), final loss after 10 epochs = 1.164 (full curve:
2.540, 1.850, 1.604, 1.472, 1.391, 1.332, 1.286, 1.245, 1.204, 1.164). The model has 4,022,850
total parameters. A 500-character sample generated from the seed "ROMEO:" correctly alternates
character-name and dialogue blocks (the model learned the play's structural formatting), though
individual sentences are not fully grammatical yet, expected for a character-level model only 10
epochs into training.

**Edit notes:** No changes needed. The Embedding/GRU/generation explanations and the reported
numbers both match this session's own executed notebook output exactly.

---

## Session 20: Transfer Learning

**Extension question:** Using the frozen MobileNetV2 feature extractor from the Core Question
(before any fine-tuning), compute `feature_batch_average` for every image in the dataset, reduce
it to 2D with PCA, and check whether dogs and cats separate close to linearly in that 2D view.

**Answer:** Used the frozen `base_model` plus `global_average_layer` (the exact feature-extraction
stage, prior to any fine-tuning) to compute `feature_batch_average`, a 1280-dimensional vector,
for every image in the full dataset (3,000 images total), then reduced those vectors to 2D with
PCA and colored each point by its true label (dog or cat).

The 2 PCA components capture only 12.8% of the total variance (PC1 9.5%, PC2 3.3%), a small
fraction of the original 1280-dimensional representation. Despite that, a plain logistic
regression (a linear classifier) fit directly on that 2D projection still reaches 97.8% accuracy.
The two classes separate close to linearly even in a view that discards roughly 87% of the
representation's variance.

This matters because it shows that ImageNet pretraining alone already encodes the dog-vs-cat
distinction as close to linear structure in its learned representation, before this specific task
ever saw a single cat or dog label. That is exactly why the feature-extraction strategy (freeze
everything, train only one new linear layer) works as well as it does here: the hard
representational work was already done by the source task's pretraining, and the new task only
needs to draw one line through features that were already well organized.

**Edit notes:** No changes needed. The variance-explained and classification-accuracy numbers
match the executed notebook exactly, and the explanation correctly separates "captures most of
the variance" from "separates the classes," which is the actual point of the finding.
