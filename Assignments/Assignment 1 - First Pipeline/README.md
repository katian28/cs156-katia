# Assignment 1: Food-Photo Allergen Detector

**Due**: Friday, October 16, 2026
**Status**: Complete, pipeline executed and verified

See `pipeline.ipynb` for the full notebook. See the approved plan at `.claude/plans/serialized-stirring-key.md` for the original design.

## 1. High-Level Explanation

I have food allergies (nuts, eggs, onions), and I kept wondering: could a photo of my food actually tell me, before I eat it, whether it's something I need to be careful with? And once I had a pile of labeled photos sitting there, a second question showed up on its own: do I actually eat allergy-risk foods more at certain times of day, or on certain days of the week? I wanted to see it in the actual data instead of just guessing from how I remember my habits.

So in practice: snap a photo before a meal, a classifier flags whether it probably contains an allergen, and over time those flags build into a personal report of when I'm actually at higher risk, not based on vibes, based on my own photo history.

Whether it works well is a separate, more honest question, and the short answer is: modestly, with some genuinely surprising detours along the way. That's most of what this write-up is actually about.

## 2. Pipeline Diagram

![Pipeline diagram: Photos app export leads to manual labeling, which branches into hand-crafted features, raw pixels, and resized 224x224 photos for transfer learning, all feeding a stratified train/test split, which feeds the comparison conditions (model comparison, transfer learning, feature-representation comparison, temporal pattern analysis), producing ROC-AUC results, a confusion matrix and threshold tuning step, an ensemble condition, allergen rate by time, and the AES novel metric.](pipeline_diagram.png)

## 3. Data

199 of my own food photos, exported from the Photos app (File → Export → Export Unmodified Originals, which keeps the EXIF timestamps, important for the temporal part later). I built a tiny labeling tool (`label_photos.ipynb`) that shows me one photo at a time and I tag it safe or allergen by hand: a photo is "allergen" if I could see or specifically recall an allergen ingredient (egg, nuts, shellfish, dairy) in that meal, based on visual judgment plus my own memory of what I'd ordered or cooked, not a nutrition database or ingredient list. All 199 ended up usable: 147 safe, 52 allergen.

That labeling process had a real error rate, and I only found out partway through the project. After training everything below, Grad-CAM showed my model wasn't attending to a visible egg in one of its own correct predictions, suspicious enough to make me re-check all 151 "safe" photos by hand. 4 actually had visible egg I'd missed the first time, about 2.6% of that batch. I fixed those and reran the entire pipeline, every model, from scratch, an interpretability tool ending up auditing my own ground truth instead of just the model. More in the Discussion.

## 4. GitHub Repo

https://github.com/katian28/cs156-katia/tree/main/Assignments/Assignment%201%20-%20First%20Pipeline, following the same notebook + README pattern I've used for every PCW session.

## 5. Equations: Hand-Crafted Feature Extraction

We hadn't covered this in class, so I researched and wrote it myself: a standard color-histogram + edge-density technique for turning a photo into a small numeric vector instead of raw pixels.

**Color histogram** (per RGB channel, 8 bins, image resized to 128x128):

$$\mathrm{hist}[c][b] = \frac{1}{N} \cdot \left|\{\, i : x_{i,c} \text{ falls in bin } b \,\}\right|, \quad b \in \{0..7\},\ c \in \{R, G, B\}$$

where $N = 128 \times 128 = 16384$ pixels per channel, so each channel's 8 bins sum to 1 (24 features total).

**Edge density** (mean absolute pixel-to-pixel gradient, on grayscale, $\mathrm{gray}[i,j] = \mathrm{mean}(R,G,B)$ at that pixel):

$$dx = \mathrm{mean}_{i,j} \left| \mathrm{gray}[i, j{+}1] - \mathrm{gray}[i, j] \right|, \qquad dy = \mathrm{mean}_{i,j} \left| \mathrm{gray}[i{+}1, j] - \mathrm{gray}[i, j] \right|$$

That's 24 histogram numbers plus 2 edge-density numbers, 26 total, feeding Logistic Regression and the Decision Tree in Conditions 1 and 2. The histogram captures color, the edge terms capture how visually "busy" or textured a photo looks. Novel metric's equation is in section 6.

I also derived equations for the taught models (Logistic Regression's log-loss, the Decision Tree's Gini impurity), plus the CNN's convolution operation, the transfer-learned model's frozen-backbone forward pass, and Grad-CAM's channel-weighting formula, in `pipeline.ipynb`. Worth flagging honestly: the CNN and transfer-learned model share that same cross-entropy loss as Logistic Regression (Session 7 covered that), but their convolutional layers themselves are something we hadn't covered, Session 7 was feed-forward networks. I had to go learn what a sliding filter actually does on my own. Full equations are in the notebook.

## 6. Novel Metric: Allergen Exposure Score (AES)

$$\mathrm{AES}(\text{bucket}) = \frac{\sum_i \mathbb{1}[\text{allergen}_i] \cdot w_i}{\sum_i w_i}, \qquad w_i = 0.5^{\,\text{days\_ago}_i / 180}$$

The plain allergen rate treats a photo from a year ago the same as one from yesterday, which never felt right to me. AES is a recency-weighted version instead, more recent photos count more (180-day half-life, so a 6-month-old photo counts half as much as today's). It's meant to answer what my habits look like *now*, closer to my actual behavior than a flat average over years I don't fully remember. It made a real difference too, the afternoon bucket went from 28.3% (plain rate) to 48.0% (AES), a gap the flat average completely hides.

## 7-8. Figures

All in `pipeline.ipynb`, in order: training curves and ROC curves for the from-scratch CNN and the three Condition-1 models, the transfer-learned model's own training curve, a representation-comparison ROC plot (Condition 2), Grad-CAM heatmap overlays for the top 3 highest-confidence correct predictions (the ones that caught my labeling mistakes and later surfaced a real confound example), bootstrap confidence-interval error bars, confusion matrices at both the default and the threshold I actually tuned, hand-crafted feature-importance bar charts, bar charts for the temporal pattern and the AES comparison, and an all-models-plus-ensembles bar chart.

## 9. Comparison Conditions

1. **Model comparison**: I ran hand-crafted features through Logistic Regression and a Decision Tree, and raw pixels through a small CNN. CNN won, ROC-AUC 0.640, against 0.505 (Logistic Regression) and 0.394 (Decision Tree), both near or below chance on this one test split. Baseline of always guessing "safe": 0.740 accuracy, 0.500 AUC by definition.
   I didn't stop there, I also ran 5-fold cross-validation, and it genuinely complicated my own story: averaged across folds, Logistic Regression actually scores 0.648 and the Decision Tree scores 0.600, both clearly above chance. So my single test split happened to be an unusually bad one for those two models, not proof they'd learned nothing. I think that's a more honest conclusion than the first one I would've reported.
   - **Extension**: I tried transfer learning, reusing a pretrained MobileNetV2 instead of training convolution from scratch, which pushed ROC-AUC to 0.682. I also tried data augmentation on top of it at first, then pulled it back out, I never actually set up a real with/without comparison, so I had no honest basis to claim it helped. Removing it didn't hurt the result, if anything it looks slightly better now.
   - At the default 0.5 cutoff, this model only catches 5 of my 13 real allergen test photos, 38%. I tuned the threshold on a held-out validation slice instead of guessing (0.30 and 0.35 actually tied exactly on that sweep, so the pick between them was arbitrary), and applying 0.30 to the test set raised the catch rate to 8 of 13, 62%, at the cost of more false alarms (10 → 14 out of 37 safe photos). A real tradeoff. More in the Discussion.
2. **Feature-representation comparison**: same model (Logistic Regression), hand-crafted features (0.505) vs. raw flattened pixels (0.430), neither clearly wins, both are close to chance. Pixels only helped once paired with something that could actually exploit spatial structure, the CNN. Looking at feature importance, the hand-crafted models seem to be leaning on one dark, low-intensity color bin and edge density, a fuzzy signal at best.
3. **Temporal pattern analysis**: using real ground-truth labels (no model involved), allergen rate declines across the day, 40.0% in the morning down to 18.9% in the evening, and Monday's my highest-risk day (38.1%), Tuesday my lowest (18.8%). Looked like a real pattern. I ran a chi-square test to check, and it's not statistically significant at this sample size (p=0.39, p=0.74); even after merging the tiny late-night bucket into evening, morning's expected allergen count is still only ~2.74 (below the usual rule-of-thumb of 5), so treat the p-value as approximate. I'm glad I checked before reporting it as a real finding.
4. **Ensemble**: I initially weighted this by each model's AUC on the test set itself, which is a real bug, it meant the weights were chosen using the exact data the ensemble was then scored on. Fixed it by deriving weights from a genuinely held-out 23-photo validation slice instead. The corrected numbers: averaging all four models equally (0.526) still does worse than the transfer-learned model alone (0.682); the "smarter" AUC-weighted version, using the honest weights, does even worse (0.522). The validation-based rankings didn't transfer to the test set well (the Decision Tree ranked *best* on validation, 0.854, but *worst* on test, 0.394), a useful lesson about how noisy model rankings get with only 23 photos to rank them on. Combining models isn't automatically better here, with weights chosen honestly or not.

**Results table, all conditions:**

| Model / Approach | Test-set ROC-AUC | 5-fold CV mean AUC |
|---|---|---|
| Logistic Regression (hand-crafted features) | 0.505 | 0.648 |
| Decision Tree (hand-crafted features) | 0.394 | 0.600 |
| CNN, trained from scratch (raw pixels) | 0.640 | — |
| Transfer-learned MobileNetV2 | 0.682 | 0.698 (3-fold) |
| Logistic Regression (raw flattened pixels) | 0.430 | — |
| Ensemble, equal-weight (all 4 models) | 0.526 | — |
| Ensemble, AUC-weighted (weights from held-out validation) | 0.522 | — |

## 10. Discussion

**How well did this work, and why?** Modestly. My from-scratch CNN reached ROC-AUC 0.640, transfer learning improved that to 0.682, though bootstrap confidence intervals between the two still overlap ([0.485, 0.803] vs. [0.530, 0.829]), a weaker check than a formal paired significance test (which I didn't run), so any edge is small relative to the noise in a 50-photo test set. The confusion matrix is the most honest view: at the default 0.5 threshold, the transfer-learned model catches only 5 of 13 real allergen photos (38%), the gap between "AUC looks okay" and "this would actually catch allergens in a pre-meal check." Tuning the threshold on a held-out validation slice raised the catch rate to 8 of 13 (62%), at the cost of false alarms rising from 10 to 14 out of 37 safe photos, a real, quantified tradeoff. Cross-validation partly walked back my Logistic Regression and Decision Tree verdicts too: their single test-set AUCs (0.505, 0.394) looked near or below chance, but 5-fold CV means (0.648, 0.600) were both clearly above 0.5, that specific test split was unusually hard for those two models rather than evidence of zero learning. Neither ensemble beat the single best model either, even after I fixed a real test-leakage bug in how the weights were originally chosen (full numbers in section 9 above); with a dataset this small, averaging in weaker models costs more than it gains.

**Interpretability evidence is mixed, and that itself is informative.** Looking at the top 3 highest-confidence correct allergen predictions instead of just one: one (`IMG_5760.HEIC`, pred=0.77) shows attention on what looks like the actual egg dish, genuine egg-detection. Another (`IMG_3679.HEIC`, pred=0.60) shows the opposite, a plate with a visible egg and beans where the heatmap's hottest region sits over starchy, potato-or-plantain-like chunks instead, while the egg and beans both stay cool. The third I can't clearly interpret from the image alone. My hand-crafted models' top feature (`B_hist_bin0`, a dark/low-intensity color bin) lines up with that confound finding, dark, starchy, textured foods acting as a proxy in at least some cases, not a universal pattern. Visual-only allergen detection has a hard ceiling for allergens that aren't visually distinctive, no amount of model improvement fixes that, and this confounding-factor finding plus my low default-threshold catch rate are sharper, more useful limits than just "the AUC is modest."

**The temporal pattern didn't survive a statistical check** (p=0.39 time-of-day, p=0.74 day-of-week), so I'm treating "I eat more allergens in the morning" as a hypothesis worth watching, not something I've proven yet. I'm one person labeling 199 photos by hand too, and got roughly 2.6% of my "safe" labels wrong on the first pass (see section 3 above), worth disclosing rather than assuming my own ground truth was perfect. What would actually improve this: more labeled photos, to give the temporal analysis real power and narrow the bootstrap intervals enough to confirm whether transfer learning's edge is real; a properly controlled with/without comparison if I revisit data augmentation; ingredient-level labels instead of binary allergen/safe; and fine-tuning MobileNetV2's later layers now that I have a working transfer-learning pipeline to build on.

Full discussion, including every limit I could find, is in `pipeline.ipynb`.

## 11. References

- `pillow-heif` documentation, for HEIC format support
- scikit-learn documentation: `LogisticRegression`, `DecisionTreeClassifier`, `roc_auc_score`, `confusion_matrix`, `ConfusionMatrixDisplay`, `StratifiedKFold`
- TensorFlow/Keras documentation: `Sequential`, `Conv2D`, `set_random_seed`, `enable_op_determinism`
- Keras Applications documentation: `MobileNetV2`, transfer learning and fine-tuning guide (keras.io/guides/transfer_learning)
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (2017)
- SciPy documentation: `scipy.stats.chi2_contingency`; NumPy documentation: `numpy.random.Generator` (bootstrap resampling)
- Course materials: Session 2 (logistic regression), Session 4 (decision trees), Session 7 (neural networks), Session 9 (ROC-AUC, train/val/test, class imbalance, cross-validation)

## Files

```
Assignment 1 - First Pipeline/
├── README.md              (this file)
├── pipeline_diagram.png    (pipeline diagram image, referenced in section 2)
├── pipeline.ipynb          (full pipeline, executed, all 4 comparison conditions + AES)
├── label_photos.ipynb     (interactive labeling tool, used to produce labels.csv)
├── food_photos/           (199 exported photos, gitignored)
└── labels.csv             (199 labels, all usable, no skips, gitignored)
```

## Technical Notes

- `pillow-heif` handles iPhone's default `.heic` format directly.
- `labels.csv` and `food_photos/` are excluded from git, only code, notebook, and results get committed.
- Every TensorFlow model here (from-scratch CNN and the transfer-learned MobileNetV2) needs `tf.keras.utils.set_random_seed(42)` and `tf.config.experimental.enable_op_determinism()` to actually reproduce, `tf.random.set_seed` alone isn't enough. Confirmed by re-running the full notebook twice and getting identical output both times.
- The transfer-learned model uses 224x224 input (MobileNetV2's native size), separate from the from-scratch CNN's 32x32. I had to bump the resolution up for Grad-CAM to even be legible, at 96x96 the feature map is only 3x3, too coarse to show anything useful.
- Grad-CAM needed a workaround for Keras 3's stricter graph rules, I couldn't splice the pretrained model's internal layer directly into the full model's graph, so I rebuilt the post-base_model forward pass (pooling → dense) manually, reusing the exact trained weights.
- My labels went through a real audit across multiple passes: 2 originally-skipped photos got labeled, and a Grad-CAM-prompted review of all 151 "safe" photos caught 4 mislabels. Pipeline got rerun from scratch after each fix, every number above is from that final, corrected run.
