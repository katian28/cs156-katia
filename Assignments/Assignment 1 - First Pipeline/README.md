# Assignment 1: Food-Photo Allergen Detector

**Due**: Friday, October 16, 2026
**Status**: Complete, pipeline executed and verified

See `pipeline.ipynb` for the full notebook. See the approved plan at `.claude/plans/serialized-stirring-key.md` for the original design.

## 1. High-Level Explanation

I have food allergies (nuts, eggs, onions), and I kept wondering: could a photo of my food actually tell me, before I eat it, whether it's something I need to be careful with? And once I had a pile of labeled photos sitting there, a second question showed up on its own: do I actually eat allergy-risk foods more at certain times of day, or on certain days of the week? Not something I could answer by just thinking about my habits, I wanted to see it in the actual data.

So in practice: snap a photo before a meal, a classifier flags whether it probably contains an allergen, and over time those flags build into a personal report of when I'm actually at higher risk, not based on vibes, based on my own photo history.

Whether it works well is a separate, more honest question, and the short answer is: modestly, with some genuinely surprising detours along the way. That's most of what this write-up is actually about.

## 2. Pipeline Diagram

![Pipeline diagram: Photos app export leads to manual labeling, which branches into hand-crafted features, raw pixels, and resized 224x224 photos for transfer learning, all feeding a stratified train/test split, which feeds the comparison conditions (model comparison, transfer learning, feature-representation comparison, temporal pattern analysis), producing ROC-AUC results, a confusion matrix and threshold tuning step, an ensemble condition, allergen rate by time, and the AES novel metric.](pipeline_diagram.png)

## 3. Data

199 of my own food photos, exported from the Photos app (File → Export → Export Unmodified Originals, which keeps the EXIF timestamps, important for the temporal part later). I built a tiny labeling tool (`label_photos.ipynb`) that shows me one photo at a time and I tag it safe or allergen by hand. All 199 ended up usable: 147 safe, 52 allergen. The labeling rule itself: I marked a photo "allergen" if I could see or specifically recall an allergen ingredient (egg, nuts, shellfish, dairy) in that meal, based on visual judgment plus my own memory of what I'd ordered or cooked, not a nutrition database or ingredient list. That's a real source of label noise, as the next paragraph shows.

Here's the part I didn't expect going in: I caught myself making real labeling mistakes, and I only found out because the model told on me. After training everything below, I ran Grad-CAM (a tool that shows which part of a photo a model is actually looking at) on one of my own "correctly" classified allergen photos, and it wasn't looking at the egg at all. That made me suspicious enough to go back and re-check all 151 of my "safe" photos by hand. 4 of them actually had visible egg in them, and I'd just missed it the first time through. Fixed those, reran everything. Full story's in the Discussion, it ended up mattering more than I expected.

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

I also derived equations for the taught models (Logistic Regression's log-loss, the Decision Tree's Gini impurity), plus the CNN's convolution operation, the transfer-learned model's frozen-backbone forward pass, and Grad-CAM's channel-weighting formula, in `pipeline.ipynb`. Worth flagging honestly: the CNN and transfer-learned model share that same cross-entropy loss as Logistic Regression (Session 7 covered that), but their convolutional layers themselves are not something we'd covered, Session 7 was feed-forward networks, not convolution. I had to go learn what a sliding filter actually does on my own. Full equations are in the notebook.

## 6. Novel Metric: Allergen Exposure Score (AES)

$$\mathrm{AES}(\text{bucket}) = \frac{\sum_i \mathbb{1}[\text{allergen}_i] \cdot w_i}{\sum_i w_i}, \qquad w_i = 0.5^{\,\text{days\_ago}_i / 180}$$

The plain allergen rate treats a photo from a year ago the same as one from yesterday, which never felt right to me. AES is a recency-weighted version instead, more recent photos count more (180-day half-life, so a 6-month-old photo counts half as much as today's). It's meant to answer "what are my habits *now*," not "what have they been, averaged over years I don't really remember." It made a real difference too, the afternoon bucket went from 28.3% (plain rate) to 48.0% (AES), a gap the flat average completely hides.

## 7-8. Figures

All in `pipeline.ipynb`, in order: training curves and ROC curves for the from-scratch CNN and the three Condition-1 models, the transfer-learned model's own training curve, a representation-comparison ROC plot (Condition 2), Grad-CAM heatmap overlays for the top 3 highest-confidence correct predictions (the ones that caught my labeling mistakes and later surfaced a real confound example), bootstrap confidence-interval error bars, confusion matrices at both the default and the threshold I actually tuned, hand-crafted feature-importance bar charts, bar charts for the temporal pattern and the AES comparison, and an all-models-plus-ensembles bar chart.

## 9. Comparison Conditions

1. **Model comparison**: I ran hand-crafted features through Logistic Regression and a Decision Tree, and raw pixels through a small CNN. CNN won, ROC-AUC 0.640, against 0.505 (Logistic Regression) and 0.394 (Decision Tree), both near or below chance on this one test split. Baseline of always guessing "safe": 0.740 accuracy, 0.500 AUC by definition.
   I didn't stop there, I also ran 5-fold cross-validation, and it genuinely complicated my own story: averaged across folds, Logistic Regression actually scores 0.648 and the Decision Tree scores 0.600, both clearly above chance. So my single test split happened to be an unusually bad one for those two models, not proof they'd learned nothing. I think that's a more honest conclusion than the first one I would've reported.
   - **Extension**: I tried transfer learning, reusing a pretrained MobileNetV2 instead of training convolution from scratch, which pushed ROC-AUC to 0.682. I also tried data augmentation on top of it at first, then pulled it back out, I never actually set up a real with/without comparison, so I had no honest basis to claim it helped. Removing it didn't hurt the result, if anything it looks slightly better now.
   - At the default 0.5 cutoff, this model only catches 5 of my 13 real allergen test photos, 38%. I tuned the threshold on a held-out validation slice instead of guessing (0.30 and 0.35 actually tied exactly on that sweep, so the pick between them was arbitrary), and applying 0.30 to the test set raised the catch rate to 8 of 13, 62%, at the cost of more false alarms (10 → 14 out of 37 safe photos). Real tradeoff, not a free lunch. More in the Discussion.
2. **Feature-representation comparison**: same model (Logistic Regression), hand-crafted features (0.505) vs. raw flattened pixels (0.430), neither clearly wins, both are close to chance. Pixels only helped once paired with something that could actually exploit spatial structure, the CNN. Looking at feature importance, the hand-crafted models seem to be leaning on one dark, low-intensity color bin and edge density, not anything I'd call a clean allergen signal.
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

**The labeling audit is probably the most important thing in this whole project, so I want to explain it first.** I didn't go looking for mislabeled photos on purpose, Grad-CAM just handed me evidence that something was off, the model wasn't attending to a visible egg in its own top prediction, and that was weird enough to chase down. Checking all 151 "safe" photos by hand turned up 7 candidates, 4 were genuine mistakes (visible egg I'd marked safe), 3 checked out fine. Fixed the 4, reran the entire pipeline from scratch. I didn't expect an interpretability tool to end up auditing my own data quality, but that's exactly what happened, and I think it's a more interesting result than anything the model itself produced.

**I tried data augmentation, then took it back out, and I want to be upfront about why.** I added it to the transfer-learned model thinking it would help with overfitting. But I never actually ran a clean with/without comparison, so I had no real evidence it was doing anything, just an assumption. Keeping it in the final write-up without proof felt like exactly the kind of unearned claim I don't want to make. After removing it, ROC-AUC went up slightly (0.669 → 0.682) and the train/validation gap got smaller (91%/61% → 90%/70%). One run doesn't prove augmentation is useless, but it's honest to say: cutting it didn't hurt me here.

**The confusion matrix told me something the AUC number alone was hiding, and it's the limit I care about most.** At the default 0.5 cutoff, my model catches only 5 of 13 real allergen photos, missing 8, and wrongly flags 10 of 37 safe ones. An AUC of 0.682 sounds like "not great, but working." This confusion matrix shows what that actually means for the thing I said I wanted to build, a pre-meal safety check, it would miss most of the real allergens if I used it as-is. One fix available right now, without retraining anything, is a lower threshold: I tuned one on a validation slice I never touched during training, and applying it once to the test set raised the catch rate to 8 of 13, at the cost of more false alarms. A genuinely better model could also raise the catch rate, the two aren't mutually exclusive, but the threshold was the lever I could pull immediately. Whether it's worth pulling depends on how bad I think a missed allergen is compared to an extra false alarm, a judgment call, not something a metric can make for me.

**My interpretability results don't agree with each other, and I think that's more honest than if they did.** I now look at the top 3 highest-confidence correct allergen predictions instead of just one. One (`IMG_5760.HEIC`, pred=0.77) shows the model looking at what's probably the actual egg dish on the plate, genuine egg-detection. Another (`IMG_3679.HEIC`, pred=0.60) shows a plate with a visible boiled egg and what looks like beans, but the heatmap's hottest region sits over starchy, potato-or-plantain-like chunks instead, not the egg, not the beans, a direct, photographic example of the model keying off a confound. The third I honestly can't interpret from the image alone. My hand-crafted models still lean on dark, textured color bins, which lines up with the `IMG_3679` finding, not a universal pattern, but real evidence that dark, starchy, textured foods can act as a confound in at least some cases. Both pieces of evidence are real, backed by actual images now rather than just a color-histogram argument, and they point in slightly different directions. I'm not going to force them into one tidy story.

**Cross-validation made me walk back part of my own "chance-level" conclusion, and I think that's worth sitting with.** The single test split made Logistic Regression and the Decision Tree look like they'd learned nothing. Five-fold cross-validation says otherwise, averaged across folds, both score clearly above chance (0.648 and 0.600). My test split was apparently an unusually hard one for those two models specifically. The transfer-learned model's cross-validation score matched its test score closely, so that one seems more stable. Lesson for me: with a dataset this small, which exact photos land in the test set can change the verdict on a model, not just the precision of the number.

The temporal pattern didn't survive the chi-square check either (p=0.39, p=0.74), so I'm treating "I eat more allergens in the morning" as a hypothesis worth watching, not something I've actually proven about myself yet.

The reproducibility bug was annoying but worth catching: `tf.random.set_seed` alone wasn't enough to make TensorFlow deterministic, I had to add `set_random_seed` plus `enable_op_determinism` before my numbers would actually repeat run to run. And one more honest limit: I'm one person labeling 199 photos by hand, and I got 4 of them wrong out of 151, about 2.6%. Worth saying out loud instead of assuming my own labels were perfect ground truth.

**One methodology choice I want to explain, not just state: `class_weight="balanced"` vs. actually balancing the dataset.** Logistic Regression and the Decision Tree use `class_weight="balanced"` directly; the CNN and transfer-learned model use a manually specified weight dictionary instead, and both neural models actually fit on only 126 of the 149 training images, Keras holds out the trailing 15% (23 photos) for its own internal validation. All four address the same imbalance, just not identically. Weighting changes how much each training mistake counts, it doesn't give the model any new allergen photos to learn from, still the same ~39 in training either way. For models that optimize over the whole dataset smoothly (Logistic Regression, the CNNs), that's usually enough. A Decision Tree is pickier, each split only looks at the literal rows sitting at that node, and with just ~39 allergen rows total, one odd photo can swing which split looks best. That's a plausible explanation for why the Decision Tree is the one model that scored below chance here, but I never ran the balancing experiment that would actually test it, and the cross-validation results above already complicate the "below chance" framing (the Tree's mean CV AUC, 0.600, is comfortably above 0.5). I'm leaving this as an open follow-up, not a conclusion.

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
- Every TensorFlow model here (from-scratch CNN and the transfer-learned MobileNetV2) needs `tf.keras.utils.set_random_seed(42)` and `tf.config.experimental.enable_op_determinism()`, not just `tf.random.set_seed`, to actually reproduce. Confirmed by re-running the full notebook twice and getting identical output both times.
- The transfer-learned model uses 224x224 input (MobileNetV2's native size), separate from the from-scratch CNN's 32x32. I had to bump the resolution up for Grad-CAM to even be legible, at 96x96 the feature map is only 3x3, too coarse to show anything useful.
- Grad-CAM needed a workaround for Keras 3's stricter graph rules, I couldn't splice the pretrained model's internal layer directly into the full model's graph, so I rebuilt the post-base_model forward pass (pooling → dense) manually, reusing the exact trained weights.
- My labels got a real audit, not a one-time pass: 2 originally-skipped photos got labeled, and a Grad-CAM-prompted review of all 151 "safe" photos caught 4 mislabels. Pipeline got rerun from scratch after each fix, every number above is from that final, corrected run.
