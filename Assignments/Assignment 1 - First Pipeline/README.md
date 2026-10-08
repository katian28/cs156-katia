# Assignment 1: Food-Photo Allergen Detector

**Due**: Friday, October 16, 2026
**Status**: Complete, pipeline executed and verified

See `pipeline.ipynb` for the full notebook. See the approved plan at `.claude/plans/serialized-stirring-key.md` for the original design.

## 1. High-Level Explanation

Can I visually detect allergen-containing foods (nuts, shellfish, dairy) in my own food photos, and does a reliable classifier reveal patterns in when I actually eat them? Automation framing: a pre-meal allergen check, snap a photo, get a quick visual risk flag, plus a personal pattern report of higher-risk times.

## 2. Pipeline Diagram

```
Photos app export (manual, EXIF preserved)
        |
        v
Manual labeling (allergen / safe / skip) -- interactive tool, label_photos.ipynb
        |
        v
Feature extraction: hand-crafted (color histogram + edge density) AND raw pixels
        |
        v
Train/test split (stratified, 75/25) --> [3 comparison conditions] --> ROC-AUC, temporal analysis
```

## 3. Data

199 food photos exported manually from the Photos app (File → Export → Export Unmodified Originals, which preserves EXIF timestamps). 197 usable after excluding 2 "skip" (not food / unclear) photos: 149 safe, 48 allergen (24.4% positive rate). Labeled by hand through an interactive tool (`label_photos.ipynb`), one photo at a time, a genuine curation step rather than a one-click download.

## 4. GitHub Repo

This folder, in `katian28/cs156-katia`, following the same notebook + README pattern used for every PCW session.

## 5. Equations

See the Logistic Regression normal-equation/gradient-descent derivation in Session 2's notebook, reused here unchanged. The novel metric equation is in section 6 below.

## 6. Novel Metric: Allergen Exposure Score (AES)

```
AES(bucket) = sum_i [ is_allergen_i * recency_weight_i ] / sum_i [ recency_weight_i ]
recency_weight_i = 0.5 ^ (days_ago_i / 180)
```

A recency-weighted allergen rate per time-of-day bucket, 180-day half-life (a photo from 6 months ago counts half as much as one from today). Unlike a plain historical rate, it answers "what are my habits *now*," not "what have they been, averaged over years." See `pipeline.ipynb` for the full derivation and results, recency weighting shifted the afternoon bucket from 24.4% (plain rate) to 44.3% (AES), a real, non-trivial difference the flat average hides.

## 7-8. Figures

Training curves (CNN accuracy over 30 epochs) and ROC curves (all three Condition-1 models) are in `pipeline.ipynb`, cells 8 and 15-18 (bar chart for temporal pattern, bar chart for AES comparison).

## 9. Comparison Conditions

1. **Model comparison** (hand-crafted features, Logistic Regression vs. Decision Tree, vs. CNN on raw pixels): CNN wins, ROC-AUC 0.636, vs. 0.535 (Logistic Regression) and 0.456 (Decision Tree, worse than chance). Baseline (always predict "safe"): accuracy 0.760, AUC 0.500.
2. **Feature-representation comparison** (Logistic Regression held constant): hand-crafted (26-dim) ROC-AUC 0.535 vs. raw flattened pixels (3072-dim) ROC-AUC 0.515, nearly identical. Raw pixels only helped when paired with the CNN's convolutional architecture, not with a linear model.
3. **Temporal pattern analysis** (real ground-truth labels, n=184 with valid timestamps): allergen rate declines across the day, morning 40.0% → midday 29.7% → afternoon 24.4% → evening 18.9%. Monday highest (38.1%), Tuesday lowest (19.4%).

## 10. Discussion

The classifier is modestly better than random (best AUC 0.636) but not reliable, consistent with a small (~150 image), from-scratch training set on visually heterogeneous allergen categories. The CNN shows real overfitting (84% train vs. 61% validation accuracy). A genuine reproducibility finding surfaced during verification: the CNN's result varied run to run (0.671, then 0.522) even with `tf.random.set_seed` fixed, requiring `tf.keras.utils.set_random_seed` + `tf.config.experimental.enable_op_determinism` to get a stable, repeatable number (0.636, confirmed across two consecutive full re-runs). The temporal analysis, which doesn't depend on classifier accuracy at all, revealed the clearest actionable finding: allergen consumption skews toward mornings and Mondays. Full discussion, including honest limits of visual-only allergen detection, is in `pipeline.ipynb`.

## 11. References

- `pillow-heif` documentation, for HEIC format support
- scikit-learn documentation: `LogisticRegression`, `DecisionTreeClassifier`, `roc_auc_score`
- TensorFlow/Keras documentation: `Sequential`, `Conv2D`, `set_random_seed`, `enable_op_determinism`
- Course materials: Session 2 (logistic regression), Session 4 (decision trees), Session 7 (neural networks), Session 9 (ROC-AUC, train/val/test, class imbalance)

## Files

```
Assignment 1 - First Pipeline/
├── README.md              (this file)
├── pipeline.ipynb          (full pipeline, executed, all 3 comparison conditions + AES)
├── label_photos.ipynb     (interactive labeling tool, used to produce labels.csv)
├── food_photos/           (199 exported photos, gitignored)
└── labels.csv             (199 labels, gitignored)
```

## Technical Notes

- `pillow-heif` handles iPhone's default `.heic` format directly.
- `labels.csv` and `food_photos/` are excluded from git, only code, notebook, and results get committed.
- The CNN requires `tf.keras.utils.set_random_seed(42)` and `tf.config.experimental.enable_op_determinism()` (not just `tf.random.set_seed`) for a reproducible result, confirmed by re-running the notebook twice and getting identical output both times.
