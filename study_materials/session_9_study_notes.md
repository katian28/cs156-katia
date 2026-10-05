# Session 9 Study Notes: Metrics and Cross-Validation

**Purpose**: Deep explanation of why evaluation is a separate problem from fitting, and how to do it honestly. Read before class.

---

## Core Concept 1: Fitting a Model and Evaluating It Are Different Problems

Every session so far (2, 6, 7, 8) focused on one question: how do you find parameters that minimize a loss on the data you have? This session asks a completely different question: once you have those parameters, how do you know if they're actually *good*, in the sense of working on data you haven't seen yet?

These are different problems because a model can always get the training loss arbitrarily low by memorizing the training data (think of an extremely deep decision tree from Session 4 that creates one leaf per training point). Low training loss doesn't imply the model learned anything general, it might have just memorized noise. The only way to tell the difference is to test the model on data it never got to adjust itself around, which is the entire reason train/test splits exist.

---

## Core Concept 2: Why a Random Split, Specifically

`train_test_split` shuffles the data randomly before dividing it. This isn't arbitrary, it's protecting a specific assumption: that both the train set and the test set are representative samples of the *same* underlying population.

**What breaks if you don't shuffle randomly**: splitting by a feature's own value (e.g. "first feature > 0 goes to train, < 0 goes to test") creates two systematically different subpopulations instead of two random samples of one population. In the circles dataset this happens to be relatively harmless, since the data is rotationally symmetric around the origin, a model trained only on the right half still sees the same local geometry it would need for the left half. But in a dataset with any real-world asymmetry (which is most real data), this kind of split would mean the model never sees an entire region of the input space during training, and the test score would no longer reflect genuine generalization, it would just reflect "does the model work on the one specific region it never got to train on," which is a different, much harsher question than "does this model generalize."

---

## Core Concept 3: Why Accuracy Alone Is a Weak Metric

The reading's example (a "no lightning now" detector being accurate 99%+ of the time) generalizes to any imbalanced classification problem: a model can rack up high accuracy by exploiting class imbalance, always predicting the majority class, without learning any actual signal.

**This session's own example makes a sharper point**: logistic regression's accuracy on the circles dataset was 38.3%, *worse* than the 50% you'd expect from random guessing on a balanced binary problem. This isn't just "accuracy hides imbalance," it's proof that a single accuracy number, computed at one arbitrary threshold (0.5), can make a genuinely broken model look like it's merely "pretty bad" instead of revealing that it has essentially zero ability to rank points correctly. The ROC-AUC score (0.517, right at the 0.5 "no better than random" baseline) tells the true story: logistic regression isn't failing because of a bad threshold choice, it's failing because a straight line fundamentally cannot separate two concentric circles, at any threshold.

---

## Core Concept 4: What ROC-AUC Actually Measures

A classifier like logistic regression or an SVM with `probability=True` doesn't just output a hard 0/1 label, it outputs a probability (or probability-like score) between 0 and 1. The usual rule is "predict class 1 if that probability is above 0.5," but 0.5 is just one arbitrary cutoff among infinitely many possible ones.

**The ROC curve plots what happens across every possible threshold**: for each cutoff from 0 to 1, compute the true positive rate (TPR, the fraction of actual positives correctly caught) and false positive rate (FPR, the fraction of actual negatives incorrectly flagged). As the threshold sweeps from 1 down to 0, you catch more and more positives (TPR rises) but also accumulate more false alarms (FPR rises too). The ROC curve traces out this whole tradeoff as one line.

**ROC-AUC is the area under that curve**, a single number summarizing the whole curve:
- AUC = 1.0: there exists a threshold where every positive scores higher than every negative, perfect separation
- AUC = 0.5: the model's scores carry no information about the true label, equivalent to random guessing
- AUC < 0.5: the model is systematically backward (rare, and basically fixable by flipping the prediction)

**Why this is more honest than accuracy**: accuracy commits to one threshold before you've even decided what tradeoff between false positives and false negatives you actually want. ROC-AUC sidesteps that choice entirely, it measures the model's intrinsic ability to rank positives above negatives, independent of wherever you eventually decide to draw the line.

---

## Core Concept 5: Precision at a Chosen Recall, From the ROC Curve Directly

The PCW's third ROC question asks something very concrete: "if we want to catch every single positive (TPR = 1), what fraction of our 'positive' predictions would actually be correct?" That's asking for **precision** at a specific **recall** (TPR) target, and you can compute it directly from the ROC curve's own axes without retraining anything.

```
At the point where TPR = 1.0:
  TP = (every actual positive, since we caught them all)
  FP = FPR * (number of actual negatives)
  Precision = TP / (TP + FP)
```

**Worked numbers from this session**: the test set had 34 actual positives and 26 actual negatives. For logistic regression, reaching TPR=1.0 required accepting FPR≈0.731, meaning `FP = 0.731 * 26 ≈ 19`, giving `precision = 34 / (34+19) ≈ 64.2%`. For the SVM, reaching TPR=1.0 only required FPR≈0.154, `FP = 0.154*26 ≈ 4`, giving `precision = 34/(34+4) ≈ 89.5%`. Same recall target (catch everything), very different cost in false alarms, exactly the tradeoff the ROC curve's shape encodes.

**Why "if TPR is the Y-axis, what's on the X-axis" is the right hint**: the X-axis (FPR) is precisely the piece of information missing from TPR alone. TPR alone tells you nothing about how many false alarms you paid to achieve it, you need both axes together, plus the known class counts, to get precision.

---

## Core Concept 6: Can a Higher-AUC Model Lose at a Single Threshold?

Yes, in general, even though it didn't happen in this session's specific run. ROC-AUC is an *aggregate* summary across the entire curve. Two ROC curves can cross each other: one model might dominate at low FPR (conservative, few false alarms) while another dominates at high FPR (aggressive, catches more positives), and whichever one has the larger *total area* wins on AUC even though it's not uniformly better everywhere.

**I verified this directly for this session's data**: comparing the SVM's and logistic regression's TPR at every matching FPR value, the SVM's curve never dips below logistic's, SVM dominates at every single threshold here. That's a property of this specific pair of models and this specific dataset, not something ROC-AUC guarantees in general. If you ever need a model that's reliably best at one *specific* operating point (not just in aggregate), you have to check the curves directly at that point, AUC alone can't tell you that.

---

## Core Concept 7: Why Hyperparameter Tuning Needs a Validation Set

Model *parameters* (like a logistic regression's weights, or a decision tree's split points) get fit directly from the training data via the methods covered in earlier sessions. *Hyperparameters* (like an SVM's `gamma` and `C`, or a tree's max depth) aren't fit that way, they're chosen by trying different values and seeing which works best. That "seeing which works best" step is itself a form of fitting, just one level up, and it has the exact same contamination risk as fitting parameters directly: if you choose hyperparameters based on test set performance, you've let information about the test set leak into your model selection process.

**The three-way split solves this cleanly**:
- **Train** (here, `train2`, 192 points): fit the model's actual parameters for a given hyperparameter choice.
- **Validation** (48 points): compare many different hyperparameter choices against each other, using their validation performance. This is the "practice test set," you can look at it as many times as you want while searching.
- **Test** (60 points): touched exactly once, at the very end, after hyperparameters are already locked in. This is the only number that honestly estimates real-world generalization.

**Grid search, concretely**: this session tried every combination of 6 gamma values and 6 C values (36 total models), trained each on `train2`, scored each on `val` via ROC-AUC, and kept whichever scored highest (`gamma=0.5, C=10`, tied with `gamma=1, C=5`, validation AUC ≈ 0.962). Only after that choice was locked in did the final model get evaluated on the untouched test set (AUC ≈ 0.988), a small improvement here over the original arbitrary `gamma=2, C=1` (AUC ≈ 0.982), since this dataset was already fairly easy for an SVM regardless of exact hyperparameters. The size of the improvement isn't the point, the discipline of never touching the test set until that final single evaluation is.

---

## Common Questions and Confusions

**Q: If logistic regression's accuracy (38.3%) is below 50%, doesn't that mean it's "anti-correlated," actively useful if you just flip its predictions?**

A: Not quite, and this is exactly what ROC-AUC near 0.5 (not below it) reveals. An AUC of 0.517 means the model's probability *scores* carry essentially zero ranking information, not that they're systematically backward. The 38.3% accuracy figure is a symptom of picking a poor fixed threshold (0.5) for a model whose probabilities aren't meaningfully separating the classes at all, flipping the predictions wouldn't reliably help, since there's no consistent signal to flip in the first place. If AUC had been well below 0.5 (like 0.1), that genuinely would suggest flipping predictions could help.

**Q: Why use ROC-AUC instead of just reporting accuracy at several different thresholds?**

A: You could, but ROC-AUC compresses that entire sweep into one number specifically designed to be threshold-independent, useful for comparing models before you've even decided what threshold you'll eventually deploy with. Once you have decided on an actual deployment threshold (based on your specific tolerance for false positives vs. false negatives), precision/recall/accuracy *at that threshold* become the relevant numbers again, ROC-AUC and single-threshold metrics answer different questions and are both useful at different stages.

**Q: Is grid search the only way to tune hyperparameters?**

A: No, the PCW explicitly allows grid search (try every combination), random search (try a random subset of combinations), or any other method. Grid search is simple and exhaustive but scales poorly (36 combinations here was fine; with more hyperparameters or finer ranges it becomes expensive fast). Random search often finds comparably good results with far fewer trials, and more advanced methods (Bayesian optimization, etc.) exist too, out of scope for this session but worth knowing the name of.

**Q: Does cross-validation replace the need for a validation set entirely?**

A: Cross-validation (this session's reading, covered more in the deep dive) is a way to get *more* out of a limited training set by rotating which subset acts as the validation fold across multiple training runs, instead of permanently sacrificing one fixed chunk of data to validation. It's a refinement of the same core idea (never tune on the test set), not a replacement for the train/val/test *principle*, the test set still gets held out separately and touched only once, regardless of whether you use a single validation set or k-fold cross-validation for the tuning step.

---

## What to Focus On for This Session

1. Be able to explain, precisely, why tuning on the test set biases the reported score, not just that "it's not allowed."
2. Practice deriving precision at a target recall directly from an ROC curve's FPR axis plus known class counts, this is a common, very practical skill.
3. Understand that ROC-AUC near 0.5 means "no ranking signal," not "systematically wrong," these are different failure modes with different fixes.

---

## Next Class Preview (Session 10)

**Topic**: Unit 1 Review & Assignment 1 Prep

This session's evaluation toolkit, train/val/test splitting, ROC-AUC, hyperparameter tuning without test-set leakage, is exactly the machinery Assignment 1 requires applying to a real personal dataset, tying together every model-fitting technique from Sessions 2 through 8 with this session's evaluation discipline.

---

**Study tip**: Before class, pick any two of your own hypothetical models (even made-up ones) and sketch what you'd expect their ROC curves to look like if one is "cautious" (high precision, lower recall) and one is "aggressive" (high recall, lower precision). Check whether your sketched curves would cross, and think through what that would mean for which model is "better."
