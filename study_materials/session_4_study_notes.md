# Session 4 Study Notes: Decision Trees and Gini Impurity

**Purpose**: Deep explanation of how trees split data and why they behave the way they do. Read before class.

---

## Core Concept 1: What a Decision Tree Actually Does

A decision tree asks a sequence of yes/no questions about your features, one at a time, until it's confident enough to make a prediction.

Example: "Is sepal length <= 5?" -> yes -> "Is petal width <= 1?" -> no -> predict "versicolor"

Each question is a **split**: pick one feature, pick one threshold, divide the data into two groups (left = passes the test, right = fails it).

The tree is built top-down: start with all data at the root, pick the best split, then repeat on each resulting group (recursively) until you stop.

---

## Core Concept 2: Gini Impurity (How "Good" Is a Group?)

Before you can pick a split, you need a way to measure whether a group of data is "pure" (mostly one class) or "mixed" (multiple classes jumbled together).

**Formula**:
```
Gini(node) = 1 - sum(p_i^2)   for each class i in the node
```
where `p_i` is the fraction of the node's points belonging to class i.

**Worked example** (binary classification, classes 0 and 1):

- All same class: p_0=1.0, p_1=0.0 -> Gini = 1 - (1^2 + 0^2) = 1 - 1 = **0** (pure)
- Perfectly mixed: p_0=0.5, p_1=0.5 -> Gini = 1 - (0.25 + 0.25) = 1 - 0.5 = **0.5** (max impurity for binary)
- Mostly one class: p_0=0.9, p_1=0.1 -> Gini = 1 - (0.81 + 0.01) = 1 - 0.82 = **0.18** (fairly pure)

**Intuition**: Gini impurity is roughly "the probability you'd misclassify a random point in this node if you guessed its class randomly, in proportion to the node's own class mix." Lower is better.

---

## Core Concept 3: Weighted Gini Impurity (Evaluating a Split)

A single split produces two groups (left and right). You need one number that summarizes "how good was this split overall."

**Formula**:
```
Weighted Gini = (n_left / n_total) * Gini(left) + (n_right / n_total) * Gini(right)
```

This is a weighted average of the two child impurities, weighted by how many points fall into each side.

**Why weight by size?** A split that makes a tiny node perfectly pure but leaves a huge node just as mixed as before isn't actually a good split. Weighting by size prevents the algorithm from being fooled by small, easy-to-purify groups.

**Code from the notebook**:
```python
def leaf_gini_impurity(leaf_ys):
    if len(leaf_ys) == 0:
        return 0
    else:
        return 1 - (sum(leaf_ys==0)/len(leaf_ys))**2 - (sum(leaf_ys==1)/len(leaf_ys))**2

def weighted_gini_impurity(left_ys, right_ys):
    total = len(left_ys) + len(right_ys)
    left_leaf = leaf_gini_impurity(left_ys) * len(left_ys)/total
    right_leaf = leaf_gini_impurity(right_ys) * len(right_ys)/total
    return left_leaf + right_leaf
```

Trace through it: `leaf_gini_impurity` computes the formula above for one group. `weighted_gini_impurity` takes two groups (what you'd get from a candidate split), computes each one's impurity, and combines them weighted by size.

---

## Core Concept 4: How the Tree Picks a Split (Brute Force Search)

For a real-valued feature, you can't just ask "yes/no" the way you would for a binary feature. Instead, treat **every unique value in the data** as a candidate threshold, and test all of them.

**Algorithm**:
```
for each feature (axis):
    for each unique value of that feature (threshold):
        split data into left (<=threshold) and right (>threshold)
        compute weighted Gini impurity of this split
        keep track of the (axis, threshold) with the lowest impurity
```

**Why this works**: since Gini impurity is only affected by which points end up on which side, you never need to test a threshold between two identical values, and you never need a threshold below the min or above the max. Testing every unique observed value covers every meaningfully different split.

**Code**:
```python
def gini_impurity(x_data, y_data, axis, threshold):
    left_mask = x_data[:, axis] <= threshold
    right_mask = x_data[:, axis] > threshold
    return weighted_gini_impurity(y_data[left_mask], y_data[right_mask])
```

This is the function you call repeatedly, once per candidate (axis, threshold) pair, to brute-force search for the best split.

---

## Core Concept 5: Recursion (Building Depth Beyond the Root)

Once you've picked the best split for the root, you don't stop. You take the left group and the right group, and **run the exact same search independently on each one**, as if each were its own tiny dataset.

**In the Session 4 notebook**: the root split was Y <= 0.147. Then:
- Left group (Y <= 0.147, 210 points): best split turned out to be X <= -0.78
- Right group (Y > 0.147, 290 points): best split turned out to be X <= 1.24

Notice the two child splits used a **different feature** (X) than the root did (Y). That's expected. Each node's best split only depends on the data inside that node; there's no rule that later splits must reuse the same feature.

This recursion is what builds tree *depth*. Depth 1 = root only. Depth 2 = root + two children. Each additional level repeats the same brute-force search, just on a smaller and smaller subset of data.

---

## Core Concept 6: Why Trees Are Unstable (Perturbation Sensitivity)

The splitting algorithm is **greedy**: at every node, it picks whichever single split is best *for that node right now*. It never asks "would a slightly worse split here lead to a better tree overall two levels down?"

**Why this causes instability**:
1. If two candidate splits have Gini scores that are very close (a near-tie), a tiny change in the data, one point added, removed, or moved, can flip which split wins.
2. Once the chosen split changes, the two child groups are now made of different points than before.
3. Every split below that point is recomputed on different data, so it can also change.
4. The result: a small change at the top can produce a completely different-looking tree, even though the overall training accuracy might barely change.

**Contrast with linear models**: In linear regression, moving one data point slightly nudges the coefficients slightly (continuous, smooth change). In a tree, moving one data point can flip a discrete decision (which side of a threshold a whole batch of points falls on), which is an all-or-nothing change. That discreteness is the root cause of the instability.

---

## Core Concept 7: Why Perfectly Fitting Training Data Is Bad

If you let a tree keep splitting until every leaf is 100% pure (Gini = 0 everywhere), it has essentially memorized the training set: every training point ends up in its own tiny, perfectly labeled region.

**Why this fails to generalize**:
- Real data has noise. Some points are mislabeled, some are outliers, some just don't fit the "true" pattern.
- A tree that perfectly separates every training point has fit that noise as if it were signal.
- On new data, the noise won't repeat in the same way, so the ultra-specific splits the tree learned won't transfer.

**Bias-variance framing**: An unconstrained tree is **low bias** (fits training data almost perfectly, nearly 0 error) but **high variance** (small changes in training data produce very different trees, and predictions swing accordingly). High variance models tend to perform worse on unseen data than their training accuracy would suggest.

**The fix (previewed, covered in later sessions)**: limit tree depth, require a minimum number of samples per leaf, or use ensembles (like Random Forests or XGBoost) that average many unstable trees together to cancel out the instability.

---

## Core Concept 8: Why Splits Are Always Axis-Aligned

Every single split in a decision tree looks at **exactly one feature** and picks **one threshold** on that feature. It never combines two features into a single diagonal cut (like "if 2*age + 3*income > 100").

**Geometric consequence in 2D**:
- A split on feature X is always a vertical line.
- A split on feature Y is always a horizontal line.
- No split is ever diagonal or curved.

**What this means for curved data** (like the two-moons dataset used in the PCW): the true boundary between the two classes is a smooth curve. A tree can never draw that curve directly. Instead, it approximates the curve using a **staircase** of many small horizontal and vertical cuts, each one carving off a rectangular corner that's mostly the "wrong" class.

**This connects directly to instability and overfitting**: to get a *good* staircase approximation of a smooth curve, you need many splits (a deep tree). But a deep tree is exactly the kind of tree that overfits and becomes unstable (Concepts 6 and 7). This is the fundamental tension in decision trees: shallow trees underfit curved boundaries, deep trees overfit and become unstable.

---

## Worked Example Recap (From the Notebook)

Using `make_moons(n_samples=500, noise=0.15, random_state=42)`:

```
Unsplit impurity: 0.500 (classes are balanced 50/50)

Root split:  axis=Y, threshold=0.147  -> weighted Gini = 0.263
  Left  (Y <= 0.147, 210 pts): impurity 0.172 before further split
    Best split: axis=X, threshold=-0.780 -> weighted Gini = 0.073
  Right (Y > 0.147, 290 pts): impurity 0.328 before further split
    Best split: axis=X, threshold=1.238  -> weighted Gini = 0.183
```

Read this top to bottom: the tree first cuts the data top/bottom (Y axis), because that single cut reduces impurity the most across the *whole* dataset. Once that's done, each half still has some remaining mixing, but now the mixing runs left/right, so the next best cut in each half is on the X axis. This is the staircase pattern from Concept 8 forming in real time.

---

## Common Questions and Confusions

**Q: Why use Gini impurity instead of just counting misclassifications?**

A: A raw error count only cares about the majority class in each node and ignores how confident that majority is. A node that's 51% class A and one that's 99% class A can have the same "majority class," but the second one is far purer. Gini impurity distinguishes between them because it uses the full class *proportions*, not just which class wins. This gives the search a smooth signal to compare many candidate splits, instead of a flat error count that ties too often.

**Q: Where does the formula `1 - sum(p_i^2)` actually come from?**

A: It's the probability of a specific kind of mistake: pick two points at random from the node (with replacement) and guess that the second one has the same class as the first. `sum(p_i^2)` is the probability you'd guess right by chance (both points happen to be the same class). `1 - sum(p_i^2)` is therefore the probability you'd be wrong. A pure node (one class) can never produce this kind of mistake, so Gini = 0. A 50/50 node produces this mistake half the time, so Gini = 0.5.

**Q: What's the difference between Gini impurity and entropy? Don't they do the same thing?**

A: Yes, both measure how mixed a node is, and in practice they pick very similar splits almost all the time. Entropy is `-sum(p_i * log2(p_i))`, which comes from information theory (bits needed to describe the class). Gini is cheaper to compute (no logarithm) and is scikit-learn's default for this reason. You won't usually see a meaningfully different tree from switching between them.

**Q: Why `<= threshold` for left and `> threshold` for right? Does the direction matter?**

A: It's a convention, not a mathematical requirement. As long as every point ends up in exactly one side (no overlap, no gaps), the Gini calculation comes out the same regardless of which side you call "left" and which you call "right." Using `<=` and `>` (rather than `<` and `>=`) just guarantees a point exactly equal to the threshold has a defined home.

**Q: What if two candidate splits have the exact same Gini score?**

A: The tie has to be broken somehow, typically by whichever candidate was found first in the search order (feature order, then threshold order). This is precisely the scenario from Concept 6 (instability): near-ties are common, and which one wins can flip with tiny data changes, even though both were equally good by the Gini metric.

**Q: Does testing "every unique value" mean the tree tests infinitely many thresholds for continuous data?**

A: No. Even though the feature is continuous (any real number is possible in theory), your *dataset* only contains a finite number of distinct observed values. You only ever need to test thresholds at (or between) those observed values, because moving the threshold between two adjacent observed values without crossing either one changes nothing about which points fall left vs. right. So the search is finite: one candidate threshold per unique value per feature.

**Q: Why did the root split pick the Y axis, but both children picked the X axis? Shouldn't it be consistent?**

A: No, and this trips people up. Each node's split is chosen completely independently, based only on the data that has already been filtered down to that node. The root sees the full moon shape, where the up/down separation happens to be the strongest single signal. Once you've already cut top/bottom, the *remaining* mixing inside each half happens to run left/right instead. There's no rule requiring successive splits to use different (or the same) features, it's whatever the data says is best at that node.

**Q: If a feature is categorical instead of continuous (e.g., color: red/blue/green), how does splitting work?**

A: Not covered directly in this PCW's code (which assumes real-valued features), but conceptually: instead of a threshold, the algorithm considers subsets of categories to send left vs. right (e.g., "red" vs. "blue or green"). The same weighted Gini impurity formula is used to score each candidate grouping.

**Q: Is Gini = 0.5 always the worst possible impurity value?**

A: Only for binary classification with balanced classes. For k classes, the maximum possible Gini is `1 - 1/k`, reached when all classes are equally represented. With 3 balanced classes, max Gini is `1 - 1/3 = 0.667`, not 0.5. The 0.5 ceiling you see in this PCW is specific to two classes.

---

## Common Pitfalls

1. **Confusing Gini impurity with accuracy.** Gini impurity is a measure of node purity, not a direct accuracy score. A Gini of 0.05 doesn't mean "95% accurate," it means the node is nearly pure.

2. **Forgetting to weight by node size.** Averaging two child Gini scores without weighting by how many points are in each child gives misleading results (a tiny pure node can look artificially good).

3. **Thinking a split must use the same feature as its parent.** It doesn't. Each node's best split is found independently, based only on the data inside that node.

4. **Assuming trees can draw diagonal boundaries.** They can't, ever. Only axis-aligned rectangles, no matter how deep the tree gets.

5. **Assuming a tree that fits training data perfectly is a good tree.** It's usually a red flag for overfitting, not a sign of success.

---

## What to Focus On for the PCW

1. **Q1**: Connect greediness (no lookahead) to both instability (1a) and the fact that perfect training fit is bad (1b). These aren't two separate facts, they're the same underlying limitation showing up in two ways.

2. **Q2-Q3**: Practice predicting a split visually first, then verifying it numerically with the brute-force Gini search. Notice how close your visual guess was.

3. **Q4**: Pay attention to *which axis* gets picked at each level. If it's the same axis as the parent, ask why; if it's different, that's the expected staircase pattern.

---

## Next Class Preview (Session 5)

**Topic**: Max Likelihood 1 - Naive Bayes

Moves away from partition-based classification (trees) toward probability-based classification (Bayes). Connects back to the Bayesian inference extension problems from Session 1.

---

**Study tip**: Before class, try hand-computing the Gini impurity for a small made-up dataset (5-6 points, 2 classes) by hand, without code. If the numbers match what the code would give you, you understand the formula, not just the code.
