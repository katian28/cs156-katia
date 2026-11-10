# Session 18 Study Notes: Autoencoders

**Purpose**: autoencoders are the bridge between everything this course has already covered
(neural nets, backprop, PCA) and generative modeling. The readings all converge on one idea:
an autoencoder is PCA's nonlinear generalization, and the bottleneck is the mechanism that
forces compression instead of memorization. Read before class.

---

## Core Concept 1: The Bottleneck Forces Compression (Jordan)

An autoencoder is trained to reproduce its own input, which sounds trivial (just copy it).
What stops it from literally copying is the **bottleneck**: a hidden layer with fewer units
than the input. Since that layer can't hold a full copy of a higher-dimensional input, the
network is forced to learn weights that combine and discard redundant information, keeping
only what's needed to reconstruct well enough. This is "undercomplete": the bottleneck has
strictly fewer dimensions than the input, which is itself the only regularizer needed, there's
no explicit penalty term required, the architecture alone creates the constraint.

**This session's own numbers make it concrete**: 64-dimensional digit images squeezed through
a 2-dimensional bottleneck, reconstructed with mean per-example MSE of 0.055, recognizable but
blurry. The bottleneck is so aggressive (64 -> 2, a 32x reduction) that detail loss is
expected, and it's exactly what Question 3 asks you to characterize.

## Core Concept 2: Linear Autoencoder = PCA, Nonlinear = More Expressive (Hinton; Jordan; Keras blog)

The exact mechanical claim, repeated across all three readings: if an autoencoder's encoder
and decoder are **purely linear** (no activation function like ReLU/sigmoid at all), the
best-possible single-hidden-layer autoencoder learns the **same subspace as PCA**. Both are
solving "find the best lower-dimensional linear projection to minimize reconstruction error."

What changes once you add nonlinear activations (and/or more layers): PCA can only ever find
a flat subspace (a hyperplane) to project data onto. A nonlinear autoencoder can bend that
subspace into a curved manifold, and encode data onto the curve instead of the flat plane, a
strictly more flexible geometry. This is literally "nonlinear dimensionality reduction": same
goal as PCA (compress to fewer numbers, minimize reconstruction error), achieved with a richer
family of possible mappings.

**Why this session's architecture confirms it**: layers 1, 2, 4, 5 all use ReLU or sigmoid
activations, nonlinear by construction, exactly the mechanism the readings describe as what
makes this *more* than PCA. (Note: `layer_3`, the bottleneck-producing layer itself, has *no*
activation function applied in the code, it's a plain linear projection into the bottleneck,
only the surrounding layers are nonlinear.)

## Core Concept 3: Reconstruction Loss and Self-Supervision

Autoencoders are trained via **reconstruction loss**: compare the output to the original
input (this session uses binary cross-entropy, since pixel values are normalized to [0,1] and
treated like probabilities). No human labels are involved, the input *is* the target. The
Keras blog's framing is useful here: this is technically **self-supervised**, not unsupervised,
since there is still a target to match, it's just generated automatically from the data itself
rather than hand-labeled. This matters for Question 4: the digit-class clustering that shows
up in the bottleneck is a side effect of this self-supervised objective, the model never saw
`y_data`.

## Core Concept 4: What Autoencoders Are Actually Good For (Keras blog)

The Keras blog is blunt about this: autoencoders are a mediocre general-purpose compression
tool (hand-engineered codecs like JPEG usually beat them, and they only work well on data
similar to what they were trained on, "data-specific"). Their two genuinely useful roles: **
denoising** (train on corrupted inputs, clean targets, Jordan's reading covers this directly)
and **dimensionality reduction for visualization/downstream use**, exactly what Question 4's
2D bottleneck scatter plot demonstrates.

## Core Concept 5: Denoising Autoencoders (Jordan)

A denoising autoencoder deliberately feeds the encoder a *corrupted* version of the input
while still targeting the *clean* original as the reconstruction target. Because input and
target now differ, simply learning to copy the input is no longer an option, the network must
learn to push corrupted data back toward the manifold where clean data actually lives. This
works reliably near the training data's distribution and degrades further from it, the same
"can't extrapolate past what it learned" limitation that shows up everywhere else in ML.

---

## Common Questions and Confusions

**Q: If a linear autoencoder is "the same as" PCA, why would anyone use PCA at all?**

A: PCA has a closed-form solution (eigendecomposition, [[session_16_study_notes|Session 16's
own derivation]]), no gradient descent, no random initialization, no training instability.
A linear autoencoder reaches the same answer, slower and less reliably, via iterative
optimization. You'd only reach for the autoencoder once you actually need the nonlinear
version, PCA is strictly simpler whenever linearity is all you need.

**Q: Does a smaller bottleneck always mean worse reconstruction?**

A: In general, yes, fewer numbers to describe the data with means more information has to be
discarded. But Question 3's per-class breakdown shows this isn't a single global stat, which
digit reconstructs worse depends on the shape complexity of that specific class, not just raw
bottleneck size.

**Q: Why does clustering by digit emerge in the bottleneck if labels were never used in training?**

A: Because reconstruction quality depends on digit *shape*, and digit shape correlates
strongly with digit *label*. The network was never asked to separate classes, it just
discovered that similar-looking inputs need similar compressed codes to reconstruct well, and
similar-looking digits usually share a label. This is the standard argument for why
unsupervised/self-supervised representations can still be useful for downstream classification.

---

## What to Focus On for This Session

1. **The bottleneck as the only thing preventing trivial copying**: undercomplete architecture,
   no extra regularizer term needed, the dimension reduction itself does the work.
2. **Linear autoencoder = PCA, nonlinear = curved-manifold generalization**: the exact
   mechanical claim tested by this session's Q1 follow-up question, and the reason the model
   uses ReLU/sigmoid rather than leaving every layer linear.
3. **Self-supervision, and why label-free clustering in the bottleneck isn't magic**: it falls
   directly out of reconstruction quality correlating with visual similarity, which correlates
   with class label, a chain of correlations rather than the model "learning classes."
