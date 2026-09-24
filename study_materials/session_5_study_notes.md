# Session 5 Study Notes: Naive Bayes, Smoothing, and Gaussian Naive Bayes

**Purpose**: Deep explanation covering the three assigned readings (Starmer's two StatQuest videos + VanderPlas's chapter), beyond what the PCW notebook itself covers. Read before class.

---

## Core Concept 1: The Naive Bayes Decision Rule (Recap)

For a new example with features `x_1, x_2, ..., x_n`, Naive Bayes picks whichever class maximizes:

```
P(class) * P(x_1 | class) * P(x_2 | class) * ... * P(x_n | class)
```

`P(class)` is the **prior**: how common that class is before looking at any features. Each `P(x_i | class)` is a **likelihood**: how probable that specific feature value is, given the class. The "naive" part is treating all the likelihood terms as independent and just multiplying them.

Everything below is about **how you estimate those `P(x_i | class)` terms**, which changes depending on whether your features are discrete counts (like word frequencies) or continuous numbers (like height or temperature).

---

## Core Concept 2: Multi-Class Naive Bayes (Starmer, "Naive Bayes, Clearly Explained!!!")

The SMS PCW only had 2 classes (spam/ham), so the decision was a simple comparison: is `P(spam | words)` bigger or smaller than `P(ham | words)`?

With more than 2 classes, the rule generalizes the exact same way, just compare more scores and take the max:

```
predicted_class = argmax over all classes c of:  P(c) * product over all features of P(x_i | c)
```

**Worked example** (3 classes instead of 2): imagine classifying a short text as "sports," "politics," or "weather" based on which words it contains. For a message with words "game" and "score":

```
score(sports)   = P(sports)   * P(game | sports)   * P(score | sports)
score(politics) = P(politics) * P(game | politics) * P(score | politics)
score(weather)  = P(weather)  * P(game | weather)  * P(score | weather)
```

You compute all three scores and predict whichever is highest. Nothing about the underlying math changes, the only difference from binary classification is you're comparing 3+ numbers instead of 2.

---

## Core Concept 3: Why You Need Smoothing (Starmer's Focus, and VanderPlas)

**The problem**: `P(word | class)` is normally estimated as:

```
P(word | class) = (count of that word in that class) / (total word count in that class)
```

But what happens if a specific word (say, "bitcoin") **never once appeared** in any ham message during training, but a new incoming message you're trying to classify contains it?

```
P("bitcoin" | ham) = 0 / (total ham words) = 0
```

Now look at the full product for ham:
```
score(ham) = P(ham) * P(word_1 | ham) * ... * P("bitcoin" | ham) * ...
           = P(ham) * (some numbers) * 0 * ...
           = 0
```

**A single zero in the product destroys the entire score**, no matter how strongly every other word in the message pointed toward ham. The model can never predict ham for any message containing "bitcoin," permanently, even if every other feature says otherwise. This is clearly too rigid: one never-before-seen word shouldn't get infinite veto power.

**The fix: Laplace (additive) smoothing.** Add a small constant to every count before dividing:

```
P(word | class) = (count of word in class + alpha) / (total words in class + alpha * vocabulary_size)
```

`alpha` is usually 1 (this specific case is called "Laplace smoothing" or "add-one smoothing"). Adding `alpha` to the numerator means no word can ever have exactly zero probability, it gets a small, non-zero floor value instead. Adding `alpha * vocabulary_size` to the denominator keeps all the probabilities for a class properly summing to 1.

**This is exactly the floor value you saw in your own notebook.** Remember the "Bottom 10 tokens by P(token | spam)" from the SMS PCW, words like "nbme," "jstfrnd," "necesity"? They all landed on the *exact same* tiny probability (0.000084). That's not a coincidence, that's the smoothing floor. Those words appeared zero or almost zero times in spam messages in training, so smoothing assigned them all the same minimum non-zero value instead of a hard zero.

**Concrete numeric example**: suppose your spam vocabulary has 8672 words total, and the word "bitcoin" appears 0 times among 50,000 total spam words.
```
Without smoothing: P(bitcoin | spam) = 0 / 50000 = 0
With smoothing (alpha=1): P(bitcoin | spam) = (0 + 1) / (50000 + 1*8672) = 1 / 58672 ≈ 0.000017
```
Small, but not zero. A message with "bitcoin" can still be classified as spam if enough *other* words in it strongly suggest spam, the one unseen word no longer nukes the entire prediction.

---

## Core Concept 4: Gaussian Naive Bayes (Starmer's second video, and VanderPlas)

Everything above (including the SMS PCW) assumed **discrete, countable features**: how many times does a word appear? That's what `MultinomialNB` is built for.

But what if a feature is a continuous number, like a person's height, a temperature reading, or (thinking back to Session 1) an iris flower's petal length? You can't sensibly ask "how many times did height=5.7 occur," continuous values essentially never repeat exactly.

**Gaussian Naive Bayes's fix**: instead of counting occurrences, assume each feature follows a **normal (bell curve) distribution within each class**, and estimate that distribution's mean and variance from the training data of that class.

```
P(x_i | class) = (1 / sqrt(2*pi*variance_class)) * exp( -(x_i - mean_class)^2 / (2*variance_class) )
```

This is just the normal distribution's probability density function, evaluated at the observed feature value `x_i`, using the mean and variance computed **separately for each class**.

**Worked example** using Iris data (from Session 1/2, a dataset you already know well): suppose you're classifying species using petal length.

```
Setosa petal lengths in training data:     mean = 1.5 cm, variance = 0.03
Versicolor petal lengths in training data: mean = 4.3 cm, variance = 0.22
```

For a new flower with petal length = 1.4 cm, Gaussian Naive Bayes computes:
```
P(1.4 | setosa)     = tall, narrow bell curve centered at 1.5 -> HIGH probability, 1.4 is very close to setosa's typical value
P(1.4 | versicolor) = bell curve centered at 4.3 -> extremely LOW probability, 1.4 is far from versicolor's typical range
```
So this flower gets classified as setosa, driven by how close its petal length sits to each class's own bell curve center, weighted by how tight or spread out that class's bell curve is.

**Key difference from smoothing (Concept 3)**: Gaussian Naive Bayes doesn't need Laplace smoothing the same way Multinomial does, since it's not counting discrete occurrences that can hit exactly zero. Instead, its own failure mode is different: if a class's variance is estimated as extremely close to zero (all training examples for that class had nearly identical feature values), the bell curve becomes razor-thin, and any new example even slightly outside that tiny range gets an extremely low, sometimes near-zero, probability. Some implementations add a small constant to the variance for the same reason Laplace smoothing exists (avoiding division by, or multiplication by, values so close to zero they behave like zero).

---

## Core Concept 5: Choosing the Right Naive Bayes Variant (VanderPlas)

Three common variants, matched to feature type:

| Variant | Feature type | Example use case |
|---|---|---|
| `GaussianNB` | Continuous numbers | Iris measurements, height, temperature |
| `MultinomialNB` | Discrete counts | Word counts / TF-IDF in text (what the SMS PCW used) |
| `BernoulliNB` | Binary (present/absent) | "Does this message contain the word 'free'?" yes/no, ignoring how many times |

**Why this matters practically**: picking the wrong variant for your feature type gives systematically wrong probability estimates. Using `GaussianNB` on word-count data would assume word counts follow a bell curve (they don't, they're skewed, mostly zero, occasionally large), producing poorly calibrated probabilities. Using `MultinomialNB` on continuous measurements like petal length doesn't even make sense mathematically, the "count" interpretation breaks down for continuous numbers.

---

## Common Questions and Confusions

**Q: Is smoothing only needed for words that never appear at all (count = 0), or does it change every probability?**

A: It changes **every** probability, not just the zero ones, but the effect is largest for rare/unseen words. For a word that appears constantly (say, 5000 times), adding `alpha=1` to the numerator barely moves the ratio at all. For a word that appears 0 or 1 times, that same `+1` has a huge relative effect. Smoothing is a general correction, its practical impact is just concentrated on the rare cases.

**Q: Why is it called "Laplace" smoothing specifically?**

A: Named after Pierre-Simon Laplace, who proposed the same `+1` adjustment for a different, older problem (estimating the probability the sun will rise tomorrow, having observed it rise every day so far, without assigning that a naive/wrong probability of exactly 1.0). It's the same mathematical idea: don't let a limited sample assign impossible (0 or 1) certainty to an event.

**Q: Does Gaussian Naive Bayes still assume features are independent, same as Multinomial?**

A: Yes, the "naive" independence assumption is identical across all variants. The only thing that changes between `GaussianNB`, `MultinomialNB`, and `BernoulliNB` is *how each individual* `P(x_i | class)` *term is calculated* (bell curve vs. word-count ratio vs. presence/absence ratio). The core naive multiplication-of-independent-terms structure from Concept 1 stays the same.

**Q: In the SMS PCW, could we have used GaussianNB instead of MultinomialNB?**

A: Technically you could run the code, but it would be a poor modeling choice. TF-IDF values are heavily skewed (mostly zero for any given message, since most messages only use a tiny fraction of the vocabulary), not remotely bell-shaped. `MultinomialNB` (or `BernoulliNB`) matches that data shape far better, which is exactly VanderPlas's point about matching the variant to the feature type.

**Q: What happens if alpha (the smoothing constant) is set very large instead of 1?**

A: Larger alpha pulls every probability estimate further toward a flat, uniform distribution (every word treated as roughly equally likely), drowning out the actual signal in your training counts. Alpha is a tuning knob: too small (close to 0) barely helps rare words, too large flattens out genuinely useful frequency differences. Alpha=1 (classic Laplace smoothing) is a reasonable default, but it can be tuned like any other hyperparameter.

---

## What to Focus On for This Session

1. Understand *why* a single zero-probability feature can break the entire product, this was the actual root cause you saw in your own bottom-30-tokens plot from the SMS PCW.
2. Be able to state the smoothing formula and explain what each term does (`+alpha` in numerator, `+alpha*vocab_size` in denominator).
3. Understand that Gaussian Naive Bayes is the *same core algorithm* as Multinomial, just with a different formula for estimating each `P(x_i | class)` term, driven by the feature being continuous instead of a count.

---

## Next Class Preview (Session 6)

**Topic**: Max Likelihood 2 - Parameter Estimation

Naive Bayes's `P(word | class)` estimates (and Gaussian NB's mean/variance estimates) are themselves examples of **maximum likelihood estimation**: picking the parameter values that make the observed training data most probable. Session 6 formalizes this idea and generalizes it beyond Naive Bayes.

---

**Study tip**: Before class, try computing Gaussian Naive Bayes probabilities by hand for 2-3 made-up Iris flowers, using rough means/variances you estimate by eye from the Session 1/2 scatter plots. If your hand-estimates roughly match what `sklearn.naive_bayes.GaussianNB` would output, you understand the mechanics, not just the formula.
