# Session 6 Study Notes: Maximum Likelihood Estimation for Logistic Regression

**Purpose**: Deep explanation of probability vs. likelihood, MLE, and how it applies to logistic regression. Covers the five assigned readings. Read before class.

---

## Core Concept 1: Probability Is Not Likelihood

These two words get used interchangeably in casual speech, but in statistics they mean opposite things, depending on which side of a conditional probability you're allowed to change.

**Probability**: `P(data | parameters)`. Parameters are **fixed** (known), data **varies**. Question: "given this coin has probability 0.5 of heads, how likely is it I see 3 heads in a row?"

**Likelihood**: `L(parameters | data)`. Data is **fixed** (already observed), parameters **vary**. Question: "I saw 3 heads in a row, how likely is it that the coin's true probability of heads is 0.5? What about 0.9? What about 0.1?"

**Same formula, different variable held fixed.** For a coin with heads-probability `p`, seeing heads-heads-heads:
```
As a probability (p fixed, data varies): P(HHH | p=0.5) = 0.5^3 = 0.125
As a likelihood (data fixed, p varies):  L(p | HHH) = p^3, evaluate at whatever p you want to test
```

**Why this distinction matters for the whole course**: every time we "train a model," we are not computing a probability, we're computing a likelihood, then searching over possible parameter values to find whichever one makes that likelihood largest. That search is MLE.

---

## Core Concept 2: The Maximum Likelihood Principle

Given a likelihood function `L(parameters | data)`, maximum likelihood estimation picks the parameter values that make the observed data as probable as possible:

```
best_parameters = argmax over all possible parameter values of L(parameters | data)
```

In practice we maximize the **log**-likelihood instead of the raw likelihood, for two purely practical reasons:

1. **Numerical stability**: multiplying together hundreds or thousands of probabilities (each less than 1) underflows to 0 in floating point almost immediately. Adding logs of those same probabilities does not.
2. **Turns a product into a sum**: sums are far easier to differentiate than products (no need for the product rule across every term), which matters once you start using gradients (Concept 5) to search for the maximum.

Since `log` is a strictly increasing function, whatever parameters maximize the likelihood also maximize the log-likelihood, so nothing is lost by working with logs.

---

## Core Concept 3: Logistic Regression as a Sequence of Weighted Coin Flips

Ordinary linear regression predicts a number directly. Logistic regression instead predicts a **probability**, and then treats each observation as a single coin flip with that probability of landing "True":

```
P(True | x)  = sigma(beta_1*x + beta_0)
P(False | x) = 1 - sigma(beta_1*x + beta_0)
```

The key move for building a likelihood function: instead of one fixed coin bias `p` (like a real coin), **every observation gets its own bias**, computed from that observation's `x` value. A person who trained 2000 hours has a different implied "coin bias" for landing a role than a person who trained 200 hours, but it's the exact same underlying formula, `sigma(beta_1*x + beta_0)`, just evaluated at a different `x`.

**Worked mini-example** with 3 people and made-up parameters `beta_0 = -2, beta_1 = 1`:

```
Person A: x=1 (1000 hrs), got the role (True)
Person B: x=3 (3000 hrs), got the role (True)
Person C: x=0.5 (500 hrs), did not get the role (False)

sigma(z) = e^z / (1 + e^z)

P(True | Person A)  = sigma(1*1 + -2)  = sigma(-1) ≈ 0.269
P(True | Person B)  = sigma(1*3 + -2)  = sigma(1)  ≈ 0.731
P(False | Person C) = 1 - sigma(1*0.5 + -2) = 1 - sigma(-1.5) ≈ 1 - 0.182 = 0.818

L(beta_0=-2, beta_1=1) = 0.269 * 0.731 * 0.818 ≈ 0.161
ln L(beta_0=-2, beta_1=1) = ln(0.269) + ln(0.731) + ln(0.818) ≈ -1.313 + -0.313 + -0.201 ≈ -1.827
```

Try a different, worse guess (`beta_0 = 2, beta_1 = -1`, roughly backwards) and you'd get a much lower (more negative) log-likelihood, since this guess assigns low probability to A and B actually succeeding and high probability to C succeeding, the opposite of what happened. That's the whole game: try different `(beta_0, beta_1)` pairs, keep whichever gives the highest log-likelihood.

---

## Core Concept 4: MLE Under Model Misspecification (Shafkat reading)

The Shafkat reading's core point (in the "Suppose we receive... variance in the samples" section): MLE doesn't require your model to be able to perfectly represent the true data-generating process. It just finds the best parameters *within whatever model family you chose*.

**Concrete framing for our PCW**: our synthetic data was generated *exactly* from a logistic curve (`sigma(3x - 5)` plus coin-flip randomness), so logistic regression is the perfectly correct model family here, MLE can, in principle, recover the true parameters given enough data. But in most real problems, the true underlying process is more complex than any model you fit (a straight-line logistic boundary is rarely the literal truth in messy real-world data). MLE still runs the same way: it finds whichever parameters, within your chosen model family, make the observed data most probable, even if no setting of those parameters could ever make the data *perfectly* probable. The "best fit within a constrained family" framing is what "constrained case" refers to.

**Why this matters**: a high log-likelihood doesn't prove your model family is correct, it only proves you found the best-fitting member of that family. A poorly-chosen model family (say, a straight line for genuinely curved data) will still get an MLE fit, it'll just be the "least bad straight line," not a good fit in any absolute sense.

---

## Core Concept 5: Gradients Point Locally, Not Globally

Once you have a log-likelihood function, MLE becomes an optimization problem: find the parameters that maximize it. One standard tool is the **gradient**, computed automatically here with JAX's `jax.grad`.

**What a gradient actually tells you**: at your *current* parameter guess, which direction increases the log-likelihood fastest, and by roughly how much, per unit step. It says nothing directly about where the true global maximum is.

**This showed up concretely in the PCW**: Aishwarya's guess was `beta_0=3, beta_1=-3`. The true `beta_0` is `-5`, far below 3. Yet the gradient at Aishwarya's point had a *positive* component for `beta_0` (`+95.09`), meaning the locally best move is to *increase* `beta_0`, away from the true value, not toward it.

**Why this isn't a contradiction**: think of the log-likelihood as a hilly landscape (or here, since we're maximizing, a landscape with peaks). Aishwarya's starting guess has entirely the wrong sign pattern (positive intercept, negative slope, versus true negative intercept, positive slope). The local slope of the landscape right where she's standing doesn't have to point in the direction of the distant true peak, it only has to point uphill from exactly where she is. If you took that gradient step, then recomputed the gradient at your new position, and repeated many times (gradient ascent), the path would eventually curve around and climb toward the true peak, just not necessarily in a straight line, and not in a single step.

**Practical takeaway**: a gradient computation is one step of an iterative search, not a one-shot answer. This is the same idea underlying the from-scratch logistic regression you built in Session 2, gradient descent (or ascent) takes many small, locally-correct steps that cumulatively converge, even when any single step looks like it's moving in a strange direction relative to the final destination.

---

## Common Questions and Confusions

**Q: If probability and likelihood use the same formula, why do we need two different words at all?**

A: Because the *question you're allowed to ask* is different, and mixing them up leads to real statistical errors. "Given these parameters, what's the probability of this data" is a well-defined single number that sums to 1 across all possible data outcomes. "Given this data, what's the likelihood of these parameters" is NOT required to sum to 1 across all possible parameter values, likelihood is a comparison tool between parameter settings, not a probability distribution over parameters. Treating a likelihood as if it were a probability (without further steps, like Bayesian priors) is a common and serious mistake.

**Q: Why did we use `if y:` instead of `if y == True:` in the bug fix, aren't those the same?**

A: They're equivalent in behavior for this case (both correctly handle `numpy.bool_`), but `if y:` is the more idiomatic and slightly more general fix, since it also correctly handles a raw `0`/`1` integer label, not just boolean-like values. The broken original was `if y is True:`, which uses Python's identity operator `is`, checking whether `y` is literally the exact same object as the singleton `True`. `numpy.bool_(True)` is a *different object* than Python's `True`, so the identity check silently fails even though the value is logically true. `==` and plain truthiness both check *value* equality, not object identity, so both correctly fix the bug. Rule of thumb: never use `is` for value comparisons, only for singleton checks like `is None`.

**Q: Does maximum likelihood always have a unique best answer?**

A: For logistic regression with enough, well-behaved data (which our synthetic dataset satisfies), yes, the log-likelihood surface is concave (one single peak, no false peaks to get stuck in), so gradient ascent reliably finds the global maximum. This is not true for every model family (neural networks, for instance, generally have many local peaks), but it is true for logistic regression specifically, which is part of why it's such a reliable, well-behaved model to fit.

**Q: In the coin-flip analogy, why does the order of heads/tails not matter to the final likelihood value?**

A: Because multiplication is commutative, `p * (1-p) * p * p * (1-p)` gives the same product regardless of what order you multiply the terms in. The likelihood only depends on *how many* heads and tails you saw in total, not the sequence they appeared in. The same is true for logistic regression's likelihood: summing (in log space) the per-observation log-probabilities gives the same total no matter what order you loop through the dataset in.

---

## What to Focus On for This Session

1. Be able to state, in your own words, the difference between `P(data | params)` and `L(params | data)`, and why the distinction matters when we say we "fit" a model.
2. Be able to write out the logistic regression log-likelihood formula from the coin-flip analogy, without looking it up, filling in `sigma(beta_1*x + beta_0)` for the per-observation "coin bias."
3. Understand why `if y is True:` fails for `numpy.bool_`, and more generally, why `is` should never be used for value comparison. This is a subtle but common real-world Python bug, worth internalizing beyond just this PCW.
4. Be comfortable with the idea that a gradient only tells you the *local* best direction, not the destination, this will come up again and again (neural networks, later sessions) as the core mechanism behind essentially all model training.

---

## Next Class Preview

**Topic**: Unit 1 Review & Assignment 1 Prep

This session's MLE machinery ties together everything from Unit 1: Session 2's from-scratch logistic regression (gradient descent to maximize log-likelihood, though we called it "minimize log-loss" there, same thing with a sign flip), Session 3's normal equation (also an MLE result, for linear regression under Gaussian-noise assumptions, per Murphy 4.2.6.1/4.2.7), and Session 5's Naive Bayes (also fit via maximum likelihood, just using simple counting instead of gradient search, since its per-word probabilities have a closed-form MLE solution).

---

**Study tip**: Before class, try hand-deriving why `ln L(p) = ln(p) + ln(1-p) + ln(p) + ln(p) + ln(1-p)` for the 5-flip coin example (HTHHT), without looking at the PCW's given answer. Then try writing the general logistic regression version yourself, one term per observation, before checking it against Question 2's answer in the notebook.
