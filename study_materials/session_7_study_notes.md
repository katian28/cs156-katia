# Session 7 Study Notes: Feed-Forward Neural Networks

**Purpose**: Deep explanation of layers, units, activation functions, and the matrix view of a forward pass. Read before class.

---

## Core Concept 1: What a Neural Network Actually Computes

Strip away the biological metaphor ("neurons," "brain-inspired") and a feed-forward neural network is just a chain of the exact same operation, repeated:

```
output = activation(input @ W + b)
```

- `input` is a vector (a row of numbers)
- `W` is a matrix of learned weights
- `b` is a vector of learned biases
- `activation` is a simple, fixed, non-linear function applied to every number in the result

Chain several of these together, feeding each one's output as the next one's input, and you have a feed-forward neural network. "Feed-forward" just means the information only flows one direction, no loops, no feedback into earlier layers.

---

## Core Concept 2: Layer vs. Unit, Precisely

**A unit** (also called a neuron or node): one single computation. Takes every value from the previous layer, multiplies each by its own personal weight, sums them all up, adds one bias, then applies an activation function. Produces exactly one output number.

**A layer**: a group of units that all see the exact same set of inputs (whatever the previous layer output), but each unit in that group has its own independent set of weights, so different units in the same layer can learn to detect different things from the same input.

**Concrete sizes from the PCW's Keras network**:
```
Flatten(784)  -> Dense(128, sigmoid)  -> Dense(64, sigmoid)  -> Dense(10, softmax)
```
- The first hidden layer has 128 units. Each of those 128 units independently looks at all 784 input pixels, with its own 784 weights + 1 bias, and produces 1 number. 128 units together produce 128 numbers, becomes the next layer's input.
- The perceptron (Sonar PCW) is the simplest possible case: exactly 1 layer, containing exactly 1 unit.

---

## Core Concept 3: Why Activation Functions Are Non-Linear

If every layer just did `x @ W + b` with no activation function afterward, stacking multiple layers would collapse mathematically into a single big matrix multiplication (since a linear function of a linear function is still linear). No amount of depth would let the network learn anything a single layer couldn't already learn.

Non-linear activation functions (sigmoid, softmax, and later ReLU) are what let stacking layers actually add power: each layer can bend, fold, or squash the space in a way the next layer's linear step can further reshape, letting the whole network approximate much more complicated, curved functions than any single linear layer could.

**Sigmoid** (`e^x / (1+e^x)`): squashes any real number into (0, 1). Used in both hidden layers of the PCW's Keras network.

**Softmax**: generalizes sigmoid to multiple classes at once. Takes a vector of raw scores and turns them into a probability distribution (all values between 0 and 1, summing to exactly 1). This is what should have been on the Keras network's final layer (10 digit classes), the PCW's given code left it off, which was one of the two real bugs.

**Step function** (used in the perceptron): `1 if activation >= 0 else 0`. The oldest and simplest activation, a hard threshold instead of a smooth curve. Its main practical drawback: it's not differentiable at the threshold, so you can't use gradient-based learning the way you can with sigmoid or softmax, the classic perceptron learning rule (used in the PCW's `train_weights`) is a slightly different, older update rule that predates modern gradient descent, though it looks superficially similar (`error * learning_rate`).

---

## Core Concept 4: The Forward Pass as Matrix Multiplication, Worked Example

Take a tiny made-up network: 2 inputs, 1 hidden layer with 2 units (sigmoid), 1 output unit (sigmoid).

```
x = [1.0, 0.5]                       shape (1, 2)

W1 = [[0.2, -0.4],
      [0.1,  0.3]]                   shape (2, 2)
b1 = [0.0, 0.1]                      shape (1, 2)

z1 = x @ W1 + b1
   = [1.0*0.2 + 0.5*0.1 + 0.0,  1.0*(-0.4) + 0.5*0.3 + 0.1]
   = [0.25, -0.15]

h1 = sigmoid(z1) = [0.562, 0.463]    shape (1, 2), the hidden layer's output

W2 = [[0.5],
      [-0.2]]                       shape (2, 1)
b2 = [0.05]                          shape (1, 1)

z2 = h1 @ W2 + b2
   = [0.562*0.5 + 0.463*(-0.2) + 0.05]
   = [0.281 - 0.0926 + 0.05]
   = [0.2384]

y = sigmoid(z2) = [0.559]            final output, a probability between 0 and 1
```

Every layer in the real Keras network (128 units, then 64 units, then 10 units) does exactly this same sequence of steps, just with bigger matrices. This is worth tracing through by hand once with tiny numbers like above, since the real network's matrices are too large to compute by hand but follow the identical pattern.

---

## Core Concept 5: The Perceptron Is the Same Pattern, One Layer Deep

The Sonar perceptron looks different from the Keras network at first glance (no Keras, a plain Python for-loop, a step function instead of sigmoid), but it's the exact same underlying computation with `L=1` layer and `1` unit:

```
activation = x @ w + b               (a dot product, since there's only 1 output unit)
output = 1.0 if activation >= 0.0 else 0.0
```

**Why the update rule looks familiar**: `train_weights()` uses `error = actual - predicted`, then updates every weight by `learning_rate * error * input_value`. This is structurally identical to the gradient-based update from Session 2's from-scratch logistic regression and Session 6's closed-form MLE gradient (`error = y - sigma(z)`, weights updated by `learning_rate * error * x`). The perceptron rule predates gradient descent historically, but converges to a similar-looking update because both are, at heart, "move the weights in the direction that would have reduced this specific mistake."

---

## Core Concept 6: Why Two Real Bugs Slipped Into the Given Code

**Bug 1 (wrong loss + missing activation)**: `BinaryCrossentropy` expects two things: labels in `{0, 1}` and a model output that's a single probability per example. But `to_categorical` turned labels into 10-dimensional one-hot vectors, and the model's raw `Dense(10)` output (no activation) isn't even bounded to `[0, 1]`, let alone a valid probability distribution. Keras doesn't always error loudly on a shape/semantic mismatch like this, it just computes *something*, silently wrong, and training limps along near-randomly (9-10% accuracy) instead of crashing. This is the more dangerous kind of bug: no error message, just quietly bad results that could be mistaken for "the model just needs more epochs."

**Bug 2 (validation preprocessing mismatch)**: this one *does* crash, loudly, with a clear shape-mismatch error. The lesson: preprocessing steps (normalizing pixel values, one-hot encoding labels) must be applied identically to every split of the data (train, validation, test). It's an easy mistake to normalize/encode the training set and forget to repeat the exact same steps on validation or test data, and the failure mode ranges from an obvious crash (as here) to a silent, much harder to notice accuracy drop.

**General lesson reinforced again this session**: run the code. An LLM interview (Question 1's exercise) can describe what code is *supposed* to do, but only actually executing it reveals whether it does that. This is the fourth session in a row (Sessions 3, 5, 6, now 7) where actually running the provided code surfaced a real, previously undetected bug.

---

## Common Questions and Confusions

**Q: If softmax and sigmoid both "squash to probabilities," why do we need both?**

A: Sigmoid squashes one number into `(0,1)`, useful for a single yes/no probability (binary classification, or each hidden unit's individual activation). Softmax takes an entire *vector* of raw scores and turns them into a probability distribution that sums to 1 across all classes simultaneously, necessary whenever the output has to be "pick exactly one of N categories" (like one of 10 digits), since the categories compete with each other for probability mass, which sigmoid applied independently to each output wouldn't enforce.

**Q: Does more layers always mean a better model?**

A: No. More layers (depth) let a network represent more complex functions in principle, but they also make training harder (more parameters to learn, more ways to overfit, and later sessions will cover specific problems like vanishing gradients that get worse with depth). The Sonar perceptron (71.0% accuracy, one unit) and the fixed Keras network (86% accuracy, three layers) both do reasonably well on their respective tasks, the "right" depth depends on how complex the true underlying pattern is.

**Q: Why did PCA help the perceptron so much on FashionMNIST (97.5%) but wasn't needed for Sonar (71.0%, already 60 features)?**

A: Sonar already has just 60 hand-engineered numeric features (signal strengths at 60 frequency bands), a small, information-dense input a perceptron can handle directly. Raw FashionMNIST images are 784 individual pixel values, most of which are redundant or irrelevant to distinguishing a shirt from pants (background pixels, minor texture noise). PCA finds the 10 directions of greatest variation across all images, effectively compressing "what actually differs between a top and a pair of trousers" into a small, information-dense representation the same simple perceptron can use directly, instead of drowning in 784 mostly-uninformative raw pixel values.

**Q: Is the step function activation ever used in modern neural networks?**

A: Rarely, and never in hidden layers, its lack of a derivative (it's flat everywhere except one infinitely steep jump) makes gradient-based learning impossible. It survives mainly as a historical/pedagogical tool (the original 1958 Rosenblatt perceptron), and conceptually, as the direct ancestor of smoother modern activations like sigmoid and ReLU that were specifically designed to keep the "threshold-like" decision behavior while being differentiable.

---

## What to Focus On for This Session

1. Be able to state, precisely, the difference between a unit and a layer, many students conflate the two.
2. Be able to write out the forward-pass matrix equations for a network given only its layer sizes and activation functions, without needing to look at code.
3. Recognize that a perceptron is not a fundamentally different thing from a "real" neural network, it's the smallest possible instance of the exact same pattern (1 layer, 1 unit).
4. Notice how the two Keras bugs (silent near-failure vs. loud crash) represent two different failure modes worth watching for in your own code: always check that your loss function's assumptions (label format, output shape) actually match what your model produces.

---

## Next Class Preview (Session 8)

**Topic**: Gradients, Multivariate Derivatives

Session 7 built the forward pass (how a network turns inputs into outputs). Session 8 covers backpropagation, how the error at the output gets pushed backward through every layer's weights, extending the single-layer gradient rules from Sessions 2 and 6 to networks with multiple layers stacked on top of each other.

---

**Study tip**: Before class, try hand-computing a full forward pass (like Core Concept 4's worked example) for a network with 3 inputs, one hidden layer of 2 units, and one output unit, using your own made-up small weight values. Confirm your by-hand answer matches what a 3-line NumPy version of the same network produces.
