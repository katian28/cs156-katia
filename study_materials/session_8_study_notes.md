# Session 8 Study Notes: Gradients and Backpropagation

**Purpose**: Deep explanation of computation graphs, the chain rule, Jacobians, and automatic differentiation. Read before class.

---

## Core Concept 1: Why We Need Gradients At All

Most models in this course so far have had one of two fitting strategies:

1. **Closed-form solution** (Session 3's normal equation, Session 6's Gaussian MLE): set the derivative to zero, solve algebraically, done in one step.
2. **Iterative gradient-based search** (Session 2's logistic regression, Session 6's MLE gradient, Session 7's perceptron): no algebraic solution exists, so instead compute the gradient (which way is downhill) and take a small step, repeatedly.

Neural networks fall entirely into category 2. There is no formula you can solve for the "best" weights of a 3-layer network, the composition of matrix multiplications and non-linear activations makes the loss function too complex. So we're stuck computing gradients, many times, for potentially millions of parameters. Backpropagation is the algorithm that makes this computationally feasible.

---

## Core Concept 2: A Computation Graph Is Just Your Code, Drawn Out

The scalar sigmoid example breaks one formula into tiny individual operations:

```
mul0 = w0 * x0
mul1 = w1 * x1
sum01 = mul0 + mul1
sum012 = sum01 + w2
neg = -sum012
exp = e^neg
plus1 = exp + 1
invert = 1/plus1
```

Each line is a "gate" (a node in the computation graph): it takes one or two inputs and produces one output. Drawing this as a graph, arrows flow left to right (the forward pass, computing the final `invert` value from the inputs).

**Why break it into such tiny pieces?** Because each individual operation (`multiply`, `add`, `negate`, `exp`, `+1`, `invert`) has a dead-simple, well-known derivative. Backpropagation exploits this: instead of finding the derivative of the whole complicated formula at once (hard), it finds the derivative of each tiny piece (easy), then chains them together.

---

## Core Concept 3: The Chain Rule, As a Backward Walk

The chain rule from calculus says: if `y` depends on `x` through some intermediate variable `u` (i.e. `y = f(u)`, `u = g(x)`), then:

```
dy/dx = dy/du * du/dx
```

Backpropagation applies this repeatedly, walking backward through the graph one gate at a time. Start at the output (`d_invert = 1.00`, since the derivative of anything with respect to itself is 1), then at each gate going backward, multiply the gradient handed to you from downstream by that gate's own local derivative, and pass the result further upstream.

**This is why the PCW's hint says "backpropagation is a strictly local process."** Standing at the `mul1 = w1 * x1` gate, you don't need to know anything about `sum01`, `sum012`, `exp`, or any other part of the graph. You only need two things: the gradient handed to you from downstream (`d_mul1`), and your own gate's local derivative rules (`d(mul1)/d(w1) = x1` and `d(mul1)/d(x1) = w1`). Multiply them together and you're done, that's your contribution.

---

## Core Concept 4: The Two Rules That Solve Every Gate in This Graph

Every gate in the sigmoid example is one of these two types:

**Multiply gate** (`mul0 = w0 * x0`): the derivative with respect to each input is the *other* input.
```
d(a*b)/da = b
d(a*b)/db = a
```
Concretely: `mul1 = w1 * x1` gives `d(mul1)/d(w1) = x1` and `d(mul1)/d(x1) = w1`. Notice the swap: the gradient with respect to `w1` uses `x1`'s *value*, not `w1`'s.

**Add gate** (`sum01 = mul0 + mul1`): the derivative with respect to each input is just 1.
```
d(a+b)/da = 1
d(a+b)/db = 1
```
An add gate is a pure "gradient distributor," it passes the incoming gradient through unchanged to both of its inputs. This is why `d_mul0` and `d_mul1` both equal `d_sum01` exactly, no scaling happens at an add gate.

Every other gate in the example (`neg`, `exp`, `plus1`, `invert`) is a single-input function, so its local derivative comes straight from single-variable calculus:
```
neg = -sum012        ->  d(neg)/d(sum012) = -1
exp = e^neg           ->  d(exp)/d(neg) = e^neg  (the function's own value!)
plus1 = exp + 1       ->  d(plus1)/d(exp) = 1
invert = 1/plus1      ->  d(invert)/d(plus1) = -1/plus1^2
```

---

## Core Concept 5: Multiple Downstream Paths Add Up

Notice `d_w2, d_sum01 = d_sum012, d_sum012`, both get the *same* value. This is because `sum012 = sum01 + w2` has two inputs, and (per the add-gate rule) both inputs receive the full incoming gradient unchanged. More generally, whenever a single value feeds into *multiple* downstream gates, its total gradient is the **sum** of the gradients flowing back from each path it feeds into. The PCW's comment captures this: "for gates with multiple inputs, when going backward they are responsible for derivatives of multiple values." In this particular example every value only feeds one downstream gate, so no actual summing is needed, but in a real neural network (where one unit's output feeds into every unit in the next layer), this summing-of-multiple-paths is constant and essential.

---

## Core Concept 6: Vectors Don't Change the Math, Only the Bookkeeping

Q2B asks you to derive `df/dw` for `f(w,x) = w^T x = w1*x1 + w2*x2 + w3*x3`, given that `df/dx = w` is already shown.

Look at any single term, `w1*x1`. By the multiply-gate rule from Concept 4, `d(w1*x1)/d(w1) = x1`. Since this holds for every term independently, stacking them back into a vector gives `df/dw = [x1, x2, x3]^T = x`.

**So**: `d(w^T x)/dw = x` and `d(w^T x)/dx = w`. This is the exact same "swap the two inputs" rule from the scalar multiply gate, just applied elementwise and packed back into vectors. Nothing new is happening mathematically, `w.T @ x` is just `w0*x0 + w1*x1` fused into one operation, so its gradient is the vector version of the same swap.

---

## Core Concept 7: Gradient vs. Jacobian

This is the distinction the PCW explicitly asks you to interview an LLM about, and it's worth being precise.

**A gradient** exists for a function that outputs a single scalar number, from possibly many inputs. It's a vector: one partial derivative per input, all describing how that one output number changes.

**A Jacobian** exists for a function that outputs a whole *vector*, from possibly many inputs. It's a matrix: every output component gets its own row of partial derivatives with respect to every input.

**Where this shows up in a real network**: the network's final loss is one scalar number, so `d(loss)/d(any single weight)` is a genuine gradient component. But look at just one layer in isolation: it takes a vector in (say 784 pixel values) and produces a vector out (say 128 activations). The relationship between that layer's input vector and output vector is vector-to-vector, so its derivative really is a full Jacobian matrix, not a gradient.

**Why backpropagation never actually computes these Jacobians explicitly**: because the final loss is scalar, at every step backward through the network, what you're actually propagating is "the gradient of the scalar loss with respect to this layer's output," which is a *vector* (one number per output unit), not a full Jacobian. Multiplying that vector by the current layer's local Jacobian (a **Jacobian-vector product**) collapses straight back down to another vector (the gradient with respect to this layer's *input*), which becomes the vector handed to the next layer backward. The full Jacobian matrix is never materialized anywhere, only ever used implicitly inside these vector products. This is precisely why reverse-mode automatic differentiation (what JAX and every deep learning framework do) scales to networks with millions of parameters: it never pays the cost of forming or storing a full Jacobian, only ever propagating vectors.

---

## Core Concept 8: What JAX Actually Automates

Every gradient computed by hand in this PCW, JAX reproduces automatically from one line:

```python
grad_sigmoid_example = jax.grad(sigmoid_example)
grad_sigmoid_example(params, x)
```

`jax.grad` takes a Python function (the forward pass, written completely normally) and returns a *new* function that computes its gradient, using reverse-mode automatic differentiation under the hood, the exact chain-rule, gate-by-gate backward walk done by hand above, just implemented generically for any function you write.

**Why this matters practically**: you only ever have to write the forward pass once. No more writing a matching, error-prone backward pass by hand for every new model architecture, verified in this PCW to match the hand-derived gradients exactly (`d_w=[-0.1966, -0.3932]`, `d_b=0.1966`, identical across all three methods: scalar-by-hand, vectorized NumPy, and JAX).

---

## Common Questions and Confusions

**Q: Why does `d(exp)/d(neg)` equal `exp` itself, not some other formula?**

A: This is a special property of the exponential function: its own derivative is itself, `d(e^x)/dx = e^x`. So at the point `neg = -1.00`, the local derivative is just `e^(-1.00) = 0.3679`, which happens to be the exact same number as the `exp` variable already computed during the forward pass. This is why the forward pass values need to be stored, they get reused directly as local derivatives during the backward pass.

**Q: Why do add gates "pass gradients through unchanged" but multiply gates "swap" them?**

A: Because their local derivatives are different shapes. An add gate's derivative with respect to either input is the constant 1 (adding doesn't scale anything), so multiplying the incoming gradient by 1 just passes it through. A multiply gate's derivative with respect to one input is literally the *value* of the other input (since `d(ab)/da = b`), so the incoming gradient gets scaled by that other value, which is why it looks like a "swap."

**Q: If backpropagation just needs local derivatives, why not compute the derivative of the whole formula at once with one big chain-rule expression?**

A: You mathematically could, but it would need to be re-derived by hand for every new model architecture, and the resulting single formula would be enormous and error-prone for anything beyond a toy example. Breaking it into small local gates means each individual local derivative rule is simple and reusable (every multiply gate anywhere in any network follows the exact same swap rule), and the chain rule mechanically strings them together. This is precisely what lets a generic tool like JAX handle *any* function you write, without needing model-specific calculus.

**Q: Does gradient descent always converge to the true minimum?**

A: Not guaranteed in general (a complex enough loss surface can have multiple local minima that gradient descent gets stuck in), but for this PCW's sigmoid example, and for logistic regression more broadly (Session 6), the loss surface is well-behaved (convex), so gradient descent reliably converges. The extension exercise confirms this empirically: starting from `w=[2,-3], b=-3` (output 0.731), 1000 iterations of gradient descent drove the output down to 0.0017, essentially reaching the target of 0.

---

## What to Focus On for This Session

1. Be able to walk the sigmoid example's backward pass from memory, using only the two local gate rules (multiply swaps, add passes through).
2. Be able to state precisely why the network's overall derivative is a gradient (scalar loss) while an individual layer's derivative is a Jacobian (vector-to-vector), and why backprop only ever needs Jacobian-vector products, never the full matrix.
3. Confirm you can get the identical answer three ways: by hand, in vectorized NumPy, and via `jax.grad`, this cross-verification is itself good practice for catching your own arithmetic mistakes.

---

## Next Class Preview (Session 9)

**Topic**: Metrics and Cross-Validation

Session 8 covered how models get fit (the gradient machinery). Session 9 shifts focus to evaluating whether a fitted model actually generalizes to new data, versus just memorizing its training set, using the loss values, gradients, and models built throughout Unit 1 as the concrete examples to evaluate.

---

**Study tip**: Before class, try hand-deriving the backward pass for a tiny 3-gate graph you make up yourself (e.g. `y = (a + b) * c`), using only the two local rules from Core Concept 4. If your by-hand answer matches what `jax.grad` gives you for the same function, you've internalized the mechanics, not just memorized this specific sigmoid example.
