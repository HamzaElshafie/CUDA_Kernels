# Safe softmax: numerical stability and GPU implementation

## Why naive softmax can be unsafe

The naive softmax formula is:

$$
y_i = \frac{e^{x_i}}{\sum_{j=0}^{N-1} e^{x_j}}
$$

This is mathematically correct, but it can be numerically unstable in floating point. The issue is the exponential function. For large positive inputs, $$e^{x}$$ grows extremely fast and can overflow to `inf` in `float32`. If any $$e^{x_i}$$ becomes `inf`, then the denominator can become `inf`, and the output can turn into `nan` due to `inf / inf` or other undefined operations.

Even when it does not overflow, very large or very small exponentials can cause loss of precision in the denominator due to limited mantissa bits.

## The key identity behind safe softmax

Safe softmax uses a simple identity:

$$
\frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
$$

where $$m$$ is any constant. The most useful choice is:

$$
m = \max_j x_j
$$

Subtracting $$m$$ shifts all inputs so that the largest value becomes `0`. That guarantees:

- the largest exponent is $$e^0 = 1$$
- all other exponents are in the range $$(0, 1]$$

This prevents overflow and usually improves numerical accuracy.

The safe softmax formula is therefore:

$$
y_i = \frac{e^{x_i - m}}{\sum_{j=0}^{N-1} e^{x_j - m}}
\quad \text{where} \quad
m = \max_{j} x_j
$$

## Logic steps of safe softmax

Safe softmax adds one extra reduction compared to naive softmax. Conceptually, the steps are:

1. Compute the maximum value:

$$
m = \max_{j=0}^{N-1} x_j
$$

3. Compute the shifted exponentials and their sum:
   
$$
\mathrm{expSum} = \sum_{j=0}^{N-1} e^{x_j - m}
$$

5. Compute the final outputs:
   
$$
y_i = \frac{e^{x_i - m}}{\mathrm{expSum}}
$$

So safe softmax is still a two pass structure, but the first pass computes a max instead of a sum.

## GPU mapping for safe per token softmax

For per token softmax, each block handles one token vector of length `N`. Threads cooperate in two block level reductions:

- a **block max reduction** to find $$m$$
- a **block sum reduction** to find $$\mathrm{expSum}$$

A typical per token safe softmax kernel follows this structure:

1. Each thread loads one element $$x_i$$.
2. Reduce across the block to get $$m$$.
3. Each thread computes $$e^{x_i - m}$$.
4. Reduce across the block to get $$\mathrm{expSum}$$.
5. Each thread writes $$y_i = e^{x_i - m} / \mathrm{expSum}$$.

This is almost identical to the naive kernel, but with an added max reduction and a subtraction before exponentiation.

## What changes compared to naive softmax

Naive softmax requires only:

- one block sum reduction over $$e^{x}$$

Safe softmax requires:

- one block max reduction over $$x$$
- one block sum reduction over $$e^{x - m}$$

The extra max reduction is the cost you pay for stability. In practice it is almost always worth it, especially in attention, because attention logits can vary widely and overflow is a real risk.

