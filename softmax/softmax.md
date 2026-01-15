# Softmax: concept and GPU implementation (per token)

## What is softmax

Softmax is a function that takes a vector of real numbers and converts it into a vector of non negative values that sum to `1`. Because the outputs sum to `1`, they can be interpreted as a probability distribution.

Given an input vector `x` of length `K`, softmax is defined elementwise as:

\[
y_i = \frac{e^{x_i}}{\sum_{j=0}^{K-1} e^{x_j}}
\]

Two important properties fall out of this definition:

- **Non negativity**: `e^{x_i} > 0`, so every output `y_i` is positive.
- **Normalisation**: the denominator is the sum of all numerators, so `sum(y) = 1`.

Softmax is used in many places, but in modern ML systems the most common case is attention. In attention, softmax turns a row of attention scores into normalised weights that are then used to take a weighted sum of values.

## How to interpret the formula

The softmax formula has a simple interpretation:

1. Exponentiate each element: `e^{x_i}`  
   This makes all values positive and emphasises larger values.

2. Compute the sum of exponentials: `Z = sum_j e^{x_j}`  
   This is the normalisation constant, sometimes called the partition function.

3. Divide each exponential by the sum: `y_i = e^{x_i} / Z`  
   This normalises the outputs so they sum to `1`.

So the softmax computation can be viewed as two passes over the data:

- **Pass 1**: compute `e^{x_i}` and accumulate the sum of exponentials
- **Pass 2**: divide each `e^{x_i}` by the sum

## What does “per token” softmax mean

In many GPU kernels, softmax is applied not to one huge vector but to many independent vectors. For example, an attention score matrix can be thought of as many rows, and softmax is applied independently to each row.

“Per token” softmax means each independent vector corresponds to one token's set of scores. Concretely, for one token you have `K` scores, and you compute softmax across those `K` values only. The next token has its own `K` values and its own independent softmax.

This independence is important: softmax normalises within one vector, not across different tokens.

## The GPU logic for per token softmax

On the GPU, the natural mapping is:

- **One block handles one token** (one vector of length `K`)
- **Threads in the block handle elements in that vector**
- The denominator `sum_j e^{x_j}` is computed as a **block reduction**
- Each thread writes its output element after the denominator is known

The core challenge is that the denominator depends on all elements, so threads must cooperate.

## Steps of a per token softmax kernel

Assuming `K == blockDim.x` and each thread handles one element:

1. Compute the global index for the element this thread owns.
2. Load `x[idx]` and compute `exp_val = expf(x[idx])`.
3. Reduce `exp_val` across the block to get `exp_sum`.
4. Compute `y[idx] = exp_val / exp_sum`.

Written as pseudocode:

```cpp
exp_val = expf(x[idx]);
exp_sum = block_reduce_sum(exp_val);
y[idx] = exp_val / exp_sum;
```