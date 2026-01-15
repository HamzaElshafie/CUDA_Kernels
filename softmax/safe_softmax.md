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

## Interview Q&A: safe softmax vs online softmax

**Q: How many global memory passes does this safe softmax kernel do**

**A:** One global read per element and one global write per element. Each thread loads `A[idx]` exactly once into a register `val`, then all further work uses register values plus block reductions. Finally each thread writes one output element `C[idx]`. The kernel performs multiple reduction phases, but those reductions operate on register values and shared memory, not by rereading `A`.

**Q: If it is one global read and one write, why do we need online softmax**

**A:** Because this one read behaviour relies on a simplifying assumption: one block covers the entire softmax row, so each element is loaded once and can be reused locally until normalisation. In practice, the per token vector length, for example `KV_LEN` in attention, can exceed what one CUDA block can handle efficiently, and the maximum threads per block is `1024`. When a single row is larger than a block, we need either multiple blocks per row or each thread must process multiple elements in a loop.

**Q: What changes when the row is larger than one block**

**A:** Block reductions such as `block_reduce_max` and `block_reduce_sum` only reduce within a block. With multiple blocks per row, we need a way to compute the row wise max and sum across blocks. That typically requires extra global coordination, for example writing per block partials to global memory and launching a second kernel to reduce them. Either way, the result is more global memory traffic and often extra passes over the row or extra intermediate writes.

**Q: What is the role of online softmax in FlashAttention style kernels**

**A:** Online softmax is a numerically stable way to stream through the row while maintaining a running max and a running normalisation term. It enables computing softmax normalisation while iterating over chunks of the row, which is essential when the row is too large to fit in a single block or in registers. In FlashAttention, it is combined with fusion: we never materialise the full softmax output to global memory. We compute softmax weights on the fly and immediately apply them to `V`, reducing global memory reads and writes and improving latency.

**Q: What is the main takeaway for the interview**

**A:** My current safe softmax is efficient for the single block per row case: one global read per element, one global write per element. Online softmax becomes necessary when the row length is larger than a block, or when we want a fused attention style implementation that avoids storing softmax outputs and instead streams through `K` and `V`.

