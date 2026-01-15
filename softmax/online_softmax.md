# Online softmax: streaming, stability, and why it exists

## The problem online softmax solves

Safe softmax for a vector `x` of length `N` is:

$$
y_i = \frac{e^{x_i - m}}{\sum_{j=0}^{N-1} e^{x_j - m}}
\quad \text{where} \quad
m = \max_{j} x_j
$$

This is stable, but it suggests a workflow:

1. compute the max `m`
2. compute the sum of shifted exponentials
3. normalise

If the whole softmax row fits in one block, this is easy: one block does a max reduction, then a sum reduction, then writes outputs.

The difficulty appears when the row is too large to fit in one block or registers, for example `KV_LEN` in attention at long sequence lengths. Then the row must be processed in **chunks**. Block reductions only work inside a block, so computing a row wise max and sum across multiple chunks can require extra global memory traffic or multiple kernels.

Online softmax is a way to compute the same stable softmax while streaming over chunks, maintaining a running max and a running normalisation term.

## Core idea: maintain a running max and running normaliser

Imagine scanning the row from left to right in chunks. After processing some prefix of the row, we would like to know:

- the maximum value seen so far
- the sum of exponentials relative to that maximum

Define after processing a prefix:

- running max: $$m$$
- running normaliser: $$l = \sum e^{x - m}$$ over the processed elements

When we extend the prefix with a new chunk, we update `m` and `l` without needing to revisit older elements.

## The key update equations

Suppose we have a running pair `(m, l)` for the prefix we have already processed.

Now we process a new chunk and compute:

- the chunk max: $$m_{\text{new}} = \max(\text{chunk})$$
- the chunk sum relative to its own max:
  
$$
l_{\text{new}} = \sum_{\text{chunk}} e^{x - m_{\text{new}}}
$$

The combined max is:

$$
m' = \max(m, m_{\text{new}})
$$

To combine the sums, we must express both sums using the same reference max `m'`.

If `m'` is larger than `m`, then the old sum `l` must be rescaled:

$$
\sum_{\text{old}} e^{x - m'} = \sum_{\text{old}} e^{x - m} \cdot e^{m - m'} = l \cdot e^{m - m'}
$$

Similarly, the new chunk sum must be rescaled if `m'` is larger than `m_new`:

$$
\sum_{\text{new}} e^{x - m'} = l_{\text{new}} \cdot e^{m_{\text{new}} - m'}
$$

So the update becomes:

$$
m' = \max(m, m_{\text{new}})
$$

$$
l' = l \cdot e^{m - m'} + l_{\text{new}} \cdot e^{m_{\text{new}} - m'}
$$

This is the essence of online softmax: whenever the running max increases, we down scale the previously accumulated sum to keep everything in a stable range.

## Why this is numerically stable

At every step, exponentials are computed after subtracting a max (`m_new` within the chunk), and the accumulated sum is always expressed relative to the current best max (`m'`). This keeps values in a reasonable range and prevents overflow, like safe softmax, but now we can do it incrementally.

## How this maps to GPU kernels

Online softmax becomes useful when `N` is large and we must stream over the row:

- each block processes the row in tiles (or multiple blocks cooperate over a row)
- each tile computes a local max and local sum
- the running `(m, l)` pair is updated using the equations above

In FlashAttention style kernels, online softmax is typically fused with the attention computation:

- compute a tile of logits `q · k`
- update running `(m, l)`
- compute softmax weights for the tile relative to the current running max
- immediately apply the weights to accumulate the output `o += p · v`

This avoids materialising the full attention matrix and often avoids writing the softmax output to global memory at all.

## What is the main takeaway

Online softmax computes the same result as safe softmax, but it is designed for the case where a softmax row must be processed in chunks. It provides a numerically stable way to maintain the correct normalisation while streaming, which is essential for long sequence attention and for fused attention kernels like FlashAttention.
