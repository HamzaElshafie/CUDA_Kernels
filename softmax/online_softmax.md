# Online softmax: streaming stability for long rows

## Why online softmax exists

Safe softmax is stable, but it assumes you can compute a row wise max and a row wise normalisation term efficiently:

$$
y_i = \frac{e^{x_i - m}}{\sum_{j=0}^{N-1} e^{x_j - m}}
\quad \text{where} \quad
m = \max_{j} x_j
$$

If the whole softmax row fits inside one CUDA block, we can do this with two block reductions: one for the max and one for the sum. The difficulty appears when the row length `N` is large, for example `KV_LEN` in attention, and the row must be processed in **chunks** because `N` can exceed what a block with at most `1024` threads can cover efficiently.

Online softmax is the standard way to compute the same numerically stable softmax while **streaming over chunks** of the row, maintaining the correct normalisation as you go.

## The key idea behind online softmax

Online softmax maintains two running quantities while scanning the row:

- a running maximum value, call it $$m$$
- a running normalisation term, call it $$l$$

After you have processed some prefix of the row, you want:

$$
m = \max(\text{values seen so far})
$$

$$
l = \sum_{\text{values seen so far}} e^{x - m}
$$

When you see a new chunk, the chunk may contain a larger maximum. If the maximum increases, all previously accumulated exponentials must be rescaled so that they are expressed relative to the new maximum. That rescaling is the heart of online softmax.

## The update equations

Assume you have already processed some prefix of the row and have a running pair $$(m, l)$$.

Now you process a new chunk and compute two chunk local quantities:

- chunk max:
  
$$
m_{\text{new}} = \max(\text{chunk})
$$

- chunk normaliser relative to the chunk max:
  
$$
l_{\text{new}} = \sum_{\text{chunk}} e^{x - m_{\text{new}}}
$$

To merge old and new, first compute the combined max:

$$
m' = \max(m, m_{\text{new}})
$$

Now express both the old sum and the new chunk sum relative to $$m'$$.

The old part rescales by $$e^{m - m'}$$ and the new chunk rescales by $$e^{m_{\text{new}} - m'}$$:

$$
l' = l \cdot e^{m - m'} + l_{\text{new}} \cdot e^{m_{\text{new}} - m'}
$$

This update ensures the running normaliser stays correct even when the maximum changes.

## Logic steps of online softmax

Conceptually, online softmax performs a repeated set of steps over chunks of the row:

1. Initialise the running state:
   
$$
m = -\infty
$$

$$
l = 0
$$

2. For each chunk of the row, compute the chunk local max and sum:

$$
m_{\text{new}} = \max(\text{chunk})
$$

$$
l_{\text{new}} = \sum_{\text{chunk}} e^{x - m_{\text{new}}}
$$

3. Update the running max and running normaliser using rescaling:

$$
m' = \max(m, m_{\text{new}})
$$

$$
l' = l \cdot e^{m - m'} + l_{\text{new}} \cdot e^{m_{\text{new}} - m'}
$$

4. Set:

$$
m \leftarrow m'
$$

$$
l \leftarrow l'
$$

5. After all chunks have been processed, the final stable softmax normalisation is given by the final running values $$m$$ and $$l$$. Each element can be normalised as:

$$
y_i = \frac{e^{x_i - m}}{l}
$$

The important difference from safe softmax is that $$m$$ and $$l$$ are produced incrementally while streaming, rather than requiring the full row to be reduced in one go.

## GPU mapping for online per token softmax

For a long row, you typically map the work like this:

- The row is processed in tiles (chunks).
- For each tile, threads compute local quantities needed for $$m_{\text{new}}$$ and $$l_{\text{new}}$$ using warp or block reductions.
- A running pair $$ (m, l) $$ is updated per row using the update equations above.
- Normalisation uses the final $$m$$ and $$l$$.

In FlashAttention style kernels, online softmax is usually fused into attention so that the softmax weights are never written to global memory. Instead, each tile’s weights are used immediately to update the output.

## What changes compared to safe softmax

Safe softmax for a row is conceptually:

- one reduction to get $$m$$ over the full row
- one reduction to get $$\sum e^{x - m}$$ over the full row
- one final normalisation

Online softmax is designed for the case where the row is processed in chunks. Instead of needing a single global max and global sum over the row upfront, it maintains the correct max and normaliser while streaming through tiles, which avoids extra global coordination when a row cannot be handled by one block.

## Main takeaway

Online softmax computes the same stable result as safe softmax, but it is structured for long softmax rows that must be processed in chunks. The running max and running normaliser update lets you stream over the row while staying numerically stable, which is essential for long sequence attention and for fused kernels like FlashAttention.
