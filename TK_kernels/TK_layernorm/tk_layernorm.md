## Fused LayerNorm in ThunderKittens

The TK kernel we implemented is a fused residual addition, optional dropout, and LayerNorm kernel. The kernel matches what the original Transformer model would be doing, namely post LayerNorm. In future kernels, I will implement RMSNorm, which is what most modern LLMs use, and that will be in the pre LayerNorm style.

Given an input branch `x` and a residual branch `residual`, the kernel first forms the residual stream:

$$
r = residual + dropout(x)
$$

It then stores this intermediate result, and finally applies LayerNorm to it:

$$
\mu = \frac{1}{D} \sum_{i=0}^{D-1} r_i
$$

$$
\sigma^2 = \frac{1}{D} \sum_{i=0}^{D-1} (r_i - \mu)^2
$$

$$
\hat{r}_i = \frac{r_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
y_i = \gamma_i \hat{r}_i + \beta_i
$$

where:

1. `D` is the hidden dimension
2. $\gamma$ is the learned scale vector
3. $\beta$ is the learned bias vector
4. $\epsilon$ is a small constant for numerical stability

So the fused kernel produces two outputs conceptually:

$$
o_{resid} = residual + dropout(x)
$$

$$
o = LayerNorm(o_{resid})
$$

---

### Why this fused form exists

In modern Transformer blocks, residual addition and LayerNorm often appear right next to each other. If we implemented them as separate kernels, we would repeatedly read and write the same token vector from HBM:

1. load `x`
2. load `residual`
3. write the residual sum
4. read that sum back again
5. normalise it
6. write the final output

That is wasteful, especially because LayerNorm is usually far more memory bound than compute bound. A fused kernel reduces this traffic by keeping intermediate values closer to the SM and only writing out what is needed.

---

### Recap: what LayerNorm does

LayerNorm normalises across the feature dimension for each sample independently.

If we take one token vector of length `D`:

$$
x = (x_0, x_1, \dots, x_{D-1})
$$

then LayerNorm computes the mean:

$$
\mu = \frac{1}{D} \sum_{i=0}^{D-1} x_i
$$

the variance:

$$
\sigma^2 = \frac{1}{D} \sum_{i=0}^{D-1} (x_i - \mu)^2
$$

and then produces the normalised result:

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

followed by the learned affine transform:

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

So, just like ordinary LayerNorm, the fused version still contains two reductions per token:

1. one reduction for the mean
2. one reduction for the variance

The only difference is that the input to those reductions is not the original `x`, but the residual updated vector:

$$
r = residual + dropout(x)
$$

---

### Why LayerNorm matters in Transformers

Before LayerNorm, BatchNorm was widely used in deep networks. But BatchNorm depends on batch level statistics, which makes it awkward for sequence models and autoregressive inference. Transformers instead need something that works independently for each token and does not couple samples together.

LayerNorm solves that by normalising within each sample, across features. This makes it a natural fit for Transformer architectures, including the original Transformer, BERT, GPT style models, and many of their descendants.

In practice, Transformer style LayerNorm is almost always applied over the hidden dimension of a token vector, which is exactly the setting of this TK kernel.

---

### Logic steps of the fused operation

For one token vector of length `D`, the fused residual dropout LayerNorm can be thought of as:

1. optionally apply dropout to the branch input `x`
2. add the result into the residual stream
3. store the residual updated vector
4. compute the mean of that vector
5. compute the variance of that vector
6. compute the inverse standard deviation
7. normalise
8. apply the learned scale and bias
9. store the final output

In pseudocode:

```text
r = residual + dropout(x)
o_resid = r

mean = reduce_sum(r) / D
var  = reduce_sum((r - mean)^2) / D
inv_std = 1 / sqrt(var + eps)

o = ((r - mean) * inv_std) * gamma + beta
```

This is still fundamentally a reduction kernel, but one wrapped around a residual update.

---

### GPU mapping in the ThunderKittens version

For Transformer workloads, LayerNorm is usually applied per token. That means each token corresponds to one vector of length D, where D is the hidden size. In this kernel, D is fixed to 1024.

The key design choice in this TK version is that it is organised as a warp level vector kernel rather than a tensor core MMA kernel.

Conceptually:

1. each warp processes one token vector
2. the token vector is staged through TK shared vectors
3. warp level operations perform the reductions and pointwise arithmetic

So instead of thinking in terms of matrix multiply tiles, it is better to think of this kernel as operating on a 1 × D vector per token.

A simple mental model is:

1. one warp
2. one token vector
3. one residual update
4. one mean reduction
5. one variance reduction
6. one normalised output

So one takeaway from this kernel is that TK is not about GEMM and attention style matrix kernels only. It also provides structured abstractions for vector style kernels such as norms and activations.

---

### How this maps onto TK abstractions

Even though this is not a matrix multiply kernel, it still uses the same overall TK programming model.

The input and output tensors live in GMEM and are described with gl.

Shared memory is used to stage token sized vectors before computation.

The actual math is then expressed through warp level operations over TK vector types.

So the data flow is still the familiar one:

gl -> shared vector -> warp operations
HBM -> SMEM -> warp compute

The difference is simply that the compute object here is a vector rather than a large matrix tile.

That is why this kernel is a good example of how broad the TK programming model really is. The same abstractions that were introduced for tiled GPU programming also adapt naturally to reduction heavy kernels like LayerNorm.

---

### Numerical stability considerations

As with any LayerNorm implementation, the variance is regularised by adding a small $\epsilon$ before taking the square root:

$$
\sqrt{\sigma^2 + \epsilon}
$$

This prevents division by zero and keeps the normalisation stable when the variance becomes very small.

LayerNorm does not involve exponentials, so it is usually easier numerically than softmax. However, precision still matters because reductions over long hidden dimensions can accumulate rounding error. In practice, many implementations keep accumulation in higher precision even when the input and output are BF16.

That matters especially for hidden dimensions such as 1024, 2048, or larger, where the cost of reduction error becomes more visible.

---

### Performance perspective

LayerNorm is usually memory bound rather than compute bound. Each element participates in only a modest amount of arithmetic, but the kernel still has to read and write a large number of values. This is exactly why fusion matters so much here.

By fusing dropout, residual addition, and LayerNorm, we reduce the number of times the same token vector has to travel back and forth to HBM. That often matters more than shaving off a few arithmetic instructions.

So when reasoning about performance, the main questions are usually:

1. how many times do we touch HBM
2. how efficiently do we stage the token vector
3. how efficiently do we perform the reductions
4. how much unnecessary intermediate storage do we avoid

---

### Unfused versus fused reasoning

To make the effect of fusion more concrete, it is helpful to first ignore dropout and RNG and compare a simplified unfused residual plus LayerNorm pipeline against a fused version. The main point of this comparison is not to get an exact final performance model. It is simply to see how much unnecessary HBM traffic disappears once the intermediate residual sum no longer has to be written out and read back again.

Assume a local post LayerNorm style operation of the form:

$$
tmp = x + residual
$$

$$
out = LayerNorm(tmp)
$$

where each token vector has length D.

**Unfused version**

In a naive unfused implementation, the residual addition and LayerNorm are separated, so the intermediate tmp has to travel through HBM.

Per token, the traffic is:

1. read x → D
2. read residual → D
3. write tmp = x + residual → D
4. read tmp again to compute the mean → D
5. read tmp again to compute the variance → D
6. read tmp again for the final normalisation pass → D
7. read gamma → D
8. read beta → D
9. write out → D

So the total per token is:

$$
\text{reads} = 7D
$$

$$
\text{writes} = 2D
$$

which gives

$$
\text{total values moved} = 9D
$$

If we assume BF16 inputs and outputs, that means each value is 2 bytes, so:

$$
\text{bytes moved} \approx 18D
$$

For a rough arithmetic estimate, we can count:

1. residual add → about D
2. mean reduction → about D
3. variance computation and reduction → about 3D
4. final normalise plus affine transform → about 4D

which gives roughly:

$$
\text{FLOPs} \approx 9D
$$

So the arithmetic intensity is approximately:

$$
AI_{unfused} \approx \frac{9D}{18D} = 0.5 \text{ FLOP/byte}
$$

This is very low, which already suggests that the unfused version is strongly memory bound.

**Fused version**

Now consider the fused version, still ignoring dropout and RNG. The arithmetic is basically the same, but now the intermediate residual sum is kept on chip rather than being written to HBM and read back again.

Per token, the traffic becomes:

1. read x → D
2. read residual → D
3. form tmp = x + residual locally → no HBM write
4. compute mean from local staged data → no extra HBM read
5. compute variance from local staged data → no extra HBM read
6. final normalisation pass reads gamma and beta → 2D
7. write out → D

So the total per token is:

$$
\text{reads} = 4D
$$

$$
\text{writes} = D
$$

which gives

$$
\text{total values moved} = 5D
$$

Again assuming BF16 inputs and outputs:

$$
\text{bytes moved} \approx 10D
$$

The arithmetic is still roughly:

$$
\text{FLOPs} \approx 9D
$$

so the arithmetic intensity becomes:

$$
AI_{fused} \approx \frac{9D}{10D} = 0.9 \text{ FLOP/byte}
$$

**What this comparison tells us**

The exact numbers are only rough estimates, but the trend is clear. Fusion does not dramatically change the amount of arithmetic. What it changes is the amount of HBM traffic. In the unfused case, the intermediate residual sum is written out and then read back several times. In the fused case, that intermediate stays local, so the denominator in the arithmetic intensity calculation drops.

That is the real reason fusion helps here.

Ignoring dropout and RNG is fine for this comparison because they do not change the main argument. Dropout adds some extra work, and RNG is not free, but the dominant effect of fusion still comes from avoiding unnecessary HBM reads and writes.

---

### Main takeaway

The ThunderKittens LayerNorm kernel is best understood as a fused residual update and LayerNorm kernel rather than just a normalisation kernel in isolation.

Mathematically, it first forms

$$
o_{resid} = residual + dropout(x)
$$

and then computes

$$
o = LayerNorm(o_{resid})
$$

From a GPU perspective, it is a reduction heavy, memory sensitive, per token vector kernel. From a TK perspective, it is a good example that the programming model is not limited to tensor core MMA kernels. The same abstractions also support vector style kernels where the main work is staging data, reducing across features, and expressing the computation through warp level operations.

The unfused versus fused comparison makes the performance intuition especially clear. In both cases, the arithmetic is broadly similar. What changes is how often the intermediate vector is forced to travel through HBM. That is exactly why fusion is such an important optimisation lever for kernels of this kind.
