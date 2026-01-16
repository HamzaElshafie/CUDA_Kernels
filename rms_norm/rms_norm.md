# RMSNorm: simpler normalisation for modern Transformers

## What RMSNorm is

RMSNorm (Root Mean Square Normalisation) is a normalisation layer that, like LayerNorm, normalises each token independently across its feature dimension. Conceptually, if a model activation tensor has shape:

- $$x \in \mathbb{R}^{M \times K}$$

where:

- $$M$$ is the number of tokens (rows)
- $$K$$ is the hidden size or feature dimension (columns)

then RMSNorm is applied **per row**, reducing over the **K features** for each token.

RMSNorm is widely used in modern Transformer architectures (for example LLaMA style models) because it is cheaper than LayerNorm while working well in practice.


## LayerNorm recap and what RMSNorm changes

LayerNorm normalises using both the mean and the variance:

$$
\mu = \frac{1}{K}\sum_{j=0}^{K-1} x_j
$$

$$
\sigma^2 = \frac{1}{K}\sum_{j=0}^{K-1}(x_j - \mu)^2
$$

$$
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i
$$

RMSNorm removes the mean subtraction. It only normalises by the RMS magnitude:

$$
\mathrm{RMS}(x) = \sqrt{\frac{1}{K}\sum_{j=0}^{K-1} x_j^2}
$$

and produces:

$$
y_i = \frac{x_i}{\mathrm{RMS}(x) + \epsilon} \cdot g_i
$$

In many implementations, the trainable scale parameter is called $$g$$ (or $$\gamma$$). Some RMSNorm variants omit an additive bias term entirely.


## Why RMSNorm exists

RMSNorm is attractive for GPU kernels because it is simpler:

- No mean computation
- No subtraction of the mean
- No variance of residuals

That means:

- one reduction instead of two
- fewer synchronisation points
- fewer FLOPs
- less shared memory traffic

This simplification can reduce normalisation overhead in Transformer blocks.


## Logic steps of RMSNorm

For one token (one row of length $$K$$), RMSNorm performs:

1. Compute the sum of squares:

$$
s = \sum_{j=0}^{K-1} x_j^2
$$

2. Convert to mean square and take inverse square root:

$$
\mathrm{invRms} = \frac{1}{\sqrt{\frac{s}{K} + \epsilon}}
$$

3. Normalise each element (and optionally scale by $$g_i$$):

$$
y_i = x_i \cdot \mathrm{invRms} \cdot g_i
$$

If $$g_i$$ is not used, then it is simply:

$$
y_i = x_i \cdot \mathrm{invRms}
$$


## GPU mapping for per token RMSNorm

The CUDA mapping is the same pattern used for per token LayerNorm and per token softmax:

- **one block handles one token**
- threads in the block cover the $$K$$ features
- threads cooperate via a block reduction to compute the normalisation scalar
- each thread writes one output element

A typical kernel structure is:

1. Load one feature value per thread
2. Compute the square $$x_i^2$$
3. Block reduce to get $$s$$
4. Thread 0 computes $$\mathrm{invRms}$$ and stores it in shared memory
5. All threads normalise and store output


## What changes compared to LayerNorm

LayerNorm needs:

- one block reduction for $$\sum x$$ to compute $$\mu$$
- one block reduction for $$\sum (x - \mu)^2$$ to compute $$\sigma^2$$

RMSNorm needs:

- one block reduction for $$\sum x^2$$ only

So RMSNorm is the same reduction pattern but with fewer stages.


## Interview Q&A: RMSNorm vs LayerNorm

**Q: What does RMSNorm normalise over**

**A:** RMSNorm normalises per token across the feature dimension. For a tensor shaped $$(M, K)$$ it computes one normalisation scalar per row and applies it to the features in that row.

**Q: Why is RMSNorm cheaper than LayerNorm**

**A:** RMSNorm removes the mean subtraction and variance calculation. It needs only one reduction (sum of squares) rather than two reductions (sum and sum of squared residuals), which reduces FLOPs and synchronisation.

**Q: Does RMSNorm still need epsilon**

**A:** Yes. $$\epsilon$$ is added inside the square root to avoid division by zero and improve numerical stability when the RMS is very small.

**Q: What is the main takeaway**

**A:** RMSNorm is a simpler, more GPU friendly normalisation layer that is applied per token like LayerNorm, but it uses only the RMS magnitude rather than mean and variance, which makes it faster to compute and easier to implement.
