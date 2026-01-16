# Layer Normalisation: definition, motivation, and GPU perspective

## What is layer normalisation

Layer Normalisation (LayerNorm) is a normalisation technique that standardises the activations of a neural network **across the feature dimension**, independently for each sample.

Given an input vector of length `N`:

$$
x = (x_0, x_1, \dots, x_{N-1})
$$

LayerNorm computes:

$$
\mu = \frac{1}{N} \sum_{i=0}^{N-1} x_i
$$

$$
\sigma^2 = \frac{1}{N} \sum_{i=0}^{N-1} (x_i - \mu)^2
$$

and produces the normalised output:

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

Optionally, a learned affine transform is applied:

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

where:
- $$\gamma$$ is a learned scale vector
- $$\beta$$ is a learned bias vector
- $$\epsilon$$ is a small constant for numerical stability

---

## Why layer normalisation exists

Before LayerNorm, **Batch Normalisation** was widely used. BatchNorm normalises activations across the batch dimension, which works well in CNNs but has serious drawbacks for sequence models:

- It depends on batch size
- It behaves poorly with small or variable batch sizes
- It introduces dependency between samples in the same batch

LayerNorm solves these issues by normalising **within each sample**, across features. This makes it:

- independent of batch size
- stable for variable length sequences
- well suited for RNNs and Transformers

This is why **early Transformers used LayerNorm extensively**, including:
- original Transformer (Vaswani et al.)
- BERT
- GPT-2

## Logic steps of layer normalisation

For one input vector of length `N`, LayerNorm conceptually performs:

1. Compute the mean:
   
$$
\mu = \frac{1}{N} \sum_{i=0}^{N-1} x_i
$$

2. Compute the variance:
   
$$
\sigma^2 = \frac{1}{N} \sum_{i=0}^{N-1} (x_i - \mu)^2
$$

3. Compute the inverse standard deviation:
   
$$
\text{invStd} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}
$$

4. Normalise each element:
   
$$
\hat{x}_i = (x_i - \mu) \cdot \text{invStd}
$$

5. Optionally apply affine transform:
   
$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

This is a **two reduction problem**: one reduction for the mean, one for the variance.

---

## GPU mapping for per token LayerNorm

For Transformer style workloads, LayerNorm is usually applied **per token**, meaning:

- Each token corresponds to one vector of length `N` (hidden size)
- One CUDA block typically handles one token
- Threads cooperate to compute statistics over the feature dimension

A typical GPU mapping is:

- **One block per token**
- **Threads cover the feature dimension**
- Block reductions compute:
  - sum of values → mean
  - sum of squared differences → variance
- Each thread normalises its own element

Conceptually:

1. Each thread loads one (or more) elements of the token vector.
2. Reduce across the block to compute $$\mu$$.
3. Reduce across the block to compute $$\sigma^2$$.
4. Broadcast $$\mu$$ and $$\text{invStd}$$ to all threads.
5. Each thread computes its output element.

---

## Numerical stability considerations

LayerNorm uses an $$\epsilon$$ term inside the square root:

$$
\sqrt{\sigma^2 + \epsilon}
$$

This prevents division by zero and stabilises gradients when variance is very small.

Unlike softmax, LayerNorm does **not** require exponentials, so overflow is less of a concern. However, precision still matters when summing large vectors, especially for large hidden sizes.

---

## Interview perspective

**Q: Why is LayerNorm better than BatchNorm for Transformers?**

**A:** Because LayerNorm normalises across features within each sample and does not depend on batch statistics. This makes it stable for variable sequence lengths, small batch sizes, and autoregressive generation.

**Q: How many reductions does LayerNorm require?**

**A:** Two reductions per token: one to compute the mean and one to compute the variance.

**Q: Is LayerNorm memory bound or compute bound?**

**A:** Usually memory bound. Each element is read once and written once, with relatively little arithmetic compared to memory traffic. Optimised implementations focus on fusing operations and reducing memory accesses.

---

## Main takeaway

Layer Normalisation standardises activations per token by removing mean and scaling by variance. On the GPU, it is naturally expressed as a per block operation with two reductions. Understanding its reduction structure and memory access pattern is essential for implementing efficient CUDA kernels and for reasoning about performance in Transformer models.
