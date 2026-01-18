# Dot product on the GPU (FP32) and how a reduction kernel implements it

## What is a dot product

Given two vectors:

- $$a \in \mathbb{R}^{N}$$
- $$b \in \mathbb{R}^{N}$$

their dot product is a single scalar:

$$
y = a \cdot b = \sum_{i=0}^{N-1} a_i b_i
$$

You can think of this as doing two operations:

1. **Elementwise multiply**: compute $$p_i = a_i b_i$$
2. **Reduce (sum)**: compute $$y = \sum_i p_i$$

So dot product is a classic example of a *map then reduce* pattern.

## Why dot product needs a reduction kernel on the GPU

On the CPU, a dot product is often just a loop:

```cpp
float y = 0.0f;
for (int i = 0; i < N; ++i) y += a[i] * b[i];