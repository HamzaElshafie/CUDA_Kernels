# 1D Convolution on the GPU

## What is 1D convolution

A 1D convolution produces an output array `y` where each output element is a weighted sum of nearby input elements from `x`.

You have:

- Input array:  
  $$x = [x_0, x_1, \dots, x_{n-1}]$$

- Filter (convolution filter) of odd length:  
  $$f = [f_0, f_1, \dots, f_{2r}]$$

Here, `r` is the **filter radius**, so the filter length is:

$$
F = 2r + 1
$$

The convolution output is:

$$
y_i = \sum_{j=0}^{2r} f_j \cdot x_{i - r + j}
$$

Equivalent form (often easier to reason about):

$$
y_i = \sum_{k=-r}^{r} f_{k+r} \cdot x_{i+k}
$$

Interpretation:
- For each output position `i`, you centre the filter at `x[i]`
- You multiply neighbouring elements by filter weights
- You sum the products to produce `y[i]`

---

## Example

Let:

$$
x = [8, 2, 5, 4, 1, 7, 3]
$$

Filter radius:

$$
r = 2
$$

Filter:

$$
f = [1, 3, 5, 3, 1]
$$

Then for `i = 2`:

$$
y_2 = 1\cdot x_0 + 3\cdot x_1 + 5\cdot x_2 + 3\cdot x_3 + 1\cdot x_4
$$

$$
y_2 = 1\cdot 8 + 3\cdot 2 + 5\cdot 5 + 3\cdot 4 + 1\cdot 1 = 52
$$

This is the “inner product” view:
- take a window of `x` of length `2r+1`
- dot it with the filter `f`

---

## Boundary conditions (ghost cells)

Near the ends of the array, the filter window goes out of bounds.

For example, for `i = 1` and `r = 2`, you would need:

$$
x_{-1}
$$

which does not exist.

A common convention is **zero padding**, meaning:

$$
x_k = 0 \quad \text{for any } k < 0 \text{ or } k \ge n
$$

So the formula becomes:

$$
y_i = \sum_{k=-r}^{r} f_{k+r}\cdot \tilde{x}_{i+k}
$$

where:

$$
\tilde{x}_{t} =
\begin{cases}
x_t & 0 \le t < n \\
0 & \text{otherwise}
\end{cases}
$$

Other applications may use different padding rules, e.g. clamp to edge, reflect, wrap, etc, but for interview and most CUDA examples, zero padding is standard.

---

## GPU mapping for 1D convolution

### Goal

Compute all `y[i]` in parallel.

### Natural mapping

- One thread computes one output element `y[i]`
- That thread loads the required neighbourhood of `x`
- It multiplies by filter weights and accumulates a sum

Thread `i` computes:

$$
y_i = \sum_{k=-r}^{r} f_{k+r}\cdot \tilde{x}_{i+k}
$$

### Thread index

In CUDA 1D launch:

$$
i = \mathrm{blockIdx.x}\cdot \mathrm{blockDim.x} + \mathrm{threadIdx.x}
$$

So each thread does:

- if `i < n`, compute `y[i]`
- otherwise do nothing

---

## Logic steps of a naive 1D convolution kernel

For each output element `i`:

1. Compute global index `i`
2. Initialise accumulator:

$$
\text{acc} \leftarrow 0
$$

3. For each filter tap `k` from `-r` to `r`:
   - compute input index:

$$
t = i + k
$$

   - if `t` is in bounds, contribute:

$$
\text{acc} \leftarrow \text{acc} + f_{k+r}\cdot x_t
$$

   - otherwise treat `x_t = 0` and contribute nothing

4. Write output:

$$
y_i \leftarrow \text{acc}
$$

---

## What matters for performance (preview)

The naive version is correct but can be inefficient because:
- many threads read overlapping regions of `x`
- filter values are reused but may be read repeatedly
- global memory traffic can dominate

In optimised CUDA versions, we usually:
- cache a tile of `x` in shared memory (including ghost cells)
- put the filter in constant memory if it is small and fixed
- reduce redundant global reads

We will implement the naive kernel first, then optimise.

---