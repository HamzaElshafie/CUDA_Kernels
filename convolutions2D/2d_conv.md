# 2D convolution

# Naive 
For image processing and computer vision, input data is typically represented as 2D arrays, with pixels in an x–y space. Image convolutions are therefore **2D convolutions**, as illustrated in **Fig. 7.4 (Pic 1)**.

In a 2D convolution the filter `f` is also a 2D array. Its x and y dimensions determine the range of neighbours to be included in the weighted sum calculation. If we assume that the dimension of the filter is $(2r_x + 1)$ in the x dimension and $(2r_y + 1)$ in the y dimension, the calculation of each output element `P` can be expressed as:

![Image 3](https://github.com/user-attachments/assets/b408a7a7-9976-44b6-8f0e-6b180a79649b)


$$
P[y, x] =
\sum_{j=-r_y}^{r_y}
\sum_{i=-r_x}^{r_x}
N[y + j, x + i] \cdot f[j + r_y, i + r_x]
$$

**(Honestly, don'f focus too much on the equation you already know what it means conceptually)**

In Fig. 7.4 we use a $5 \times 5$ filter for simplicity; that is, $r_y = 2$ and $r_x = 2$. In general, the filter does not have to be, but is typically, a square array.

To generate an output element, we take the subarray whose centre is at the corresponding location in the input array `N`. We then perform pairwise multiplication between elements of the filter array and those of the image array. For our example the result is shown as the $5 \times 5$ product array below `N` and `P` in Fig. 7.4. The value of the output element is the sum of all elements of the product array.


## Worked example: computing $P_{2,2}$

The example in Fig. 7.4 shows the calculation of $P_{2,2}$.

For brevity, we use $N_{y,x}$ to denote `N[y][x]` in a C-style array. Since `N` and `P` are most likely dynamically allocated arrays, our actual code uses linearised indices, but the maths is easier to read in 2D form.

The calculation is:

$$
\begin{aligned}
P_{2,2} =\;&
N_{0,0}M_{0,0} + N_{0,1}M_{0,1} + N_{0,2}M_{0,2} + N_{0,3}M_{0,3} + N_{0,4}M_{0,4} \\
&+ N_{1,0}M_{1,0} + N_{1,1}M_{1,1} + N_{1,2}M_{1,2} + N_{1,3}M_{1,3} + N_{1,4}M_{1,4} \\
&+ N_{2,0}M_{2,0} + N_{2,1}M_{2,1} + N_{2,2}M_{2,2} + N_{2,3}M_{2,3} + N_{2,4}M_{2,4} \\
&+ N_{3,0}M_{3,0} + N_{3,1}M_{3,1} + N_{3,2}M_{3,2} + N_{3,3}M_{3,3} + N_{3,4}M_{3,4} \\
&+ N_{4,0}M_{4,0} + N_{4,1}M_{4,1} + N_{4,2}M_{4,2} + N_{4,3}M_{4,3} + N_{4,4}M_{4,4}
\end{aligned}
$$

In the numeric example shown in the figure, this corresponds to summing all elements of the $5 \times 5$ product array.


## Boundary conditions in 2D convolution

Like 1D convolution, 2D convolution must also deal with boundary conditions. With boundaries in both the x and y dimensions, the calculation of an output element may involve boundary conditions along a horizontal boundary, a vertical boundary, or both.

Fig. 7.5 illustrates the calculation of a `P` element that involves both boundaries.

From Fig. 7.5, the calculation of $P_{1,0}$ involves two missing columns and one missing row in the subarray of `N`. As in 1D convolution, different applications assume different default values for these missing `N` elements. In this example we assume the default value is `0`.

These boundary conditions also affect the efficiency of tiling. We will come back to this point soon.

![Image 4](https://github.com/user-attachments/assets/4db6ae54-c43f-4cd3-8437-cb953681e3de)

-----

# Constant memory and caching for 2D convolution

Consider the naive 2D convolution kernel:

```cpp
/**
 * Assuming filter is square, thats why I only calculate radius once, otherwise I would need radius_y and radius_x
 */
__global__ void naive_conv_2d(const float* x, const float* f, float* y, int M, int K, int f_h, int f_w) {
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = (f_w - 1) / 2;

    if (ty >= M || tx >= K) return;

    float res = 0.0f;

    for (int i = 0; i < f_h; i++){
        for (int j = 0; j < f_w; j++) {
            int out_row = ty - radius + i;
            int out_col = tx - radius + j;
            float val = (out_row >= 0 && out_row < M && out_col >=0 && out_col < K) ? x[out_row * K + out_col] : 0.0f;
            res += val * f[i * f_w + j];
        }
    }

    y[ty * K + tx] = res;
}
```

## Observation 1: Control flow divergence

Threads that compute output elements near the image boundaries must handle **ghost cells**.  
Because each output position near an edge has a different number of missing neighbors, threads in the same warp may take different paths through the boundary checks.

For example:
- the thread computing `P[0][0]` will skip most multiply-accumulate operations
- the thread computing `P[0][1]` will skip fewer
- interior threads will skip none

This leads to **control divergence** inside a warp.

In practice, this effect is usually modest:
- convolution is often applied to large images
- filters are typically small
- only a small fraction of threads are near the boundaries

As a result, divergence impacts only a small portion of the total work.


## Observation 2: Memory bandwidth is the real bottleneck

A much more serious issue is **global memory bandwidth**.

In the inner loop:
- each iteration performs **2 floating-point operations** (multiply + add)
- but loads **8 bytes** from global memory (`x` and `f`)

This yields a compute-to-memory ratio of approximately:

$$
0.25 \text{ FLOPs / byte}
$$

As seen previously in matrix multiplication, such a low arithmetic intensity means this kernel will run at only a small fraction of peak GPU performance.

To improve performance, we need to **reduce global memory traffic**.

The next two sections introduce two key techniques:
1. constant memory for the filter  
2. shared memory tiling with halo cells  


## Why the convolution filter is ideal for constant memory

The filter array `f` has three important properties:

### 1. Small size
Most convolution filters are small (e.g. 3×3, 5×5, 7×7).  
Even a 3D filter with radius 3 contains only:

$$
(2 \cdot 3 + 1)^3 = 343 \text{ elements}
$$

### 2. Read-only during kernel execution
The filter does not change while the kernel is running.

### 3. Uniform access pattern across threads
All threads:
- access the filter
- access the filter in the same order
- access the same filter indices at the same time inside the nested loops

These properties make the filter an excellent candidate for **constant memory**.

## Constant memory in CUDA

CUDA provides a special memory space called **constant memory**:
- visible to all thread blocks
- read-only during kernel execution
- limited in size
- aggressively cached by the hardware

Constant memory variables are declared as global variables using the `__constant__` qualifier and must be defined **outside any function**.

Unlike global memory pointers, constant memory variables:
- are not passed as kernel arguments
- are accessed directly by name inside the kernel

On the host side, data is copied into constant memory using a special API call that informs the CUDA runtime that the data will not change during kernel execution.


## Why constant memory is fast for convolution filters

Although constant memory is physically stored in DRAM, it has a **specialized cache**:
- optimized for read-only access
- no need to support writes
- much simpler and more energy-efficient than a general cache

The most important performance feature is **warp-level broadcast**:
- if all threads in a warp access the same constant memory address
- the value is fetched once
- and broadcast to all threads in the warp

This is exactly what happens in convolution:
- filter indices are independent of thread indices
- all threads access `f[i * f_w + j]` at the same time

Because the filter is small, it typically fits entirely in the constant cache.  
As a result, accesses to the filter incur **almost no DRAM traffic**.

## Effect on arithmetic intensity

With constant memory:
- filter loads effectively come from cache
- only the input image `x` contributes to global memory traffic

The arithmetic intensity improves to approximately:

$$
0.5 \text{ FLOPs / byte}
$$

This doubles the compute-to-memory ratio compared to the naive version.

While still memory-bound, this is a meaningful improvement and costs very little implementation effort.

----

# Shared memory tiling with halo cells for 2D convolution

Even after placing the filter in constant memory, the kernel is still **memory bound**. Each thread repeatedly loads overlapping input pixels from global memory. The next optimisation step is to reduce this redundancy using **shared memory tiling**.

### Core idea

Instead of every thread independently reading its full neighbourhood from global memory, all threads in a block:

1. **Collaboratively load a tile of the input image into shared memory**
2. Synchronise to ensure the tile is fully loaded
3. Compute output elements using the shared tile

This dramatically increases **data reuse** and reduces global memory traffic.

### Output tile vs input tile

In tiled convolution, each thread block computes a **tile of output pixels**.  
However, to compute those pixels, the block requires a **larger tile of input pixels** that includes halo cells.

If:

- Output tile size = $$T_{out,x} \times T_{out,y}$$  
- Filter radius = $$r$$

Then:

$$
\text{Input tile size} = (T_{out,x} + 2r) \times (T_{out,y} + 2r)
$$

The extra $$2r$$ in each dimension accounts for halo cells needed by output elements near tile edges.

**This is the main difference from tiled matrix multiplication**:
- GEMM input tiles and output tiles have the same size
- Convolution input tiles are larger due to halos

![Image 6](https://github.com/user-attachments/assets/bb5acdba-d25c-4a83-8dcf-8f37b1d791e5)

The central region corresponds to the output tile, while the surrounding border represents halo cells.

### Shared memory role

Each block allocates shared memory to store the **entire input tile**, including halos. Once loaded:

- All threads read input pixels from shared memory instead of global memory
- The same input pixels are reused by multiple threads
- Global memory is accessed only during tile loading and final output write

This is the same principle used in tiled matrix multiplication, but with additional indexing complexity due to halo regions.

### Thread organisation strategies

Because the input tile is larger than the output tile, thread mapping is less straightforward than GEMM.

There are two common approaches:

#### Strategy A — Block dimensions match input tile

- Each thread loads exactly one input pixel
- Loading logic is simple
- Some threads become idle during output computation
- Slightly less efficient use of compute resources

#### Strategy B — Block dimensions match output tile (more common)

- Each thread computes one output pixel
- No threads are idle during compute
- Threads must iterate to load the larger input tile
- Slightly more complex loading logic

Both strategies are valid; the second usually offers better resource utilisation.

### Why this improves performance

Without tiling:
- Each output pixel loads many input pixels from global memory
- Neighboring threads reload the same pixels

With tiling:
- Each input pixel is loaded once per block
- Reused by many threads from shared memory

This increases **arithmetic intensity** and reduces the global memory bottleneck.

### Handling boundaries and halo cells

At image boundaries:
- Some halo elements fall outside the image
- These are treated as ghost cells (usually zero)

The important optimisation detail:

Boundary checks are performed **only during tile loading**, not inside the inner convolution loops. This minimises control divergence during the heavy compute phase.

### Memory traffic evolution

| Version | Filter source | Input source during compute | Data reuse |
|--------|----------------|-----------------------------|------------|
| Naive | Global memory | Global memory | None |
| + Constant memory | Constant cache | Global memory | Filter reused |
| + Shared memory tiling | Constant cache | Shared memory | Filter and input reused |

Shared memory tiling removes most redundant global memory loads and is the key to high performance convolution kernels.

---

### Big picture

An optimised 2D convolution kernel uses:

- **Constant memory** for the filter  
- **Shared memory tiling** for the input  
- **Halo regions** to support boundary pixels  

This pattern is one of the most important stencil optimisation techniques in CUDA.
