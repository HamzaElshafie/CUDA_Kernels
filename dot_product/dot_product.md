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
```

On the GPU, thousands of threads can multiply different elements in parallel, but they must still **cooperate** to produce a single scalar result. That cooperation is the *reduction* part.

A dot product kernel therefore usually follows this structure:

- each thread computes a local partial product
- threads reduce those partial products into a block sum
- blocks combine their partial sums into the final answer, often using `atomicAdd` or a second kernel


## GPU mapping for a basic dot product kernel

A common mapping is:

- one CUDA block handles a contiguous chunk of the vectors
- each thread loads one element from `a` and `b`, multiplies them, and produces one partial product
- the block reduces all partial products into a single block sum
- one thread writes or accumulates that block sum into the global output

If we choose `NUM_THREADS = 256`:

- block `0` handles indices `0..255`
- block `1` handles indices `256..511`
- and so on

The global element index owned by a thread is:

$$
\mathrm{idx} = \mathrm{blockIdx.x} \cdot \mathrm{NUM\_THREADS} + \mathrm{threadIdx.x}
$$

Each thread computes:

$$
p =
\begin{cases}
a_{\mathrm{idx}} \cdot b_{\mathrm{idx}} & \mathrm{idx} < N \\
0 & \text{otherwise}
\end{cases}
$$

The `0` for out of bounds threads keeps the reduction valid without extra branching.


## Logic steps of a dot product reduction kernel

Conceptually, the kernel does:

1. **Load and multiply**

$$
p \leftarrow a_{\mathrm{idx}} \cdot b_{\mathrm{idx}}
$$

2. **Reduce inside each warp**  
   Reduce the 32 values in a warp into a single warp sum using a warp reduction.

3. **Store warp sums to shared memory**  
   The warp leader writes its warp sum to shared memory.

4. **Reduce warp sums to a block sum**  
   The first warp loads the warp sums and reduces them to one final block sum.

5. **Accumulate into the global result**  
   One thread performs:

$$
y \leftarrow y + \mathrm{blockSum}
$$

typically using `atomicAdd` if many blocks contribute.

This is the same reduction pattern we used before for block sum reductions, except the inputs are products instead of raw values.


## What makes this efficient

- The multiply happens entirely in registers.
- The first stage reduction uses warp shuffles, avoiding shared memory.
- Shared memory is only used for the small “warp sums” array (for 256 threads, that is 8 floats).
- Only one atomic operation per block is used, not one per element.

## Notes for interview discussion

**Q: Is this one global read or multiple passes?**  
Each thread reads `a[idx]` and `b[idx]` exactly once. All reduction work is done using registers and shared memory. The only global write is the final atomic add per block.

**Q: Why use an atomic at the end?**  
Because multiple blocks each compute a partial dot product, and they must combine into a single global scalar. Without a second kernel launch, an atomic is the simplest correct way to combine them.

**Q: What would you do if atomics were too slow?**  
You could write per block sums into a separate output array and launch a second kernel, or use a hierarchical reduction, to reduce those block sums into one final value.
