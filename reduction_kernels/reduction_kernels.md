![sum_reduce excalidraw](https://github.com/user-attachments/assets/07ee5539-cc36-42f6-af04-4a11e7733734)# Reduction kernels: from warp reductions to block reductions

A **reduction kernel** is a GPU kernel whose job is to combine many input elements into a smaller number of output elements using an associative operator such as sum, max, or min. The simplest example is summation: given an array `a[0..N-1]`, compute a single value `y = sum(a)`. Reductions appear everywhere in ML systems and numerical computing: computing losses, normalisation statistics, softmax denominators, dot products, gradient accumulation, and many other operations all boil down to reductions.

The reason reductions deserve special attention is that they are not like elementwise kernels where each thread writes to a unique output. In a reduction, many threads ultimately contribute to the same output, which introduces two performance challenges. First, we need an efficient way to combine values within a block without repeatedly going to slow global memory. Second, if the final output is shared across blocks, we need a correct way to combine partial results produced by different blocks without data races.

A common high performance pattern is therefore a **hierarchical reduction**. Instead of trying to sum `N` values in one step, we reduce in stages: threads compute local partial sums, then warps reduce within themselves, then blocks reduce across their warps, and finally the grid combines block results. This hierarchy matches the GPU execution model and lets us use fast communication methods at each level.

## Two key ideas behind fast reduction kernels

The first idea is to keep work local and fast for as long as possible. Each thread starts by loading one or more elements and accumulating them in a register. Registers are the fastest storage on the GPU, so it is ideal to do as much accumulation as possible before communicating.

The second idea is to use the right communication mechanism at each level. Within a warp, threads can exchange register values directly using warp shuffle functions, which avoids shared memory and synchronisation. Across warps within the same block, threads must use shared memory plus `__syncthreads()` because warps do not automatically synchronise with each other. Across blocks, there is no shared memory, so combining results usually requires either multiple kernel launches or global atomics.

## The general block reduction structure

A very common reduction kernel structure looks like this:

1. **Per thread input and local accumulation**
2. **Warp level reduction** to produce one partial sum per warp
3. **Write warp partial sums to shared memory**
4. **Synchronise the block**
5. **Final reduction by a single warp** over the per warp partial sums
6. **Combine block results across the grid**, often with `atomicAdd` or a second kernel

This structure is a practical implementation of the reduction tree you often see drawn in diagrams. The diagram shows the logical algorithm: many values are combined into fewer values layer by layer until one value remains. The GPU implementation follows the same tree, but chooses faster primitives depending on where the data needs to move.

<img width="1043" height="785" alt="sum_reduce excalidraw" src="https://github.com/user-attachments/assets/0724419f-057b-41de-b66b-a86c75759bc1" />


## Stage 1: each thread contributes a value

A reduction kernel typically assigns each thread a global index `idx` and loads one input element:

```cpp
float sum = (idx < N) ? a[idx] : 0.0f;
```

## Stage 2: reduce within each warp using shuffles

Threads in a block are executed in groups of `32` called warps, so the first natural reduction step is to reduce values **within each warp**. At this point, each thread holds a partial sum in a register, and no communication across warps is required yet.

This reduction is performed using warp shuffle functions, which allow threads in the same warp to exchange register values directly without using shared memory. Because all threads in a warp execute in lockstep, these exchanges are fast and do not require explicit synchronisation.

Conceptually, a warp reduction applies a tree reduction pattern over the `32` lanes of the warp. Each lane starts with its own value, then repeatedly exchanges and adds values with a partner lane determined by an XOR mask. After a fixed number of steps, every lane holds the sum of all values in the warp.

In code, this step typically looks like:

```cpp
sum = warp_reduce_sum_f32(sum);
```

After this call, all lanes in the warp contain the same value: the sum of the contributions from the 32 threads in that warp. Although the value is replicated across all lanes, this is not wasteful, because only one lane per warp will later write the result to shared memory.

Reducing within a warp first is important for performance. It keeps the data in registers, avoids shared memory traffic, and avoids the need for synchronisation. By the end of this stage, a block with NUM_THREADS threads has effectively reduced its data down to NUM_WARPS = NUM_THREADS / 32 partial sums, one per warp, which can then be combined in the next stage.

## Stage 3: write warp partial sums to shared memory

After the warp level reduction, each warp has produced a single partial sum representing the combined contribution of its `32` threads. Although this value is replicated in every lane of the warp, only one copy is needed for the next stage of the reduction.

To pass these warp level results to the rest of the block, a single designated thread per warp, usually the warp leader (lane `0`), writes the warp sum to shared memory:

```cpp
if (lane == 0) {
    reduce_smem[warp] = sum;
}
```

Here, warp is the warp index within the block, and reduce_smem is a shared memory array sized to hold one value per warp. Each warp writes to a unique location in this array, so there are no write conflicts.

Shared memory is used at this point because warp shuffle operations cannot communicate across different warps. Shared memory acts as the handoff mechanism between warps, allowing the partial results produced by each warp to be collected in one place.

By the end of this stage, the block has reduced its original NUM_THREADS values down to NUM_WARPS values stored in shared memory, where NUM_WARPS is the number of warps in the block. These values will be combined in the next stage to produce a single block level result.

## Stage 4: synchronise the block

After stage 3, each warp leader has written its warp partial sum into shared memory. However, warps within a block do not automatically stay in sync with each other. Some warps may reach the next stage earlier than others, which means a warp could start reading `reduce_smem` before every warp has finished writing its value.

To prevent this race, the kernel inserts a block wide barrier:

```cpp
__syncthreads();
```
This call guarantees two things for all threads in the block:
1. Execution barrier: no thread continues past this point until all non exited threads in the block have reached the barrier.
2. Shared memory visibility: all writes to shared memory made before the barrier are visible to all threads after the barrier.

This synchronisation is required specifically because we are now communicating across warps using shared memory. Warp shuffle operations do not require __syncthreads() because warp lanes execute in lockstep, but shared memory communication across warps does.

By the end of this stage, it is safe for the next stage to read the per warp sums from reduce_smem and perform the final block level reduction.

## Stage 5: the first warp reduces the warp sums into a block sum

After the block synchronisation, shared memory contains `NUM_WARPS` partial sums, one produced by each warp. The goal of this stage is to combine these `NUM_WARPS` values into a single **block sum**.

A common approach is to reuse the warp reduction routine again, but this time only the **first warp** performs the reduction. This keeps the final reduction fast, because it is still implemented using shuffle operations.

First, lanes in warp `0` load values from shared memory. Each lane loads one warp sum if it exists; lanes beyond `NUM_WARPS` load `0.0f` so they do not affect the result:

```cpp
sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
```

Then, only warp 0 performs a warp reduction over these loaded values:

```cpp
if (warp == 0) {
    sum = warp_reduce_sum_f32(sum);
}
```

Logically, this corresponds to another layer of the reduction tree. The block has already reduced NUM_THREADS values down to NUM_WARPS values, and now it reduces those NUM_WARPS values down to 1. Practically, this is efficient because only a single warp performs the work and the reduction itself stays in registers using shuffles.

By the end of this stage, warp 0 holds the final block sum. As with previous warp reductions, the sum is replicated across the lanes of warp 0, and the next stage will typically choose a single thread to write or accumulate it.

## Stage 6: combine block results across the grid

After stage 5, each block has computed a single value: the **block sum** for the chunk of the input that block processed. If the goal is a single global result such as `y = sum(a)`, then the block sums from all blocks must be combined into one final value.

Since blocks run independently and can finish in any order, they cannot safely write to the same global output location without coordination. A simple and correct way to combine block sums is to have one thread per block perform an atomic update:

```cpp
if (tid == 0) {
    atomicAdd(y, sum);
}
```

Here, tid == 0 ensures exactly one thread per block performs the update, avoiding unnecessary contention. atomicAdd ensures correctness by making the read–modify–write sequence on y atomic, so multiple blocks can accumulate into y without data races.

This approach is straightforward and works well when the number of blocks is not huge. However, when there are many blocks, the atomic can become a bottleneck because all blocks contend for the same memory location. In high performance implementations, a common alternative is a two-pass reduction: first write one block sum per block into an intermediate array, then launch a second kernel to reduce that array. Both approaches compute the same logical result; they differ only in how they trade simplicity for scalability.
