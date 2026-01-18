#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

/**
Warp sum reduction helper for f32

Steps:
1. We loop over a mask which divides in half each time starting from kWarpSize / 2
2. Perform an XOR shuffle sync with the current val and mask
3. Add the result from the obtained val after XOR shuffle to current argument value
4. Return val after loop ends


****** __forceinline__ strongly hints to the compiler to inline this function. ******

Why do we want force inline here?
- This function is very small and performance critical.
- Inlining removes function call overhead.
- It allows better instruction scheduling and register allocation across the caller and callee.

****** #pragma unroll tells the compiler to fully unroll this loop at compile time. ******

Why do we want this?
- The loop has a fixed, small number of iterations (log2(kWarpSize), e.g. 5 for 32).
- Unrolling removes loop control overhead (branch, counter, compare).
- It allows the compiler to schedule shuffle and add instructions more efficiently.

Why might we NOT want to unroll?
- If the loop trip count were large or dynamic, unrolling would increase register pressure and code size.
- Here it is safe and desirable because kWarpSize is compile-time constant and small.
*/
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_f32(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

/** 
Dot product kernel f32

Steps:
1. Get the global index of the thread which will determine which element it will access 
2. Calculate NUM_WARPS and allocate SMEM
3. Loop over the shared dimension "N" (No looping tho, paralellism!!!)
4. During each iteration we will get product of a[i] & b[i] and pass this to the warp reduction kernel
5. Each warp's leader stores in the smem region allocated to it
6. We do another warp reduce on those (effectively this is a block reduce)
7. Store result back in c
*/
template <const int NUM_THREADS = 256>
__global__ void dot_product_f32(float* a, float* b, float* c, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_prod[NUM_WARPS];

    float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
    prod = warp_reduce_f32<WARP_SIZE>(prod);

    if (lane == 0) {
        reduce_prod[warp] = prod;
    }
    __syncthreads();

    prod = (lane < NUM_WARPS) ? reduce_prod[lane] : 0.0f;
    
    if (warp == 0) {
        prod = warp_reduce_f32<NUM_WARPS>(prod);
    }

    if (tid == 0) {
        atomicAdd(c, prod);
    }
}