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
1. We loop over a mask which divides in half each time starting from WARP_SIZE / 2
2. Perform an XOR shuffle sync with the current val and mask
3. Add the result from the obtained val after XOR shuffle to current argument value
4. Return val after loop ends
*/
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int mask = kWarpSize >>1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

/**
Block sum reduction helper for f32

Steps:
1. Perform warp sum reduction for each warp at first
2. Initialise a SMEM with [NUM_WARPS]
3. Warp leaders store their sum in the shared memory idx it belongs to
4. Perform another warp sum reduction over what was stored in SMEM
5. Broadcast the global sum obtained to all threads
6. Return sum
*/
template <const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp = tid / WARP_SIZE;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    static __shared__ float reduce_smem[NUM_WARPS];

    float sum = warp_reduce_sum_f32<WARP_SIZE>(val);

    if (lane == 0) {
        reduce_smem[warp] = sum;
    }
    __syncthreads();

    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    float global_sum = warp_reduce_sum_f32<NUM_WARPS>(sum);

    return global_sum;
}

/**
RMS normalisation kernel

Steps:
1. 
*/
template <const int NUM_THREADS>
__global__ void rms_norm_f32(float* x, float* y, float eps, int M, int K) {
    // TODO
}

