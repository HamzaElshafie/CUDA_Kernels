#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float* A, float* y, int N) {
    // Find thread idx
    const int tid = threadIdx.x;
    // Which index of the input vector will this thread work on?
    int idx = blockIdx.x * NUM_THREADS + tid;
    // Find the lane of the thread within the warp
    int lane = tid % WARP_SIZE;
    // How many warps do we have?
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE; // CEIL division
    // Which warp am I in?
    int warp = tid / WARP_SIZE;

    // Allocate SMEM
    __shared__ float reduce_smem[NUM_WARPS];

    // Each thread loads its value. if the idx of the thread is out of bound (>N), load 0
    float sum = (idx < N) ? A[idx] : 0.0f;
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);

    if (lane == 0) {
        reduce_smem[warp] = sum;
    }
    __syncthreads();

    // Final reduction by a single warp over the per warp partial sums
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) {
        sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
    }

    // Combine block results across the grid, often with atomicAdd or a second kernel
    if (tid == 0) {
        atomicAdd(y, sum);
    }
}

