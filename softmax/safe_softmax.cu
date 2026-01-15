#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }

    return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }

    return val;
}

template <const int NUM_THREADS = 256>
__device__  __forceinline__ float block_reduce_sum_f32(float val) {
// Find the lane of the thread within the warp
int lane = threadIdx.x % WARP_SIZE;
// Which warp am I in?
int warp = threadIdx.x / WARP_SIZE;
// How many warps do we have?
constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE; // CEIL division

// Allocate SMEM
static __shared__ float reduce_smem[NUM_WARPS];

// Each thread loads its value. if the idx of the thread is out of bound (>N), load 0
float sum = warp_reduce_sum_f32<WARP_SIZE>(val);

if (lane == 0) {
    reduce_smem[warp] = sum;
}
__syncthreads();

sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
sum = warp_reduce_sum_f32<NUM_WARPS>(sum);

// Broadcast to all threads in warp
sum = __shfl_sync(0xffffffff, sum, 0);

return sum;
}

template <const int NUM_THREADS = 256>
__device__  __forceinline__ float block_reduce_max_f32(float val) {
// Find the lane of the thread within the warp
int lane = threadIdx.x % WARP_SIZE;
// Which warp am I in?
int warp = threadIdx.x / WARP_SIZE;
// How many warps do we have?
constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE; // CEIL division

// Allocate SMEM
static __shared__ float reduce_smem[NUM_WARPS];

// Each thread loads its value. if the idx of the thread is out of bound (>N), load 0
float max = warp_reduce_max_f32<WARP_SIZE>(val);

if (lane == 0) {
    reduce_smem[warp] = max;
}
__syncthreads();

max = (lane < NUM_WARPS) ? reduce_smem[lane] : -FLT_MAX;
max = warp_reduce_max_f32<NUM_WARPS>(max);

// Broadcast to all threads in warp
max = __shfl_sync(0xffffffff, max, 0);

return max;
}

/**
Softmax per token kernel

Steps:
1. Exponentiate each element
2. Compute sum of exponentials using reduction to get denominator
3. Divide each exponential by the denominator (normalise)
*/
template <int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token(float* A, float* C, int N) {
    // thread's global idx
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < N) ? A[idx] : -FLT_MAX;
    // Find max
    float max = block_reduce_max_f32<NUM_THREADS>(val);
    float exp_val = (idx < N) ? expf(val-max) : 0.0f;
    // Warp reduce on expVal to get denominator
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);

    // Normalise all values and store back 
    if (idx < N) {
        C[idx] = exp_val / exp_sum;
    }
}
