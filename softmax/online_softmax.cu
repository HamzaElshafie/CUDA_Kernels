#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

struct MD {
    float m;
    float d;
};

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_MD(MD value) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(0xffffffff, value.m, mask);
        other.d = __shfl_xor_sync(0xffffffff, value.d, mask);

        // Which is bigger to decide which to scale?
        bool value_bigger = (value.m > other.m);
        MD bigger_m = value_bigger ? value : other;
        MD smaller_m = value_bigger ? other : value;

        value.d = smaller_m.d * expf(smaller_m.m - bigger_m.m) + bigger_m.d;
        value.m = bigger_m.m;
    }

    return value;
}


/**
Online Softmax per token kernel

Steps:
1. Initialise each thread's MD
2. Calculate number of warps the block has 
3. Allocate shared memory (one per warp)
4. Perform warp reduce on the MD structure within warp
5. Store the returned MD structure of each warp in SMEM
6. Perform another warp reduction on the SMEM stored MDs
7. Scale each element with the global identified MD 
*/
template <const int NUM_THREADS = 256>
__global__ void online_softmax_f32_per_token_kernel(const float* x, float* y, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_idx = threadIdx.x / WARP_SIZE;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    MD val;
    val.m = (tid < N) ? x[tid] : -FLT_MAX;
    val.d = (tid < N) ? 1.0f : 0.0f;
    __shared__ MD reduce_smem[NUM_WARPS];

    MD local_md = warp_reduce_MD<WARP_SIZE>(val);

    if (lane == 0) {
        reduce_smem[warp_idx] = local_md;
    }
    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        MD block_val = (threadIdx.x < NUM_WARPS) ? reduce_smem[threadIdx.x] : MD{-FLT_MAX, 0.0f};
        MD global_md = warp_reduce_MD<NUM_WARPS>(block_val);
        if (threadIdx.x == 0) {
            reduce_smem[0] = global_md;
        }
    }
    __syncthreads();

    MD final_md =  reduce_smem[0];
    if (tid < N) {
        y[tid] = expf(x[tid] - final_md.m) / final_md.d;
    }
}