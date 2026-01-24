%%writefile sgemv.cu
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>
#include "utils.h"

#define WARP_SIZE 32

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int mask = kWarpSize >>1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

/**
grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
*/
template <const int NUM_THREADS=256>
__global__ void sgemv(const float* a, const float* x, float* y, int M, int K) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int block_idx = blockIdx.x;
    int NUM_WARPS_K = (K + WARP_SIZE - 1) / WARP_SIZE;
    
    int lane = tx % WARP_SIZE;
    int m = block_idx * blockDim.y + ty;
    float sum = 0.0f;

    if (m < M) {
        for (int w = 0; w < NUM_WARPS_K; w++) {
            int k = w * WARP_SIZE + lane;
            sum += a[m * K + k] * x[k];
        }
    }

    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0) {
        y[m] = sum;
    }
}

// CPU reference for SGEMV: y = A * x
// A: MxK row-major
// x: length K
// y: length M
void sgemv_cpu_ref(const float* A, const float* x, float* y, int M, int K) {
    for (int m = 0; m < M; ++m) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[m * K + k] * x[k];
        }
        y[m] = acc;
    }
}

int main() {
    int M = 1 << 10;
    int K = 1 << 10;

    // Calculate sizes
    size_t size_a = M * K * sizeof(float);
    size_t size_x = K * sizeof(float);
    size_t size_y = M * sizeof(float);

    // Create host pointers and allocate host memory
    float* A_host = (float*)malloc(size_a);
    float* x_host = (float*)malloc(size_x);
    float* y_host_cpu = (float*)malloc(size_y);
    float* y_host_gpu = (float*)malloc(size_y);

    // Initialise the arrays 
    float* array_A[] {A_host};
    float* array_x[] {x_host};
    initialiseArrays(array_A, 1, M * K, 0.0f, 1.0f, 0);
    initialiseArrays(array_x, 1, K, 0.0f, 1.0f, 0);

    // Measure CPU reference time
    float cpu_time = measureExecutionTime([&](){
        sgemv_cpu_ref(A_host, x_host, y_host_cpu, M, K);
    });

    // Create device ptrs and allocate device memory
    float* A_device;
    float* x_device; 
    float* y_device;

    CUDA_CHECK(cudaMalloc((void**)&A_device, size_a));
    CUDA_CHECK(cudaMalloc((void**)&x_device, size_x));
    CUDA_CHECK(cudaMalloc((void**)&y_device, size_y));

    // Copy arrays to device memory
    CUDA_CHECK(cudaMemcpy(A_device, A_host, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_device, x_host, size_x, cudaMemcpyHostToDevice));

    // Configure grid and block dims
    dim3 blockDim(32, 4);
    int num_blocks = (M + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(num_blocks);

    // Execute kernel & measure time
    double gpu_time = measureKernelTime([&]() {
        sgemv<<<gridDim, blockDim>>>(A_device, x_device, y_device, M, K);
        CUDA_CHECK(cudaDeviceSynchronize());
    });

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(y_host_gpu, y_device, size_y, cudaMemcpyDeviceToHost));

    // Compare against cpu reference for correctness
    bool result_match = compareResults(y_host_cpu, y_host_gpu, M, 1e-4, 1e-5);
    std::cout << (result_match ? "Results match!" : "Results do not match!") << std::endl;
    std::cout << "CPU time (ms): " << cpu_time << std::endl;
    std::cout << "GPU time (ms): " << gpu_time << std::endl;

    // Free memories
    free(A_host);
    free(x_host);
    free(y_host_cpu);
    free(y_host_gpu);

    CUDA_CHECK(cudaFree(A_device));
    CUDA_CHECK(cudaFree(x_device));
    CUDA_CHECK(cudaFree(y_device));
    
    return 0;
}
