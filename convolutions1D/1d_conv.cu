#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstddef>
#include <iostream>

#include "../utils.h"

__global__ void naive_conv_1d(const float* x, const float* f, float* y, int f_w, int N) {
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = (f_w - 1) / 2;
    float res = 0.0f;

    if (t_idx >= N) return;

    for (int i = 0; i < f_w; i++) {
        int global_idx = t_idx - radius + i;
        float val = (global_idx >= 0 && global_idx < N) ? x[global_idx] : 0.0f;
        res += val * f[i];
    }

    y[t_idx] = res;
}

void conv1d_cpu_ref(const float* x, const float* f, float* y, int N, int f_w) {
    const int radius = (f_w - 1) / 2;

    for (int t = 0; t < N; ++t) {
        float acc = 0.0f;

        for (int i = 0; i < f_w; ++i) {
            const int xi = t - radius + i;              // input index
            const float xv = (xi >= 0 && xi < N) ? x[xi] : 0.0f;
            acc += xv * f[i];
        }

        y[t] = acc;
    }
}

int main() {
    int N = 1 << 10;
    int f_w = 5;

    // Calculate size of 1D vector
    size_t size_x = N * sizeof(float);
    // Calculate size of 1D convolution filter
    size_t size_f = f_w * sizeof(float);
    size_t size_y = N * sizeof(float);

    // Create host pointers and allocate host memory
    float* X_host = (float*)malloc(size_x);
    float* F_host = (float*)malloc(size_f);
    float* Y_host_cpu = (float*)malloc(size_y);
    float* Y_host_gpu = (float*)malloc(size_y);

    // Initialise the arrays 
    float* array_X[] {X_host};
    float* array_F[] {F_host};
    initialiseArrays(array_X, 1, N, 0.0f, 100.0f, 0);
    initialiseArrays(array_F, 1, f_w, 1.0f, 50.0f, 0);

    // Measure CPU reference time
    float cpu_time = measureExecutionTime([&](){
        conv1d_cpu_ref(X_host, F_host, Y_host_cpu, N, f_w);
    });

    // Create device ptrs and allocate device memory
    float* X_device;
    float* F_device; 
    float* Y_device;

    cudaMalloc((void**)&X_device, size_x);
    cudaMalloc((void**)&F_device, size_f);
    cudaMalloc((void**)&Y_device, size_y);

    // Copy arrays to device memory
    cudaMemcpy(X_device, X_host, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(F_device, F_host, size_f, cudaMemcpyHostToDevice);

    // Configure grid and block dims
    int num_blocks = (N + 32 - 1 ) / 32;
    dim3 blockDim(32);
    dim3 gridDim(num_blocks);

    // Execute kernel & measure time
    double gpu_time = measureKernelTime([&]() {
        naive_conv_1d<<<gridDim, blockDim>>>(X_device, F_device, Y_device, f_w, N);
        cudaDeviceSynchronize();
    });

    // Copy results back to host
    cudaMemcpy(Y_host_gpu, Y_device, size_y, cudaMemcpyDeviceToHost);

    // Compare against cpu reference for correctness
    bool result_match = compareResults(Y_host_cpu, Y_host_gpu, N, 1e-4, 1e-5);
    std::cout << (result_match ? "Results match!" : "Results do not match!") << std::endl;

    // Free memories
    free(X_host);
    free(F_host);
    free(Y_host_cpu);
    free(Y_host_gpu);

    cudaFree(X_device);
    cudaFree(F_device);
    cudaFree(Y_device);
    
    return 0;
}
