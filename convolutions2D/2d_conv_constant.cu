
%%writefile conv_2d_constant.cu
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
#include "utils.h"

#define FILTER_RADIUS 2
#define FILTER_H (2 * FILTER_RADIUS + 1)
#define FILTER_W (2 * FILTER_RADIUS + 1)
#define FILTER_ELEMENTS (FILTER_H * FILTER_W)
__constant__ float c_filter[FILTER_ELEMENTS];

/**
 * Assuming filter is square, thats why I only calculate radius once, otherwise I would need radius_y and radius_x
 */
__global__ void constant_conv_2d(const float* x, float* y, int M, int K) {
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ty >= M || tx >= K) return;

    float res = 0.0f;

    for (int i = 0; i < FILTER_H; i++){
        for (int j = 0; j < FILTER_W; j++) {
            int out_row = ty - FILTER_RADIUS + i;
            int out_col = tx - FILTER_RADIUS + j;
            float val = (out_row >= 0 && out_row < M && out_col >=0 && out_col < K) ? x[out_row * K + out_col] : 0.0f;
            res += val * c_filter[i * FILTER_W + j];
        }
    }

    y[ty * K + tx] = res;
}

// Naive CPU reference: 2D convolution with zero padding.
// Input  : in  (H x W) row-major
// Filter : filt (FH x FW) row-major (typically odd dims)
// Output : out (H x W) row-major
//
// out[y, x] = sum_{j=0..FH-1} sum_{i=0..FW-1}
//               in[y + (j-ry), x + (i-rx)] * filt[j, i]
// where ry = FH/2, rx = FW/2, and out-of-bounds in[] reads are treated as 0.
void conv2d_cpu_ref_zero_pad(const float* in,
                             const float* filt,
                             float* out,
                             int H, int W,
                             int FH, int FW)
{
    const int ry = FH / 2;
    const int rx = FW / 2;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;

            for (int j = 0; j < FH; ++j) {
                for (int i = 0; i < FW; ++i) {
                    const int in_y = y + (j - ry);
                    const int in_x = x + (i - rx);

                    const float v =
                        (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
                            ? in[in_y * W + in_x]
                            : 0.0f;

                    acc += v * filt[j * FW + i];
                }
            }

            out[y * W + x] = acc;
        }
    }
}

int main() {
    int M = 1 << 10;
    int K = 1 << 10;
    int f_h = 5;
    int f_w = 5;

    // Calculate size of 1D vector
    size_t size_x = M * K * sizeof(float);
    // Calculate size of 1D convolution filter
    size_t size_f = f_h * f_w * sizeof(float);
    size_t size_y = M * K * sizeof(float);

    // Create host pointers and allocate host memory
    float* X_host = (float*)malloc(size_x);
    float* F_host = (float*)malloc(size_f);
    float* Y_host_cpu = (float*)malloc(size_y);
    float* Y_host_gpu = (float*)malloc(size_y);

    // Initialise the arrays 
    float* array_X[] {X_host};
    float* array_F[] {F_host};
    initialiseArrays(array_X, 1, M * K, 0.0f, 100.0f, 0);
    initialiseArrays(array_F, 1, f_h * f_w, 1.0f, 50.0f, 0);

    // Measure CPU reference time
    float cpu_time = measureExecutionTime([&](){
        conv2d_cpu_ref_zero_pad(X_host, F_host, Y_host_cpu, M, K, f_h, f_w);
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

    // Copy host filter to device constant memory
    cudaMemcpyToSymbol(c_filter, F_host, size_f);

    // Configure grid and block dims
    int num_blocks_row = (M + 32 - 1 ) / 32;
    int num_blocks_col = (K + 32 - 1 ) / 32;
    dim3 blockDim(32, 32);
    dim3 gridDim(num_blocks_row, num_blocks_col);

    // Execute kernel & measure time
    double gpu_time = measureKernelTime([&]() {
        constant_conv_2d<<<gridDim, blockDim>>>(X_device, Y_device, M, K);
        cudaDeviceSynchronize();
    });

    // Copy results back to host
    cudaMemcpy(Y_host_gpu, Y_device, size_y, cudaMemcpyDeviceToHost);

    // Compare against cpu reference for correctness
    bool result_match = compareResults(Y_host_cpu, Y_host_gpu, M * K, 1e-4, 1e-5);
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
