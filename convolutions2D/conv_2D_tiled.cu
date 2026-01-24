
%%writefile conv_2d_tiled.cu
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

#define FILTER_RADIUS 1
#define FILTER_H (2 * FILTER_RADIUS + 1)
#define FILTER_W (2 * FILTER_RADIUS + 1)
#define FILTER_NUM_ELEMENTS (FILTER_H * FILTER_W)

// Allocate constant memory space
__constant__ float c_filter[FILTER_NUM_ELEMENTS];

// Kernel
template <const int TILE_SIZE_IN, const int TILE_SIZE_OUT>
__global__ void conv_2d_tiled(const float* x, float* y, int M, int K) {
    // Allocate SMEM including Halo
    __shared__ float ts[TILE_SIZE_IN * TILE_SIZE_IN];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    int row = blockIdx.y * TILE_SIZE_OUT + ty - FILTER_RADIUS;
    int col = blockIdx.x * TILE_SIZE_OUT + tx - FILTER_RADIUS;

    // Phase 1: load input tile (including halo) into shared memory
    float val = (row >= 0 && row < M && col >= 0 && col < K) ? x[row * K + col] : 0.0f;
    ts[ty * TILE_SIZE_IN + tx] = val;
    __syncthreads();

    // Phase 2: Compute phase: compute output only for interior threads (turn off halo threads)
    int tile_ty = ty - FILTER_RADIUS;
    int tile_tx = tx - FILTER_RADIUS;
    
    if (tile_ty >=0 && tile_ty < TILE_SIZE_OUT && tile_tx >=0 && tile_tx < TILE_SIZE_OUT) {
        int out_row = blockIdx.y * TILE_SIZE_OUT + tile_ty;
        int out_col = blockIdx.x * TILE_SIZE_OUT + tile_tx;

        float res = 0.0f;
        if (out_row < M && out_col < K) {
            for (int i = 0; i < FILTER_H; i++){
                for (int j = 0; j < FILTER_W; j++) {
                    float val = ts[(tile_ty + i) * TILE_SIZE_IN + (tile_tx + j)];
                    res += val * c_filter[i * FILTER_W + j];
                }
            }

        // Phase 3: Store
        y[out_row * K + out_col] = res;
        }
    }
}

// CPU Reference
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

// Main
int main() {
    int M = 1 << 10;
    int K = 1 << 10;
    int f_h = 3;
    int f_w = 3;

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
    constexpr int TILE_SIZE_OUT = 30;
    constexpr int TILE_SIZE_IN = TILE_SIZE_OUT + 2 * FILTER_RADIUS;
    int num_blocks_row = (M + TILE_SIZE_OUT - 1 ) / TILE_SIZE_OUT;
    int num_blocks_col = (K + TILE_SIZE_OUT - 1 ) / TILE_SIZE_OUT;
    dim3 blockDim(TILE_SIZE_IN, TILE_SIZE_IN);
    dim3 gridDim(num_blocks_col, num_blocks_row);

    // Execute kernel & measure time
    double gpu_time = measureKernelTime([&]() {
        conv_2d_tiled<TILE_SIZE_IN, TILE_SIZE_OUT><<<gridDim, blockDim>>>(X_device, Y_device, M, K);
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
