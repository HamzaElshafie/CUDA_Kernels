#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#include <utils.h>

#define TILE_WIDTH 32

__global__ void tiled_matmul(const float* A, const float* B, float* C, int M, int N, int K)
{
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * blockDim.y + ty;
    int column = blockIdx.x * blockDim.x + tx;

    


} 


int main()
{
    // Specify matrices & block dimensions

    // Calculate memory size required

    // Allocate host memory

    // Initialise matrices

    // Allocate device memory

    // Copy data to device

    // Configure grid and block dimensions for kernel launch

    // Measure GPU exeuction time

    // Calculate the number of FLOPs the kernel does

    // Convert GPU time to seconds

    // Calculate throughput in TFLOPs/sec

    // Copy results back to host memory

    // Free device memory

    // Free host memory

    return 0;
}