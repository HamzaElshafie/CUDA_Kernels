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

    float cumulative_sum = 0.0f;
    int num_tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    
    // Iterate over tiles (Phase 1: Loading data)
    for (int t = 0; t < num_tiles; t++)
    {
        // Load tiles from A
        int tile_row = row;
        int tile_column = t * TILE_WIDTH + tx;
        sharedA[ty][tx] = (tile_row < M && tile_column < N) ? A[tile_row * N + tile_column] : 0.0f;

        // Load tiles from B
        tile_row =  t * TILE_WIDTH + ty;
        tile_column = column;
        sharedB[ty][tx] = (tile_row < N && tile_column < K) ? B[tile_row * K + tile_column] : 0.0f;

        __syncthreads();
        
        // Phase 2: Compute partial results iteratively
        for (int i = 0; i < TILE_WIDTH; i++)
        {
            cumulative_sum += sharedA[ty][i] * sharedB[i][tx];
        }
        __syncthreads(); // Ensure all threads finish using shared memory before it gets overwritten
    }
   
    // Check out of bounds
    if (row < M && column < K)
    {
        C[row * K + column] = cumulative_sum;
    }
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