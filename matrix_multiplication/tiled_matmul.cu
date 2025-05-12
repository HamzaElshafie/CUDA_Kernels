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

void matmulCPU(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            float cumulative_sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                cumulative_sum += A[i * N + k] * B[k * K + j];
            }
            int index = i * K + j;
            C[index] = cumulative_sum;
        }
    }
}

int main()
{
    // Specify matrices & block dimensions
    int num_rows_a = 1 << 10; // M
    int num_columns_a = 1 << 10; // N
    int num_rows_b = 1 << 10; // N
    int num_columns_b = 1 << 11; // K
    int block_size_rows = 32;
    int block_size_columns = 32;

    // Calculate memory size required
    size_t size_a = (num_rows_a * num_columns_a) * sizeof(float);
    size_t size_b = num_rows_b * num_columns_b * sizeof(float);
    size_t size_c = num_rows_a * num_columns_b * sizeof(float);

    // Allocate host memory
    float* A_host = (float*)malloc(size_a);
    float* B_host = (float*)malloc(size_b);
    float* C_host_cpu = (float*)malloc(size_c);
    float* C_host_gpu = (float*)malloc(size_c);

    // Initialise matrices
    float* array_a [] {A_host};
    float* array_b [] {B_host};
    initialiseArrays(array_a, 1, num_rows_a * num_columns_a, 1.0f, 100.0f, 0);
    initialiseArrays(array_b, 1, num_rows_b * num_columns_b, 1.0f, 100.0f, 0);

    // Measure CPU execution time
    double cpu_time = measureExecutionTime([&](){
        matmulCPU(A_host, B_host, C_host_cpu, num_rows_a, num_columns_a, num_columns_b);
    });

    std::cout << "CPU execution time: " << cpu_time << "ms" << std::endl;

    // Allocate device memory
    float* A_device;
    float* B_device;
    float* C_device;

    CUDA_CHECK(cudaMalloc((void**)&A_device, size_a));
    CUDA_CHECK(cudaMalloc((void**)&B_device, size_b));
    CUDA_CHECK(cudaMalloc((void**)&C_device, size_c));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(A_device, A_host, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_device, B_host, size_b, cudaMemcpyHostToDevice));

    // Configure grid and block dimensions for kernel launch
    int num_blocks_rows = (num_rows_a + block_size_rows - 1) / block_size_rows;
    int num_blocks_columns = (num_columns_b + block_size_columns - 1) / block_size_columns;
    dim3 gridDim (num_blocks_columns, num_blocks_rows, 1);
    dim3 blockDim (block_size_columns, block_size_rows, 1);

    std::cout << "Grid configuration:" << std::endl;
    std::cout << "  Number of blocks (columns): " << gridDim.x << std::endl;
    std::cout << "  Number of blocks (rows): " << gridDim.y << std::endl;
    std::cout << "  Threads per block (columns): " << blockDim.x << std::endl;
    std::cout << "  Threads per block (rows): " << blockDim.y << std::endl;

    // Measure GPU exeuction time
    float gpu_time = measureKernelTime([&](){
        tiled_matmul<<<gridDim, blockDim>>>(A_device, B_device, C_device, num_rows_a, num_columns_a, num_columns_b);
        CUDA_CHECK(cudaDeviceSynchronize());
    });

    std::cout << "GPU execution time: " << gpu_time << "ms" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;

    // Calculate the number of FLOPs the kernel does
    double total_flops = 2.0 * num_rows_a * num_columns_a * num_columns_b;

    // Convert GPU time to seconds
    double gpu_time_sec = gpu_time / 1000.0f;

    // Calculate throughput in TFLOPs/sec
    double tflops = total_flops / (gpu_time_sec * 1e12);
    std::cout << "Throughput: " << tflops << " TFLOPs/s" << std::endl;

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C_host_gpu, C_device, size_c, cudaMemcpyDeviceToHost));

    // Verify results
    bool results_match = compareResults(C_host_cpu, C_host_gpu, num_rows_a * num_columns_b, 1e-4, 1e-5);
    std::cout << (results_match? "Results match!" : "Results do not match!") << std::endl;

    // Free device memory
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    // Free host memory
    free(A_host);
    free(B_host);
    free(C_host_cpu);
    free(C_host_gpu);


    return 0;
}

// ptxas info    : 0 bytes gmem
// ptxas info    : Compiling entry function '_Z12tiled_matmulPKfS0_Pfiii' for 'sm_89'
// ptxas info    : Function properties for _Z12tiled_matmulPKfS0_Pfiii
//     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
// ptxas info    : Used 38 registers, 8192 bytes smem, 388 bytes cmem[0]
// CPU execution time: 15171.1ms
// Grid configuration:
//   Number of blocks (columns): 64
//   Number of blocks (rows): 32
//   Threads per block (columns): 32
//   Threads per block (rows): 32
// GPU execution time: 1.94726ms
// Speedup: 7791x
// Throughput: 2.20564 TFLOPs/s
// Results match!