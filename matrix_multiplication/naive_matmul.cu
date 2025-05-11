#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#include <utils.h>

__global__ void matmulKernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int column_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < M && column_idx < K)
    {
        float cumulative_sum = 0.0f;
        for (int i = 0; i < N; i++)
        {
            cumulative_sum += A[row_idx * N + i] * B[i * K + column_idx];
        }
        int index = row_idx * K + column_idx;
        C[index] = cumulative_sum;
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
    // Calculate memory size required
    int num_rows_a = 1 << 9; // M
    int num_columns_a = 1 << 9; // N
    int num_rows_b = 1 << 9; // N
    int num_columns_b = 1 << 10; // K
    int block_size_rows = 32;
    int block_size_columns = 32;

    size_t size_a = num_rows_a * num_columns_a * sizeof(float);
    size_t size_b = num_rows_b * num_columns_b * sizeof(float);
    size_t size_c = num_rows_a * num_columns_b * sizeof(float);

    // Allocate host memory
    float* A_host = (float*)malloc(size_a);
    float* B_host = (float*)malloc(size_b);
    float* C_host_cpu = (float*)malloc(size_c);
    float* C_host_gpu = (float*)malloc(size_c);

    // Create an array of arrays to be initialised
    float* array_a[] {A_host};
    float* array_b[] {B_host};

    // Intialise arrays
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

    CUDA_CHECK(cudaMalloc((void**)&A_device,size_a));
    CUDA_CHECK(cudaMalloc((void**)&B_device,size_b));
    CUDA_CHECK(cudaMalloc((void**)&C_device,size_c));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(A_device, A_host, size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_device, B_host, size_b, cudaMemcpyHostToDevice));

    // Configure grid and block dimensions for kernel launch
    int num_blocks_rows = (num_rows_a + block_size_rows - 1) / block_size_rows;
    int num_block_columns = (num_columns_b + block_size_columns - 1) / block_size_columns;
    dim3 gridDim (num_block_columns, num_blocks_rows, 1);
    dim3 blockDim (block_size_columns, block_size_rows, 1);

    std::cout << "Grid configuration:" << std::endl;
    std::cout << "  Number of blocks (columns): " << gridDim.x << std::endl;
    std::cout << "  Number of blocks (rows): " << gridDim.y << std::endl;
    std::cout << "  Threads per block (columns): " << blockDim.x << std::endl;
    std::cout << "  Threads per block (rows): " << blockDim.y << std::endl;

    // Measure GPU execution time
    float gpu_time = measureKernelTime([&](){
        matmulKernel<<<gridDim, blockDim>>>(A_device, B_device, C_device, num_rows_a, num_columns_a, num_columns_b);
        CUDA_CHECK(cudaDeviceSynchronize());
    });

    std::cout << "GPU execution time: " << gpu_time << "ms" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;

    // Calculate total FLOPs for matrix multiplication: 2 * M * N * K
    double total_flops = 2.0 * num_rows_a * num_columns_a * num_columns_b;

    // Convert GPU time to seconds
    double gpu_time_sec = gpu_time / 1000.0;
    
    // Throughput in TFLOPs/s
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

// CPU execution time: 1938.05ms
// Grid configuration:
//   Number of blocks (columns): 32
//   Number of blocks (rows): 16
//   Threads per block (columns): 32
//   Threads per block (rows): 32
// GPU execution time: 0.414464ms
// Speedup: 4676.03x
// Throughput: 1.29534 TFLOPs/s
// Results match!