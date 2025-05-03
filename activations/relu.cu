#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#include <utils.h>

__global__ void ReluKernel(const float* A, float* C, int rows, int columns)
{
    // Get thread row index
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    // Get thread column index
    int column_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < rows && column_idx < columns)
    {
         int index = row_idx * columns + column_idx;
         C[index] = fmaxf(0, A[index]);
    }
}

void ReluCPU(const float* A, float* C, int rows, int columns)
{   
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            int index = (i * columns) + j;
            C[index] = fmaxf(0.0f, A[index]);
        }
    }
}

int main()
{
    int num_rows = 1 << 12;
    int block_size_rows = 32;
    int num_columns = 1 << 12;
    int block_size_columns = 32;
    int num_elements = num_rows * num_columns;

    // Calculate memory size required
    const size_t size = num_rows * num_columns * sizeof(float);

    // Allocate host memory
    float* A_host = (float*)(malloc(size));
    float* C_host_cpu = (float*)(malloc(size));
    float* C_host_gpu = (float*)(malloc(size));

    // Create an array of the arrays to be initialised
    float* arrays[] {A_host};
    initialiseArrays(arrays, 1, num_elements, -100.0f, 100.0f, 0);

    // Measure CPU execution time
    double cpu_time = measureExecutionTime([&](){
        ReluCPU(A_host, C_host_cpu, num_rows, num_columns);
    });
    
    std::cout << "CPU execution time: " << cpu_time << "ms" << std::endl;
    
    // Allocate device memory
    float* A_device;
    float* C_device;
    CUDA_CHECK(cudaMalloc((void**)&A_device, size));
    CUDA_CHECK(cudaMalloc((void**)&C_device, size));
    
    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice));

    int num_blocks_rows = (num_rows + block_size_rows - 1) / block_size_rows;
    int num_block_columns = (num_columns + block_size_columns - 1) / block_size_columns;
    dim3 gridDim(num_block_columns, num_blocks_rows, 1);
    dim3 blockDim(block_size_columns, block_size_rows, 1);

    std::cout << "Grid configuration:" << std::endl;
    std::cout << "  Number of blocks (columns): " << gridDim.x << std::endl;
    std::cout << "  Number of blocks (rows): " << gridDim.y << std::endl;
    std::cout << "  Threads per block (columns): " << blockDim.x << std::endl;
    std::cout << "  Threads per block (rows): " << blockDim.y << std::endl;

    float gpu_time = measureKernelTime([&](){
        ReluKernel<<<gridDim, blockDim>>>(A_device, C_device, num_rows, num_columns);
        CUDA_CHECK(cudaDeviceSynchronize());
    });

    std::cout << "GPU execution time: " << gpu_time << "ms" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(C_host_gpu, C_device, size, cudaMemcpyDeviceToHost)); 
    
    // Verify results
    bool results_match = compareResults(C_host_cpu, C_host_gpu, num_rows * num_columns);
    std::cout << (results_match ? "Results match!" : "Results do not match!") << std::endl;

    // Free device memory
    cudaFree(A_device);
    cudaFree(C_device);

    // Free host memory
    free(A_host);
    free(C_host_cpu);
    free(C_host_gpu);

    return 0;
}

// CPU execution time: 90.4373ms
// Grid configuration:
//   Number of blocks (columns): 128
//   Number of blocks (rows): 128
//   Threads per block (columns): 32
//   Threads per block (rows): 32
// GPU execution time: 0.53808ms
// Speedup: 168.074x
// Results match!