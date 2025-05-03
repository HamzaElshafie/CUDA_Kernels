#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <utils.h>

__global__ void ReluKernel(const float* A, float* C, int rows, int columns)
{

}

void ReluCPU(const float* A, float* C, int rows, int columns)
{

}

int main()
{
    int num_rows = 1 << 14;
    int block_size_rows = 32;
    int num_columns = 1 << 14;
    int block_size_columns = 32;

    // Calculate memory size required
    const size_t size = num_rows * num_columns * sizeof(float);

    // Allocate host memory
    float* A_host = (float*)(malloc(size));
    float* C_host_cpu = (float*)(malloc(size));
    float* C_host_gpu = (float*)(malloc(size));

    // Create an array of the arrays to be initialised
    float* arrays[] {A_host};
    initialiseArrays(arrays, 1, size, min=0.0f, max=100.0f, seed=0);

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
    std::cout << "Number of row blocks: " << num_blocks_rows << std::endl;
    int num_block_columns = (num_columns + block_size_columns - 1) / block_size_columns;
    std::cout << "Number of column blocks: " << num_block_columns << std::endl;
    dim3 gridDim(num_blocks_rows, num_block_columns, 1);
    dim3 blockDim(block_size_rows, block_size_columns, 1);

    float gpu_time = measureKernelTime([&](){
        ReluKernel<<<gridDim, blockDim>>>(A_device, C_device, num_rows, num_columns);
    });

    std::cout << "GPU execution time: " << gpu_time << "ms" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(C_host_gpu, C_device, size, cudaMemcpyDeviceToHost));    

    // Free device memory
    cudaFree(A_device);
    cudaFree(C_device);

       // Free host memory
    free(A_host);
    free(C_host_cpu);
    free(C_host_gpu);

    return 0;
}