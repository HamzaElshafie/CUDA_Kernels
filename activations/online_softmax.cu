#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#include <utils.h>

__global__ void online_softmax(const float* __restrict__ A, float* __restrict__ C, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    float current_max = -INFINITY;
    float norm_factor = 0.0f;

    if (row < M)
    {
        // Phase 1: Get normalisation factor and global max
        for (int i = 0; i < N; i++)
        {
            float x = A[row * N + i]; 
            float previous_max = current_max;
            current_max = max(current_max, x);
            norm_factor = norm_factor * expf(previous_max - current_max) + expf(x - current_max);
        }

        // Phase 2: Get softmax output
        for (int k = 0; k < N; k++)
        {
            int index = row * N + k;
            float y = expf(A[index] - current_max) / norm_factor;
            C[index] = y;
        }
    }
}

int main()
{
    // Set matrix and block dimensions
    int num_rows = 1 << 10; // M
    int num_columns = 1 << 10; // N

    size_t size = (num_rows * num_columns) * sizeof(float);

    // Allocate host memory
    float* A_host = (float*)malloc(size);
    float* C_host_cpu = (float*)malloc(size);
    float* C_host_gpu = (float*)malloc(size);

    // Initialise matrix
    float* a_array[] {A_host};
    initialiseArrays(a_array, 1, num_rows * num_columns, -100.0f, 100.0f, 0);

    // Measure CPU execution time

    // Allocate device memory
    float* A_device;
    float* C_device;

    CUDA_CHECK(cudaMalloc((void**)&A_device, size));
    CUDA_CHECK(cudaMalloc((void**)&C_device, size));

    // Copy data from host 
    CUDA_CHECK(cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice));

    // Define kernel configs
    int threads_per_block = 256;
    int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;
    dim3 blockDim(threads_per_block);
    dim3 gridDim(blocks_per_grid);

    std::cout << "Grid configuration: " << std::endl;
    std::cout << "Grid dimension: " << gridDim.x << std::endl;
    std::cout << "Block dimension: " << blockDim.x << std::endl;

    // Measure GPU execution time
    float gpu_time = measureKernelTime([&](){
        online_softmax<<<gridDim, blockDim>>>(A_device, C_device, num_rows, num_columns);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    });

    std::cout << "GPU execution time: " << gpu_time << "ms" << std::endl;

    // Calculate throughput

    // Copy results to host
    CUDA_CHECK(cudaMemcpy(C_host_gpu, C_device, size, cudaMemcpyDeviceToHost));

    // Verify results

    // Free device memory
    CUDA_CHECK(cudaFree(A_device));
    CUDA_CHECK(cudaFree(C_device));

    // Free host memory
    free(A_host);
    free(C_host_cpu);
    free(C_host_gpu);

    return 0;
}
