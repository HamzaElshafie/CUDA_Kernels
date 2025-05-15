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

void online_softmax_cpu(const float* A, float* C, int M, int N) {
    for (int row = 0; row < M; row++) {
        // Phase 1: Find the maximum value using the online algorithm
        float local_max = -INFINITY;
        float local_norm = 0.0f;
        
        for (int col = 0; col < N; col++) {
            float x = A[row * N + col];
            if (x > local_max) {
                local_norm = local_norm * expf(local_max - x) + 1.0f; // Same adjustment as CUDA
                local_max = x;
            } else {
                local_norm += expf(x - local_max);
            }
        }
        
        float global_max = local_max;
        float global_norm = local_norm;
        
        // Phase 2: Calculate softmax values
        for (int col = 0; col < N; col++) {
            C[row * N + col] = expf(A[row * N + col] - global_max) / global_norm;
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
    double cpu_time = measureExecutionTime([&](){
        online_softmax_cpu(A_host, C_host_cpu, num_columns, num_columns);
    });

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
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;

    // Calculate throughput

    // Copy results to host
    CUDA_CHECK(cudaMemcpy(C_host_gpu, C_device, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool results_match = compareResults(C_host_cpu, C_host_gpu, num_rows * num_columns, 1e-4, 1e-5);
    std::cout << (results_match? "Results match!" : "Results do not match!") << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(A_device));
    CUDA_CHECK(cudaFree(C_device));

    // Free host memory
    free(A_host);
    free(C_host_cpu);
    free(C_host_gpu);

    return 0;
}