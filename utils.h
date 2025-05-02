/**
 * @file utils.h
 * @brief Utility functions for CUDA development
 *
 * This header provides core utility functions focusing on error checking timing,
 * initialising arrays in memory and result comparison.
 */

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \ 
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
        << cudaGetErrorString(error) << " (" << error << ") " << std::endl; \
        exit(EXIT_FAILURE); \
} \
}while(0)

void initialiseArrays(float** arrays, int num_arrays, size_t size, float min=0.0f, float max=1.0f, unsigned int seed=0)
{
    // Set random seed
    if (seed == 0)
    {
        seed = static_cast<unsigned int>(time(0)); // get current time
    }
    srand(seed);

    float range = max - min;

    for (int i = 0; i < num_arrays; i++) // Iterate through each array pointer
    {
        for (size_t j = 0; j < size; j++) // Iterate through each element
        {
            arrays[i][j] = min + (static_cast<float>(rand()) / RAND_MAX) * range;
        }
    }
}

template <typename Function>
double measureExecutionTime(Function function)
{
    auto start = std::chrono::steady_clock::now();
    function();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = (end - start);
    return duration.count();
}

template <typename KernelFunc, typename... Args>
float measureKernelTime(KernelFunc kernel, dim3 grid, dim3 block, Args... args)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsed_time;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start stopwatch
    CUDA_CHECK(cudaEventRecord(start));
    // Launch kernel
    kernel<<<grid, block>>>(args...);
    // Stop stopwatch
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsed_time;
}

bool compareResults(const float *cpu_result, const float *gpu_result,
                    size_t size, float tolerance = 1e-5f)
{
    for (size_t i = 0; i < size; i++)
    {
        float diff = fabs(cpu_result[i] - gpu_result[i]);
        if (diff > tolerance)
        {
            std::cout << "Mismatch at index " << i
                      << ": CPU = " << cpu_result[i]
                      << ", GPU = " << gpu_result[i]
                      << ", diff = " << diff
                      << std::endl;
            return false;
        }
    }
    return true;
}

#endif
