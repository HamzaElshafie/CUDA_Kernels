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
#include <cmath>
#include <cuda_runtime.h>

/**
 * @brief CUDA error checking macro
 *
 * Evaluates a CUDA runtime call and checks for errors.
 * If an error is detected, prints detailed information and terminates the program.
 */
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
        << cudaGetErrorString(error) << " (" << error << ") " << std::endl; \
        exit(EXIT_FAILURE); \
} \
}while(0)

/**
 * @brief Initialise multiple arrays with random values in a specified range
 *
 * @param arrays     Array of pointers to initialize
 * @param num_arrays Number of arrays to initialize
 * @param size       Number of elements in each array
 * @param min       Minimum value for random numbers (default: 0.0)
 * @param max       Maximum value for random numbers (default: 1.0)
 * @param seed       Seed for random generator, 0 means use time(0) (default: 0)
 */
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

/**
 * @brief Measure CPU execution time using std::chrono
 *
 * @tparam Func     Function type
 * @param function  Function or lambda to measure
 * @return double   Execution time in milliseconds
 */
template <typename Function>
double measureExecutionTime(Function function)
{
    auto start = std::chrono::steady_clock::now();
    function();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = (end - start);
    return duration.count();
}

/**
 * @brief Measure GPU kernel execution time using CUDA events
 *
 * @tparam KernelFunc  Kernel function type
 * @tparam Args        Kernel argument types
 * @param kernel       Kernel function to measure
 * @param grid         Grid dimensions
 * @param block        Block dimensions
 * @param args         Kernel arguments
 * @return float       Execution time in milliseconds
 */
template <typename KernelFunc>
float measureKernelTime(KernelFunc kernel)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsed_time;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start stopwatch
    CUDA_CHECK(cudaEventRecord(start));
    // Launch kernel
    kernel();
    // Stop stopwatch
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    // Free events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsed_time;
}

/**
 * @brief Compare results between CPU and GPU outputs using absolute and relative tolerance
 *
 * @param cpu_result  CPU computed results
 * @param gpu_result  GPU computed results
 * @param size        Number of elements to compare
 * @param atol        Absolute tolerance (default: 1e-4)
 * @param rtol        Relative tolerance (default: 1e-5)
 * @return bool       True if results match within tolerances, false otherwise
 */
bool compareResults(const float *cpu_result, const float *gpu_result,
                    size_t size, float atol = 1e-4f, float rtol = 1e-5f)
{
    for (size_t i = 0; i < size; i++)
    {
        float a = cpu_result[i];
        float b = gpu_result[i];
        float abs_diff = std::fabs(a - b);
        float rel_diff = abs_diff / std::fmax(std::fabs(a), std::fabs(b));

        if (abs_diff > atol && rel_diff > rtol)
        {
            std::cout << "Mismatch at index " << i
                      << ": CPU = " << a
                      << ", GPU = " << b
                      << ", abs diff = " << abs_diff
                      << ", rel diff = " << rel_diff
                      << std::endl;
            return false;
        }
    }
    return true;
}

#endif
