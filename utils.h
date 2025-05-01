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

#define CUDA_CHECK(call) \
do { \ 
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << __LINE__ << " - " \
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

#endif