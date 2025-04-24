#include <iostream>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void matrixAdd(const float* A, const float* B, float* C, int rows, int columns)
{
    // TODO
}

void matrixAddCPU(const float* A, const float* B, float* C, int rows, int columns)
{
    // TODO
}

void initilaiseVectors(float* A, float* B, int N)
{
    // TODO
}

// Experiment with auto instead of template later
template <typename Func>
double measureExecutionTime(Func func)
{
    // TODO
}

bool compareResults(const float *A, const float *B, int N)
{
    // TODO
}

int main()
{
    // TODO
}