#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <ctime>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    // Element_id (i) = block_id * block_size + thread_id
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i]; // A[i] will translate to *(A + i)
    }
}

void vectorAddCPU(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int N = 1 << 20;
    size_t size = N * sizeof(float); // Memory size needed to store the vectors for addition

    // Allocate memory on the host (CPU)
    float *A_host = (float*)malloc(size);
    float *B_host = (float*)malloc(size);
    float *C_host_cpu = (float*)malloc(size);
    float *C_host_gpu = (float*)malloc(size);
}
