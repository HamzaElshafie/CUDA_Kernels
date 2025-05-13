#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#include <utils.h>

__global__ void online_softmax(const __restrict__ float* A, __restrict__ float* C, int M, int N)
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
    return 0;
}
