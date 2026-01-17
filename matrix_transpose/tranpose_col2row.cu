#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>


/**
Tranposing kernel. Naive row-major → row-major transpose.

Steps:
1. Find global index of thread to find its corresponding element
1. Find the global row index of thread
2. Find the global col index of thread
*/ 
__global__ void mat_transpose_f32(float* x, float* y, const int M, const int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / K;
    int col = idx % K;

    if (idx < M * K) {
        y[col * M + row] = x[idx];
    }
}
