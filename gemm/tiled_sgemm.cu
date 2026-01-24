#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#include <utils.h>

// Kernel logic
template <const int TILE_SIZE>
__global__ void naive_sgemm(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float smem_A[TILE_SIZE * TILE_SIZE];
    __shared__ float smem_B[TILE_SIZE * TILE_SIZE];

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    int ty = threadIdx.x / TILE_SIZE;
    int tx = threadIdx.x % TILE_SIZE;

    A += block_row * TILE_SIZE * K; 
    B += block_col * TILE_SIZE;
    C += (block_row * TILE_SIZE * N) + (block_col * TILE_SIZE);

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    float res = 0.0f;

    // Phase 1: Loading phase GMEM to SMEM
    for (int t = 0; t < num_tiles; t++){
        smem_A[ty * TILE_SIZE + tx] = A[ty * K + tx];
        smem_B[ty * TILE_SIZE + tx] = B[ty * N + tx];
        __syncthreads();

        // Phase 2: Compute phase
        for (int i = 0; i < TILE_SIZE; i++) {
            res += smem_A[ty * TILE_SIZE + i] * smem_B[i * TILE_SIZE + tx];
        }
        __syncthreads();

        A += TILE_SIZE;
        B += TILE_SIZE * N;
    }

    // Phase 3: Store phase
    C[ty * N + tx] = res;
}

// CPU reference
void matmulCPU(const float* A, const float* B, float* C, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float cumulative_sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                cumulative_sum += A[i * K + k] * B[k * N + j];
            }
            int index = i * N + j;
            C[index] = cumulative_sum;
        }
    }
}

/**
Entry point

Steps:
. Allocate memory on the host
. Create & initialise arrays / matrices
. Measure CPU reference time
. Allocate memory on device
. Copy data from host to device
. Configure grid and block dimensions for kernel launch
. Execute kernel & measure time
. Calculate FLOPS
. Convert GPU time to seconds
. Calculate throughput GFLOPS/TFLOPS
. Copy results back to host
. Verify results
. Free memories
*/
int main() {
    int M = 1 << 10;
    int K = 1 << 10; 
    int N = 1 << 10; 

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float* A_host = (float*)malloc(size_a);
    float* B_host = (float*)malloc(size_b);
    float* C_host_cpu = (float*)malloc(size_c);
    float* C_host_gpu = (float*)malloc(size_c);

    float* array_a[] {A_host};
    float* array_b[] {B_host};

    initialiseArrays(array_a, 1, M*K, 1.0f, 100.0f, 0);
    initialiseArrays(array_b, 1, K*N, 1.0f, 100.0f, 0);
    
    double cpu_duration = measureExecutionTime([&]() {
        matmulCPU(A_host, B_host, C_host_cpu, M, K, N);
    });

    std::cout << "CPU execution time: " << cpu_duration << "ms" << std::endl;

    float* A_device;
    float* B_device;
    float* C_device;

    cudaMalloc((void**)&A_device, size_a);
    cudaMalloc((void**)&B_device, size_b);
    cudaMalloc((void**)&C_device, size_c);

    cudaMemcpy(A_device, A_host, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, size_b, cudaMemcpyHostToDevice);

    // Define grid
    int block_num_rows = (M + 32 - 1) / 32;
    int block_num_cols = (N + 32 - 1) / 32;
    dim3 gridDim(block_num_cols, block_num_rows);

    // Define block
    dim3 blockDim(32*32);

    // Measure GPU kernel time
    float gpu_duration = measureKernelTime([&]() {
        naive_sgemm<>32<<<gridDim,blockDim>>>(A_device, B_device, C_device, M, K, N);
        cudaDeviceSynchronize();
    });

    std::cout << "GPU execution time: " << gpu_duration << "ms" << std::endl;
    std::cout << "Speedup: " << cpu_duration / gpu_duration << "x" << std::endl;

    // Calculate flops
    double total_flops = 2.0 * M * N * K;
    double gpu_duration_sec = gpu_duration / 1000.0;

    // Throughput in TFLOPs/s
    double tflops = total_flops / (gpu_duration_sec * 1e12);
    std::cout << "Throughput: " << tflops << " TFLOPs/s" << std::endl;

    // Copy results back to host
    cudaMemcpy(C_host_gpu, C_device, size_c, cudaMemcpyDeviceToHost);

    // Verify results
    bool results_match = compareResults(C_host_cpu, C_host_gpu, M * N, 1e-4, 1e-5);
    std::cout << (results_match? "Results match!" : "Results do not match!") << std::endl;

    // Free host memory
    free(A_host);
    free(B_host);
    free(C_host_cpu);
    free(C_host_gpu);

    // Free device memory
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    return 0;
}