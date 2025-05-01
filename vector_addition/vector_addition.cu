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

void initialiseVectors(float* A, float* B, int N)
{
    srand(static_cast<unsigned int>(time(0)));

    for (int i = 0; i < N; i++)
    {
        A[i] = static_cast<float>(rand()); // divide by RAND_MAX later if you want to normalise values
        B[i] = static_cast<float>(rand());
    }
}

template <typename Func>
double measureExecutionTime(Func func)
{
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

bool compareResults(const float *A, const float *B, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (fabs(A[i] - B[i]) > 1e-4)
        {
            std::cout << "Mismatch at index " << i << ": CPU=" << A[i] << " GPU=" << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    int N = 1 << 20; // 1 million elements
    size_t size = N * sizeof(float); // Memory size needed to store the vectors for addition

    // Allocate memory on the host (CPU)
    float* A_host = (float*)malloc(size); // malloc return a void pointer
    float* B_host = (float*)malloc(size);
    float* C_host_cpu = (float*)malloc(size);
    float* C_host_gpu = (float*)malloc(size);

    initialiseVectors(A_host, B_host, N);

    // Measure CPU execution time for vector addition
    double cpu_time = measureExecutionTime([&]() 
    {
        vectorAddCPU(A_host, B_host, C_host_cpu, N);
    });

    std::cout << "CPU execution time: " << cpu_time << "ms" << '\n';

    // Allocate memory on the device (GPU)
    float* A_device;
    float* B_device;
    float* C_device;

    cudaMalloc((void**)&A_device, size);
    cudaMalloc((void**)&B_device, size);
    cudaMalloc((void**)&C_device, size);

    cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, size, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    double gpu_time = measureExecutionTime([&]()
    {
        vectorAdd<<<blocks_per_grid, threads_per_block>>>(A_device, B_device, C_device, N);
        cudaDeviceSynchronize();
    });

    std::cout << "GPU execution time: " << gpu_time << "ms" << '\n';

    cudaMemcpy(C_host_gpu, C_device, size, cudaMemcpyDeviceToHost);

    bool success = compareResults(C_host_cpu, C_host_gpu, N);
    std::cout << (success ? "CPU and GPU results match!" : "Results mismatch!");

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    free(A_host);
    free(B_host);
    free(C_host_cpu);
    free(C_host_gpu);

    return 0;
}

// CPU execution time: 166.674ms
// Launching kernel with 131072 blocks of 256 threads.
// GPU execution time: 1.6756ms
// CPU and GPU results match!