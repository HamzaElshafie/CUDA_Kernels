#include <iostream>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void matrixAdd(const float* A, const float* B, float* C, int rows, int columns)
{
    // Get thread row index
    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
    // Get thread column index
    int column_index = blockIdx.x * blockDim.x + threadIdx.x;
    // Check out of bounds
    if (row_index < rows && column_index < columns)
    {
        int index = row_index * columns + column_index;
        C[index] = A[index] + B[index];
    }
}

void matrixAddCPU(const float* A, const float* B, float* C, int rows, int columns)
{
    for (int row_index = 0; row_index < rows; row_index++)
    {
        for (int column_index = 0; column_index < columns; column_index++)
        {
            int index = row_index * columns + column_index;
            C[index] = A[index] + B[index];
        }
    }
}

void initialiseVectors(float *A, float *B, int N)
{
    srand(static_cast<unsigned int>(time(0)));

    for (int i = 0; i < N; i++)
    {
        A[i] = static_cast<float>(rand());
        B[i] = static_cast<float>(rand());
    }
}

// Experiment with auto instead of template later
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
    int num_rows = 1 << 12;
    int block_size_rows = 32;
    int num_columns = 1 << 12;
    int block_size_columns = 32;

    size_t size = num_rows * num_columns * sizeof(float);

    // Allocate memory on host
    float* A_host = (float*)malloc(size);
    float* B_host = (float*)malloc(size);
    float* C_mat_cpu = (float*)malloc(size);
    float* C_mat_gpu = (float*)malloc(size);

    // Initialise matrix
    initialiseVectors(A_host, B_host, num_rows * num_columns);

    // Measure CPU execution time
    double cpu_time = measureExecutionTime([&]()
    {
        matrixAddCPU(A_host, B_host, C_mat_cpu, num_rows, num_columns);
    });
    std::cout << "CPU execution time: " << cpu_time << " ms" << '\n';

    // Allocate memory on device
    float* A_device;
    float* B_device;
    float* C_device;

    cudaMalloc((void**)&A_device, size);
    cudaMalloc((void**)&B_device, size);
    cudaMalloc((void**)&C_device, size);

    // Copy data from host to device
    cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, size, cudaMemcpyHostToDevice);

    // Define grid
    int num_blocks_rows = (num_rows + block_size_rows - 1) / block_size_rows;
    int num_blocks_columns = (num_columns + block_size_columns - 1) / block_size_columns;

    dim3 block(block_size_columns, block_size_rows, 1);
    dim3 grid(num_blocks_columns, num_blocks_rows, 1);

    // Measure GPU execution time
    double gpu_time = measureExecutionTime([&]()
    {
        matrixAdd<<<grid, block>>>(A_device, B_device, C_device, num_rows, num_columns);
        cudaDeviceSynchronize();
    });
    std::cout << "GPU execution time: " << gpu_time << " ms" << '\n';

    // Copy results from device to host
    cudaMemcpy(C_mat_gpu, C_device, size, cudaMemcpyDeviceToHost);

    bool success = compareResults(C_mat_cpu, C_mat_gpu, num_rows * num_columns);
    std::cout << (success ? "CPU and GPU results match!" : "Results mismatch!") << '\n';

    // Free device memory
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    // Free host memory
    free(A_host);
    free(B_host);
    free(C_mat_cpu);
    free(C_mat_gpu);

    return 0;
}

// CPU execution time: 349.847 ms
// GPU execution time : 3.42599 ms 
// CPU and GPU results match !