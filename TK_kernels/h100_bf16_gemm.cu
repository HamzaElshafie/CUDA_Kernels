#include "kittens.cuh"
#include "prototype.cuh"
#include "common.cuh"

#include <iostream>
#include <cuda_bf16.h>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using base_tile = st_bf<64, 64>; // base tile is 64 by 64. Output block of C = Rows: M_BLOCK * 64 Cols: N_BLOCK * 64 
    using global_layout = gl<bf16, 1, 1, -1 ,-1, base_tile>; // -1, -1 are the runtime rows and cols dims of the matrices
    struct globals {global_layout A, B, C;};
    struct input_block {base_tile a[M_BLOCK], b[N_BLOCK];}; // describes one stage of the smem input buffer. Contains tiles needed for one K slice (one iter) of the GEMM.
    struct finish_block {base_tile c[M_BLOCK][N_BLOCK];}; // Smem used to stage at the end for TMA stores
    struct common_state {int2 coord;}; // Coordinate position (x,y) measured in 64x64 tile
    // FP32 register accumulator fragment for one consumer warpgroup: covers N_BLOCK*64 columns (e.g. 256 when N_BLOCK=4); 
    // 16 is TK’s internal fragment height used by warpgroup MMA for the m64 output
    struct consumer_state {rt_fl<16, N_BLOCK*base_tile::cols> accum;}; 
};

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK;
    static constexpr int N_BLOCK = _N_BLOCK;
    static constexpr int SUPER_M = _SUPER_M;
    using layout = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>; 
    static constexpr int NUM_CONSUMER_WARPS = M_BLOCK * 4;
    static constexpr int INPUT_PIPE_STAGES = 4;
    static constexpr int PRODUCER_BARRIER_ARRIVALS = 1; // only one lane issues TMA ops, so one arrival
    // Returns grid dimensions. If PERSISTENT_GRID is true, launch one block per H100 SM (132) and 
    // let blocks loop over tiles via task_iter; otherwise launch one block per C tile group (M_BLOCK×N_BLOCK base tiles).
    template<bool PERSISTENT_GRID=true>
    __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERSISTENT_GRID ? 132 : (M * N) / (M_BLOCK * N_BLOCK * layout::base_tile::num_elements));
    }
    // TK's template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int r_blocks = args.globals.C.rows() / (M_BLOCK * 64);
        int c_blocks = args.globals.C.cols() / (N_BLOCK * 64);

        int super_rows   = (r_blocks / SUPER_M) * SUPER_M;
        int final_rows   = r_blocks - super_rows;
        int super_repeat = SUPER_M * c_blocks;
        int task_id = args.task_iter * gridDim.x + blockIdx.x;    
        if (task_id < super_rows * c_blocks) {
            args.common.coord = {
                SUPER_M * (task_id / super_repeat) + task_id % SUPER_M,
                (task_id % super_repeat) / SUPER_M
            };
        }
        else if (task_id < r_blocks * c_blocks) { // handle remainder rows (final_rows > 0)
            int remainder_id = task_id - super_rows * c_blocks;
            args.common.coord = {
                super_rows + (remainder_id % final_rows), // task_row in the remainder band
                remainder_id / final_rows  // task_col
            };
        }
        else {
            args.num_iters = -1;
            return ;
        }
        args.num_iters = args.globals.A.cols() / 64;
        constexpr int NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS / 4; // No. of consumer warp groups = M_BLOCK
        int wg = warpgroup::groupid();
        bool is_prod = (wg == NUM_CONSUMER_WARPGROUPS);
        int row_in_block = is_prod ? 0 : wg;
        args.common.coord.x = args.common.coord.x * M_BLOCK + row_in_block;
        args.common.coord.y = args.common.coord.y * N_BLOCK;
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // Allocate fewer registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                
                for (int i = 0; i < M_BLOCK; i++) {
                    tma::load_async(
                        args.input.a[i], // destination. smem tile (64 * 64)
                        args.globals.A, // source: global A
                        {args.common.coord.x + i, args.iter}, // source tile row & col index in A
                        args.inputs_arrived // barrier signal copy is complete
                    );
                }
                for (int i = 0; i < N_BLOCK; i++) {
                    tma::load_async(
                        args.input.b[i], // destination. smem tile (64 * 64)
                        args.globals.B, // source: global B
                        {args.iter, args.common.coord.y + i}, // source tile row & col index in B
                        args.inputs_arrived // barrier signal copy is complete
                    );
                }
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumer
            kittens::warp::zero(args.state.accum); // initialise accumilator to zero
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AB(
                args.state.accum,
                args.input.a[warpgroup::groupid()], // A operand: this warpgroup’s 64×64 A tile from shared memory
                reinterpret_cast<wide_tile&>(args.input.b) // B operand: treat b[0..N_BLOCK-1] as one 64×(64*N_BLOCK) shared slab
            );
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) {arrive(args.inputs_finished);}  // signal stage is done
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(
                reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), // this warpgroup’s output row as a 64×(64*N_BLOCK) SMEM slab
                args.state.accum
            );
            warpgroup::sync(warpgroup::groupid()+4); 
            if (warpgroup::laneid() == 0) {
                for (int i = 0; i < N_BLOCK; i++) {
                    tma::store_async(
                        args.globals.C, 
                        args.finish.c[warpgroup::groupid()][i],
                        {args.common.coord.x, args.common.coord.y+i}
                    );
                    tma::store_async_read_wait();  // wait until TMA finished reading that SMEM tile
                }
            }
            kittens::warp::zero(args.state.accum);  // reset accumulator for next persistent task
            if (warp::laneid() == 0) {arrive(args.finish_finished);} // signal finish stage done and reusable
        }
    };
};

template<typename mmt>
void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using global_layout = typename mmt::layout::global_layout;
    using globals = typename mmt::layout::globals;
    global_layout Ag{d_A, nullptr, nullptr, M, K};
    global_layout Bg{d_B, nullptr, nullptr, K, N};
    global_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt><<<grid, block, ::MAX_SHARED_MEMORY-1024>>>(G);
}

// TK benchmark 

template<typename mmt>
double run_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    std::cout << "--------------------  [TK]  M=" << M << " N=" << N << " K=" << K
              << "  --------------------\n";
    std::cout << "Block size: " << mmt::M_BLOCK*64 << "x" << mmt::N_BLOCK*64 << "\n";

    sleep_ms(500);

    // Allocate enough buffer groups to exceed the L2 cache
    int l2_cache_size;
    CUDACHECK(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0));
    const size_t arg_size       = 2 * (size_t(M)*K + size_t(N)*K + size_t(M)*N);
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int    arg_group_count = (arg_size > ideal_arg_size)
                                       ? 1
                                       : int(ideal_arg_size / arg_size) + 1;

    std::vector<__nv_bfloat16*> d_A(arg_group_count), d_B(arg_group_count), d_C(arg_group_count);
    __nv_bfloat16* d_C_ref;
    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M*K*sizeof(__nv_bfloat16)));
        CUDACHECK(cudaMalloc(&d_B[i], K*N*sizeof(__nv_bfloat16)));
        CUDACHECK(cudaMalloc(&d_C[i], M*N*sizeof(__nv_bfloat16)));
    }
    CUDACHECK(cudaMalloc(&d_C_ref, M*N*sizeof(__nv_bfloat16)));
    std::cout << "Allocated device memory\n";

    uint64_t seed = 42;
    for (int i = 0; i < arg_group_count; i++) {
        fill<__nv_bfloat16, FillMode::RANDOM>  (d_A[i], M*K, seed + i*100,     -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::RANDOM>  (d_B[i], K*N, seed + i*100 + 1, -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_C[i], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_C_ref, M*N, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Initialised matrices on device\n";

    // Reference GEMM (transpose_b=false: B is row-major K×N).
    // The reference kernel is a naive thread-per-element implementation and is only
    // practical for small sizes; skip the correctness check for large matrices.
    const bool do_correctness = (M <= 1024 && N <= 1024 && K <= 1024);
    if (do_correctness) {
        reference_gemm<__nv_bfloat16, __nv_bfloat16, false>(d_C_ref, d_A[0], d_B[0], M, N, K);
        CUDACHECK(cudaDeviceSynchronize());
        std::cout << "Computed reference GEMM\n";
    } else {
        std::cout << "Skipping reference GEMM (matrix too large for naive kernel)\n";
    }

    // Set dynamic shared memory limit
    CUDACHECK(cudaFuncSetAttribute(
        prototype::lcf::kernel<mmt>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        ::MAX_SHARED_MEMORY - 1024));

    dim3 grid  = mmt::grid(M, N, K);
    dim3 block = kittens::prototype::detail::NUM_THREADS_v<mmt>;

    int num_warmups = ncu ? 0 : 5;
    int num_iters   = ncu ? 1 : 10;

    for (int i = 0; i < num_warmups; i++)
        inner_run<mmt>(d_A[i % arg_group_count], d_B[i % arg_group_count],
                       d_C[i % arg_group_count], M, N, K, grid, block);

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++)
        inner_run<mmt>(d_A[i % arg_group_count], d_B[i % arg_group_count],
                       d_C[i % arg_group_count], M, N, K, grid, block);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float milliseconds;
    CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops  = 2.0 * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel time : " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    if (do_correctness) check_correctness(d_C[0], d_C_ref, M * N);

    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]); cudaFree(d_B[i]); cudaFree(d_C[i]);
    }
    cudaFree(d_C_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return tflops;
}

// cuBLAS benchmark

void inner_run_cublas(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C,
                      size_t M, size_t N, size_t K) {
    float alpha = 1.0f, beta = 0.0f;
    // Row-major C = A*B  ↔  col-major C^T = B^T * A^T
    CUBLASCHECK(cublasGemmEx(get_cublas_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        B, CUDA_R_16BF, (int)N,
        A, CUDA_R_16BF, (int)K,
        &beta,
        C, CUDA_R_16BF, (int)N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

double run_cublas_benchmark(size_t M, size_t N, size_t K, bool ncu = false) {
    std::cout << "--------------------  [cuBLAS]  M=" << M << " N=" << N << " K=" << K
              << "  --------------------\n";

    sleep_ms(500);

    int l2_cache_size;
    CUDACHECK(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0));
    const size_t arg_size       = 2 * (size_t(M)*K + size_t(N)*K + size_t(M)*N);
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int    arg_group_count = (arg_size > ideal_arg_size)
                                       ? 1
                                       : int(ideal_arg_size / arg_size) + 1;

    std::vector<__nv_bfloat16*> d_A(arg_group_count), d_B(arg_group_count), d_C(arg_group_count);
    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M*K*sizeof(__nv_bfloat16)));
        CUDACHECK(cudaMalloc(&d_B[i], K*N*sizeof(__nv_bfloat16)));
        CUDACHECK(cudaMalloc(&d_C[i], M*N*sizeof(__nv_bfloat16)));
    }

    uint64_t seed = 42;
    for (int i = 0; i < arg_group_count; i++) {
        fill<__nv_bfloat16, FillMode::RANDOM>  (d_A[i], M*K, seed + i*100,     -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::RANDOM>  (d_B[i], K*N, seed + i*100 + 1, -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_C[i], M*N, 0.0f);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Warm-up cuBLAS (first call initialises internal state)
    get_cublas_handle();
    int num_warmups = ncu ? 0 : 5;
    int num_iters   = ncu ? 1 : 10;

    for (int i = 0; i < num_warmups; i++)
        inner_run_cublas(d_A[i % arg_group_count], d_B[i % arg_group_count],
                         d_C[i % arg_group_count], M, N, K);

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++)
        inner_run_cublas(d_A[i % arg_group_count], d_B[i % arg_group_count],
                         d_C[i % arg_group_count], M, N, K);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float milliseconds;
    CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops  = 2.0 * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel time : " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]); cudaFree(d_B[i]); cudaFree(d_C[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return tflops;
}

// Main

int main() {
    std::cout << "\n========== ThunderKittens vs cuBLAS  |  BF16 GEMM  |  H100 ==========\n\n";

    // Warm up the cuBLAS handle once before the timed loops
    get_cublas_handle();

    constexpr size_t SIZES[] = {512, 1024, 2048, 4096, 8192};

    for (size_t N : SIZES) {
        std::cout << "\n==================== N = " << N << " ====================\n";
        double tk_tflops     = run_benchmark<matmul_template<2,4,8>>(N, N, N);
        double cublas_tflops = run_cublas_benchmark(N, N, N);
        std::cout << "  TK / cuBLAS ratio : " << (tk_tflops / cublas_tflops) << "x\n";
    }

    return 0;
}



// ========== ThunderKittens vs cuBLAS  |  BF16 GEMM  |  H100 ==========


// ==================== N = 512 ====================
// --------------------  [TK]  M=512 N=512 K=512  --------------------
// Block size: 128x256
// Allocated device memory
// Initialised matrices on device
// Computed reference GEMM
// Average kernel time : 10.4576 us
// Achieved performance: 25.6689 TFLOPs
// abs mean:      6.03036
// abs max:         37.75
// err mean:  1.19575e-06
// err max:        0.0625
// --------------------  [cuBLAS]  M=512 N=512 K=512  --------------------
// Average kernel time : 8.6944 us
// Achieved performance: 30.8745 TFLOPs
//   TK / cuBLAS ratio : 0.831395x

// ==================== N = 1024 ====================
// --------------------  [TK]  M=1024 N=1024 K=1024  --------------------
// Block size: 128x256
// Allocated device memory
// Initialised matrices on device
// Computed reference GEMM
// Average kernel time : 15.84 us
// Achieved performance: 135.573 TFLOPs
// abs mean:      8.49961
// abs max:          56.5
// err mean:  6.10005e-06
// err max:          0.25
// --------------------  [cuBLAS]  M=1024 N=1024 K=1024  --------------------
// Average kernel time : 8.5216 us
// Achieved performance: 252.005 TFLOPs
//   TK / cuBLAS ratio : 0.53798x

// ==================== N = 2048 ====================
// --------------------  [TK]  M=2048 N=2048 K=2048  --------------------
// Block size: 128x256
// Allocated device memory
// Initialised matrices on device
// Skipping reference GEMM (matrix too large for naive kernel)
// Average kernel time : 27.1424 us
// Achieved performance: 632.953 TFLOPs
// --------------------  [cuBLAS]  M=2048 N=2048 K=2048  --------------------
// Average kernel time : 25.904 us
// Achieved performance: 663.213 TFLOPs
//   TK / cuBLAS ratio : 0.954374x

// ==================== N = 4096 ====================
// --------------------  [TK]  M=4096 N=4096 K=4096  --------------------
// Block size: 128x256
// Allocated device memory
// Initialised matrices on device
// Skipping reference GEMM (matrix too large for naive kernel)
// Average kernel time : 180.458 us
// Achieved performance: 761.614 TFLOPs
// --------------------  [cuBLAS]  M=4096 N=4096 K=4096  --------------------
// Average kernel time : 173.549 us
// Achieved performance: 791.933 TFLOPs
//   TK / cuBLAS ratio : 0.961715x

// ==================== N = 8192 ====================
// --------------------  [TK]  M=8192 N=8192 K=8192  --------------------
// Block size: 128x256
// Allocated device memory
// Initialised matrices on device
// Skipping reference GEMM (matrix too large for naive kernel)
// Average kernel time : 1405.56 us
// Achieved performance: 782.256 TFLOPs
// --------------------  [cuBLAS]  M=8192 N=8192 K=8192  --------------------
// Average kernel time : 1378.2 us
// Achieved performance: 797.785 TFLOPs
//   TK / cuBLAS ratio : 0.980535x
// root@test:~/CUDA_Kernels/TK_kernels# 