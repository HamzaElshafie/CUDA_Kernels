/**
Possible modifications to play around with
1. Use the tile_reg_1xD abstarction instead of vec_smem_1xD
2. Use TMA to load the vec_smem_1xD
3. Extend to use double buffering
*/
#include "kittens.cuh"
#include "common.cuh"

#include <iostream>
#include <cuda_bf16.h>

using namespace kittens;

// Precision conversion helpers
__device__ __forceinline__ bf16 fp32_to_bf16(float x) {return __float2bfloat16(x);}
__device__ __forceinline__ float bf16_to_fp32(bf16 x) {return __bfloat162float(x);}

// Define how many workers we will need per block and how many threads per block
static constexpr int NUM_WORKERS = 2;
static constexpr int NUM_THREADS = NUM_WORKERS * kittens::WARP_THREADS;

// Define the kernel args in a struct so we can pack together
template <int _d_model>
struct norm_args {
    static constexpr int d_model = _d_model;

    // Define the TK objects we might need in kernel
    using vec_smem_1xD = sv_bf<d_model>;
    using tile_smem_1xD = st_bf<1, d_model>;
    using tile_reg_1xD = rt_bf<1, d_model>;
    using vec_reg_1xD = rv_bf<d_model>;

    // Define the global tensor descriptor TK objects
    using x_gl = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using residual_gl = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using out_gl = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    // Note: type cannot be named `norm_weight` (would clash with the field name).
    using norm_weight_gl = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;

    // Struct args
    x_gl x;
    residual_gl residual;
    out_gl out;
    norm_weight_gl norm_weight;

    const int n_tile_size;
    const int n_per_tile;
    const float norm_eps;
};

// grid.y -> batch dimension.
// grid.x -> sequence tiles; each block covers n_per_tile = NUM_WORKERS tokens.
template<int d_model>
__global__ void __launch_bounds__(NUM_THREADS)
rms_norm_tk(const __grid_constant__ norm_args<d_model> g) {
    // Get thread's description
    auto warp_id = kittens::warpid();
    auto lane_id = kittens::laneid();
    int batch = blockIdx.y;
    int seq_start = NUM_WORKERS *  blockIdx.x;

    // Type alias the TK objects we will need (based on args_t)
    using args_t = norm_args<d_model>;
    using vec_smem_1xD = typename args_t::vec_smem_1xD;

    // Allocate the memory for the TK objects
    extern __shared__ alignment_dummy smem[];
    shared_allocator al((int*)&smem[0]);
    vec_smem_1xD (&x_s)[NUM_WORKERS] = al.allocate<vec_smem_1xD, NUM_WORKERS>();
    vec_smem_1xD (&res_s)[NUM_WORKERS] = al.allocate<vec_smem_1xD, NUM_WORKERS>();
    vec_smem_1xD (&norm_weight_s) = al.allocate<vec_smem_1xD>();
    
    // Load norm weight
    if (warp_id == 0) {
        warp::load(norm_weight_s, g.norm_weight, {0,0,0,0});
    }

    // Load phase (per token)
    warp::load_async(x_s[warp_id], g.x, {batch,0,seq_start + warp_id,0});
    warp::load_async(res_s[warp_id], g.residual, {batch,0,seq_start + warp_id,0});
    load_async_wait();
    __syncthreads();
    
    // Compute phase (per token)
    warp::add(res_s[warp_id], res_s[warp_id], x_s[warp_id]);
    __syncwarp();

    bf16 norm_factor = fp32_to_bf16(0.0f);

    // Reductions
    warp::mul(x_s[warp_id], res_s[warp_id], res_s[warp_id]);
    warp::sum(norm_factor, x_s[warp_id]);
    norm_factor = norm_factor / fp32_to_bf16(d_model);
    norm_factor = fp32_to_bf16(sqrt(bf16_to_fp32(norm_factor + fp32_to_bf16(g.norm_eps))));

    warp::div(res_s[warp_id], res_s[warp_id], norm_factor);
    warp::mul(res_s[warp_id], res_s[warp_id], norm_weight_s);
    __syncwarp();

    // Store phase (per token)
    warp::store(g.out, res_s[warp_id], {batch,0,seq_start + warp_id,0});
}

void dispatch_rmsnorm(
    bf16 *d_x_bf,
    bf16 *d_residual_bf,
    bf16 *d_norm_weight_bf,
    bf16 *d_o,
    size_t B,
    size_t N
) {
    constexpr int D = 1024;

    using args_t = norm_args<D>;
    using x_gl = typename args_t::x_gl;
    using residual_gl = typename args_t::residual_gl;
    using out_gl = typename args_t::out_gl;
    using norm_weight_gl = typename args_t::norm_weight_gl;

    x_gl x_arg{d_x_bf, B, 1, N, D};
    residual_gl residual_arg{d_residual_bf, B, 1, N, D};
    out_gl out_arg{d_o, B, 1, N, D};
    norm_weight_gl norm_weight_arg{d_norm_weight_bf, 1, 1, 1, D};

    constexpr int n_per_tile = NUM_WORKERS;
    const int n_tile_size = static_cast<int>(N / n_per_tile);
    args_t g{x_arg, residual_arg, out_arg, norm_weight_arg, n_tile_size, n_per_tile, 1e-5f};

    // x_s[NUM_WORKERS] + res_s[NUM_WORKERS] + norm_weight_s = (2*NUM_WORKERS+1) * sizeof(vec)
    constexpr unsigned long mem_size = (2 * NUM_WORKERS + 1) * (size_t)D * sizeof(bf16);

    cudaError_t attr_err = cudaFuncSetAttribute(
        rms_norm_tk<D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    if (attr_err != cudaSuccess) {
        std::cerr << "cudaFuncSetAttribute failed: " << cudaGetErrorString(attr_err) << std::endl;
        return;
    }

    dim3 grid(n_tile_size, static_cast<unsigned int>(B), 1);
    rms_norm_tk<D><<<grid, NUM_THREADS, mem_size>>>(g);
}

#ifdef TK_COMPILE_FUSED_RMSNORM
#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor> fused_rmsnorm(
    const at::Tensor x,
    const at::Tensor residual,
    const at::Tensor norm_weight
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(norm_weight.is_cuda(), "norm_weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be bfloat16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(norm_weight.is_contiguous(), "norm_weight must be contiguous");

    const int64_t b = x.size(0);
    const int64_t n = x.size(1);
    constexpr int64_t d = 1024;

    TORCH_CHECK(x.dim() == 3, "x must have shape [B, N, D]");
    TORCH_CHECK(residual.dim() == 3, "residual must have shape [B, N, D]");
    TORCH_CHECK(norm_weight.dim() == 1, "norm_weight must have shape [D]");
    TORCH_CHECK(b == residual.size(0), "B mismatch between x and residual");
    TORCH_CHECK(n == residual.size(1), "N mismatch between x and residual");
    TORCH_CHECK(x.size(2) == d, "x last dim must be 1024");
    TORCH_CHECK(residual.size(2) == d, "residual last dim must be 1024");
    TORCH_CHECK(norm_weight.size(0) == d, "norm_weight size must be 1024");
    TORCH_CHECK((n % NUM_WORKERS) == 0, "N must be divisible by NUM_WORKERS (2)");

    at::Tensor out = at::empty({b, n, d}, x.options());
    // Kernel fused the add internally but does not store r; match fused LayerNorm-style API.
    at::Tensor out_resid = at::empty({b, n, d}, x.options());
    at::add_out(out_resid, x, residual);

    bf16 *d_x_bf = reinterpret_cast<bf16 *>(x.data_ptr<c10::BFloat16>());
    bf16 *d_residual_bf = reinterpret_cast<bf16 *>(residual.data_ptr<c10::BFloat16>());
    bf16 *d_norm_weight_bf = reinterpret_cast<bf16 *>(norm_weight.data_ptr<c10::BFloat16>());
    bf16 *d_o = reinterpret_cast<bf16 *>(out.data_ptr<c10::BFloat16>());

    dispatch_rmsnorm(
        d_x_bf, d_residual_bf, d_norm_weight_bf, d_o,
        static_cast<size_t>(b), static_cast<size_t>(n)
    );

    cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(launch_err));

    return std::make_tuple(out, out_resid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_rmsnorm",
        &fused_rmsnorm,
        "Fused RMSNorm TK: r=x+residual in-kernel; returns (RMSNorm(r), r) with r from add_out."
    );
}
#else
#include "harness.impl"
#endif
