#include "kittens.cuh"
#include "common.cuh"

#include <iostream>
#include <cuda_bf16.h>

using namespace kittens;

__device__ __forceinline__ bf16 fp32_to_bf16(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float bf16_to_fp32(bf16 x) { return __bfloat162float(x); }

static constexpr int NUM_WORKERS    = 4;  // warps per block
static constexpr int GROUPS_PER_TILE = 4;
static constexpr int NUM_THREADS    = NUM_WORKERS * kittens::WARP_THREADS;

template<int _d_model>
struct norm_args {
    static constexpr int d_model = _d_model;

    // Three possible ways to represent a 1xD object. LayerNorm fundamentally
    // operates on a token vector, not on a 2D matrix tile, so TK's shared
    // vector abstraction is the most natural fit here.
    using vec_smem_1xD  = sv_bf<d_model>;
    using tile_smem_1xD = st_bf<1, d_model>;
    using tile_reg_1xD  = rt_bf<1, d_model>;

    // Global descriptors
    using x_gl            = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using residual_gl     = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using out_gl          = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using out_residual_gl = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using norm_weight_gl  = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;
    using norm_bias_gl    = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;

    // Global descriptors are views over HBM tensors rather than storage objects.
    x_gl x;
    residual_gl residual;
    out_gl out;
    out_residual_gl out_residual;
    norm_weight_gl norm_weight;
    norm_bias_gl norm_bias;

    const int n_tile_size;
    const int n_per_tile;
    const float eps;
};

// grid.y  — batch dimension.
// grid.x  — sequence tiles; each block covers n_per_tile = GROUPS_PER_TILE * NUM_WORKERS tokens.
// Each block has NUM_WORKERS warps; each warp owns one token slot in shared memory.
// Within a block, GROUPS_PER_TILE groups of NUM_WORKERS tokens are processed in a double-buffered loop:
//   - while warp computes on the tic buffer, the toc buffer is prefetched from HBM.
//   - load_async_wait<1>() waits for tic only, leaving toc in flight.
template<int d_model>
__global__ void __launch_bounds__(NUM_THREADS)
layer_norm_tk(const __grid_constant__ norm_args<d_model> g) {
    // batch, token pos, warpid, laneid, sequence starting point
    auto warp_id = kittens::warpid();
    auto lane = kittens::laneid();

    int batch = blockIdx.y;
    // Each block covers n_per_tile = GROUPS_PER_TILE * NUM_WORKERS tokens.
    int seq_start = blockIdx.x * g.n_per_tile;

    // Allocate dynamic smem. Will hold staging of x, residual, bias, weight
    extern __shared__ alignment_dummy smem[];
    shared_allocator al((int*)&smem[0]);

    using args_t = norm_args<d_model>;
    using vec_smem_1xD = typename args_t::vec_smem_1xD;
    using tile_smem_1xD = typename args_t::tile_smem_1xD;
    using tile_reg_1xD = typename args_t::tile_reg_1xD;

    // We will allocate smem space for each stage and for each warp each of size the vector of shape d_model
    vec_smem_1xD (&x_s)[2][NUM_WORKERS] = al.allocate<vec_smem_1xD, 2, NUM_WORKERS>();
    vec_smem_1xD (&res_s)[2][NUM_WORKERS] = al.allocate<vec_smem_1xD, 2, NUM_WORKERS>();
    // Shared across the whole block
    vec_smem_1xD (&norm_weight_s) = al.allocate<vec_smem_1xD>();
    vec_smem_1xD (&norm_bias_s) = al.allocate<vec_smem_1xD>();

    // Double buffering
    int tic = 0, toc = 1;

    // Load norm weight and bias vectors by only one of the WORKERS since they will be shared across block
    if (warp_id == 0) {
        warp::load(norm_bias_s, g.norm_bias, {0,0,0,0});
        warp::load(norm_weight_s, g.norm_weight, {0,0,0,0});
    }
    
    // aync token loads 
    warp::load_async(x_s[tic][warp_id], g.x, {batch, 0, seq_start + warp_id, 0});
    warp::load_async(res_s[tic][warp_id], g.residual, {batch, 0, seq_start + warp_id, 0});
    __syncthreads();

    int n_blocks = g.n_per_tile / NUM_WORKERS;
    for (int block = 0; block < n_blocks; block++, tic^=1, toc^=1) {
        int curr_idx = block * NUM_WORKERS + warp_id;
        int next_idx = (block + 1) * NUM_WORKERS + warp_id;

        // Prefetch next group into the toc (inactive) buffer while we compute on tic.
        if (block < n_blocks - 1) {
            warp::load_async(x_s[toc][warp_id], g.x, {batch, 0, seq_start + next_idx, 0});
            warp::load_async(res_s[toc][warp_id], g.residual, {batch, 0, seq_start + next_idx, 0});
        }
        // Wait for the current (tic) buffer only
        load_async_wait<1>();
        __syncwarp();

        // Compute phase
        warp::add(res_s[tic][warp_id], res_s[tic][warp_id], x_s[tic][warp_id]);
        warp::store(g.out_residual, res_s[tic][warp_id], {batch, 0, seq_start+curr_idx, 0});
        __syncwarp();

        bf16 mean = fp32_to_bf16(0.0f);
        bf16 var  = fp32_to_bf16(0.0f);

        // reductions
        warp::sum(mean, res_s[tic][warp_id]);
        mean = mean / fp32_to_bf16(d_model);
        warp::sub(res_s[tic][warp_id], res_s[tic][warp_id], mean);
        warp::mul(x_s[tic][warp_id], res_s[tic][warp_id], res_s[tic][warp_id]);
        warp::sum(var, x_s[tic][warp_id]);
        var = var / fp32_to_bf16(d_model);
        var = fp32_to_bf16(sqrt(bf16_to_fp32(var + fp32_to_bf16(g.eps))));

        // Compute norm
        warp::div(res_s[tic][warp_id], res_s[tic][warp_id], var);
        warp::mul(res_s[tic][warp_id], res_s[tic][warp_id], norm_weight_s);
        warp::add(res_s[tic][warp_id], res_s[tic][warp_id], norm_bias_s);
        __syncwarp();

        // Write output back to gmem
        warp::store(g.out, res_s[tic][warp_id], {batch, 0, seq_start+curr_idx, 0});
    }
}

void dispatch_layernorm(
    bf16 *d_x_bf,
    bf16 *d_residual_bf,
    bf16 *d_norm_weight_bf,
    bf16 *d_norm_bias_bf,
    bf16 *d_o,
    bf16 *d_o_resid,
    size_t B,
    size_t N
) {
    constexpr int D = 1024;

    using args_t = norm_args<D>;
    using x_gl = typename args_t::x_gl;
    using residual_gl = typename args_t::residual_gl;
    using out_gl = typename args_t::out_gl;
    using out_residual_gl = typename args_t::out_residual_gl;
    using norm_weight_gl = typename args_t::norm_weight_gl;
    using norm_bias_gl = typename args_t::norm_bias_gl;

    x_gl x_arg{d_x_bf, B, 1, N, D};
    residual_gl residual_arg{d_residual_bf, B, 1, N, D};
    out_gl out_arg{d_o, B, 1, N, D};
    out_residual_gl out_residual_arg{d_o_resid, B, 1, N, D};
    norm_weight_gl norm_weight_arg{d_norm_weight_bf, 1, 1, 1, D};
    norm_bias_gl norm_bias_arg{d_norm_bias_bf, 1, 1, 1, D};

    // Each block covers GROUPS_PER_TILE groups of NUM_WORKERS tokens = GROUPS_PER_TILE * NUM_WORKERS tokens.
    constexpr int n_per_tile = GROUPS_PER_TILE * NUM_WORKERS;
    const int n_tile_size = static_cast<int>(N / n_per_tile);
    args_t g{x_arg, residual_arg, out_arg, out_residual_arg, norm_weight_arg, norm_bias_arg, n_tile_size, n_per_tile, 1e-5f};

    constexpr unsigned long mem_size = 36864;
    cudaError_t attr_err = cudaFuncSetAttribute(
        layer_norm_tk<D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    if (attr_err != cudaSuccess) {
        std::cerr << "cudaFuncSetAttribute failed: " << cudaGetErrorString(attr_err) << std::endl;
        return;
    }

    dim3 grid(n_tile_size, static_cast<unsigned int>(B), 1);
    layer_norm_tk<D><<<grid, NUM_THREADS, mem_size>>>(g);
}

#ifdef TK_COMPILE_FUSED_LAYERNORM
#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor> fused_layernorm(
    const at::Tensor x,
    const at::Tensor residual,
    const at::Tensor norm_weight,
    const at::Tensor norm_bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(norm_weight.is_cuda(), "norm_weight must be a CUDA tensor");
    TORCH_CHECK(norm_bias.is_cuda(), "norm_bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be bfloat16");
    TORCH_CHECK(norm_bias.scalar_type() == at::kBFloat16, "norm_bias must be bfloat16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(norm_weight.is_contiguous(), "norm_weight must be contiguous");
    TORCH_CHECK(norm_bias.is_contiguous(), "norm_bias must be contiguous");

    const int64_t b = x.size(0);
    const int64_t n = x.size(1);
    constexpr int64_t d = 1024;

    TORCH_CHECK(x.dim() == 3, "x must have shape [B, N, D]");
    TORCH_CHECK(residual.dim() == 3, "residual must have shape [B, N, D]");
    TORCH_CHECK(norm_weight.dim() == 1, "norm_weight must have shape [D]");
    TORCH_CHECK(norm_bias.dim() == 1, "norm_bias must have shape [D]");
    TORCH_CHECK(b == residual.size(0), "B mismatch between x and residual");
    TORCH_CHECK(n == residual.size(1), "N mismatch between x and residual");
    TORCH_CHECK(x.size(2) == d, "x last dim must be 1024");
    TORCH_CHECK(residual.size(2) == d, "residual last dim must be 1024");
    TORCH_CHECK(norm_weight.size(0) == d, "norm_weight size must be 1024");
    TORCH_CHECK(norm_bias.size(0) == d, "norm_bias size must be 1024");
    constexpr int64_t n_per_tile_check = GROUPS_PER_TILE * NUM_WORKERS;
    TORCH_CHECK((n % n_per_tile_check) == 0, "N must be divisible by GROUPS_PER_TILE * NUM_WORKERS (", n_per_tile_check, ")");

    at::Tensor out = at::empty({b, n, d}, x.options());
    at::Tensor out_resid = at::empty({b, n, d}, x.options());

    bf16 *d_x_bf = reinterpret_cast<bf16 *>(x.data_ptr<c10::BFloat16>());
    bf16 *d_residual_bf = reinterpret_cast<bf16 *>(residual.data_ptr<c10::BFloat16>());
    bf16 *d_norm_weight_bf = reinterpret_cast<bf16 *>(norm_weight.data_ptr<c10::BFloat16>());
    bf16 *d_norm_bias_bf = reinterpret_cast<bf16 *>(norm_bias.data_ptr<c10::BFloat16>());
    bf16 *d_o = reinterpret_cast<bf16 *>(out.data_ptr<c10::BFloat16>());
    bf16 *d_o_resid = reinterpret_cast<bf16 *>(out_resid.data_ptr<c10::BFloat16>());

    dispatch_layernorm(
        d_x_bf, d_residual_bf,
        d_norm_weight_bf, d_norm_bias_bf,
        d_o, d_o_resid,
        static_cast<size_t>(b), static_cast<size_t>(n)
    );

    cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(launch_err));

    return std::make_tuple(out, out_resid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_layernorm",
        &fused_layernorm,
        "Fused LayerNorm TK (x, residual, norm_weight, norm_bias) -> (out, out_resid)"
    );
}
#else
#include "harness.impl"
#endif