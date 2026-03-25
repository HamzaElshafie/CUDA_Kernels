#include "kittens.cuh"
#include "common.cuh"

#include <iostream>
#include <cuda_bf16.h>

using namespace kittens;

__device__ __forceinline__ bf16 fp32_to_bf16(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float bf16_to_fp32(bf16 x) { return __bfloat162float(x); }

static constexpr int NUM_WORKERS = 2;
static constexpr int NUM_THREADS = NUM_WORKERS * kittens::WARP_THREADS;

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

// Kernel is designed around exactly two warps per block.
// grid.y moves across batch elements.
// grid.x moves across sequence chunks, where each block handles two tokens.
// Each block has two worker warps, and each warp handles one token vector across its full d_model dimension.
template<int d_model>
__global__ void __launch_bounds__(NUM_THREADS)
layer_norm_tk(const __grid_constant__ norm_args<d_model> g) {
    // batch, token pos, warpid, laneid, sequence starting point
    auto warp_id = kittens::warpid();
    auto lane = kittens::laneid();

    int batch = blockIdx.y;
    int seq_start = blockIdx.x * NUM_WORKERS;

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

        // Prefetch next token pair
        if (block < n_blocks - 1) {
            warp::load_async(x_s[toc][warp_id], g.x, {batch, 0, seq_start + next_idx, 0});
            warp::load_async(res_s[toc][warp_id], g.residual, {batch, 0, seq_start + next_idx, 0});
        }
        load_async_wait();
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