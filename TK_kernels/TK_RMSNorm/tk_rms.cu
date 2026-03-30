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
    using norm_weight = gl<bf16, -1, -1, -1, -1, vec_smem_1xD>;

    // Struct args
    x_gl x;
    residual_gl residual;
    out_gl out;
    norm_weight norm_weight;

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
    __syncthreads();
    
    // Compute phase (per token)
    // 
    warp::add(res_s[warp_id], res_s[warp_id], x_s[warp_id]);
    __syncwarp();

    bf16 norm_factor = fp32_to_bf16(0.0f);

    // Reductions
    warp::mul(x_s[warp_id], res_s[warp_id], res_s[warp_id]);
    warp::sum(norm_factor, x_s[warp_id]);
    norm_factor = norm_factor / fp32_to_bf16(d_model);
    norm_factor = fp32_to_bf16(sqrt(bf16_to_fp32(norm_factor + fp32_to_bf16(g.eps))));

    warp::div(res_s[warp_id], res_s[warp_id], l);
    warp::mul(res_s[warp_id], res_s[warp_id], norm_weight_s);
    __syncwarp();

    // Store phase (per token)
    warp::store(g.out, res_s[warp_id], {batch,0,seq_start + warp_id,0});
}

