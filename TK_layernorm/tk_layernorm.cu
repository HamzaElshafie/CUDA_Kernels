#include "kittens.cuh"
#include "common.cuh"

#include <iostream>
#include <cuda_bf16.h>

using namespace kittens;

template<int _d_model>
struct norm_args {
    static constexpr int d_model = _d_model;
    static constexpr float dropout_p = 0.0f; // FIX: Possible remove from here and just keep as runtime arg

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
};