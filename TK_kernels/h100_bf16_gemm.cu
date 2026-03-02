#include "kittens.cuh"
#include "prototype.cuh"
#include "../common.cuh"

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
    // Returns grid dimensions. If PERISISTENT_GRID is true, launch one block per H100 SM (132) and 
    // let blocks loop over tiles via task_iter; otherwise launch one block per C tile group (M_BLOCK×N_BLOCK base tiles).
    template<bool PERISISTENT_GRID=true>
    __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : (M * N) / (M_BLOCK * N_BLOCK * layout::base_tile::num_elements));
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
        // Will do tmrw
    }
}