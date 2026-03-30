# ThunderKittens RMSNorm (H100)

Same environment as `TK_layernorm` (CMake ≥ 3.24, CUDA, ThunderKittens, optional Python/Triton for baselines).

## Build (standalone harness)

```bash
export CUDA_HOME=/usr/local/cuda-12.8   # or your toolkit path
export THUNDERKITTENS_ROOT=~/ThunderKittens

cd ~/CUDA_Kernels/TK_kernels/TK_RMSNorm
rm -rf build
cmake -S . -B build -DCUDAToolkit_ROOT=$CUDA_HOME -DTHUNDERKITTENS_ROOT=$THUNDERKITTENS_ROOT
cmake --build build -j
./build/tk_rms
```

## Python benchmarks (`baselines.py` uses `TK_layernorm/baselines.py` for Triton)

```bash
cd ~/CUDA_Kernels/TK_kernels/TK_RMSNorm
python3 -m pip install torch triton matplotlib  # if needed
python3 benchmark.py --with-tk
python3 benchmark.py --plot --with-tk
```

Triton baseline is loaded from the sibling `TK_layernorm` directory; keep that folder present.

## Kernel API (PyTorch extension)

- `fused_rmsnorm(x, residual, weight) -> (out, out_resid)`
- Shapes: `x`, `residual`: `[B, N, 1024]` bf16 contiguous; `weight`: `[1024]` bf16.
- `N` must be divisible by **2** (`NUM_WORKERS` in `tk_rms.cu`).
