import argparse
import os
from pathlib import Path
from typing import Callable

import torch
from torch.utils.cpp_extension import load

from baselines import (
    pytorch_fused_residual_layer_norm_fwd,
    triton_fused_residual_layer_norm_fwd,
)


def _load_tk_extension():
    """
    Build/load the TK fused layernorm extension from tk_layernorm.cu.
    Returns loaded module, or None if build fails.
    """
    this_dir = Path(__file__).resolve().parent
    source = this_dir / "tk_layernorm.cu"
    tk_kernels_dir = this_dir.parent  # contains common.cuh

    include_paths = [str(tk_kernels_dir)]
    tk_root = os.environ.get("THUNDERKITTENS_ROOT", "")
    if tk_root:
        include_paths.append(str(Path(tk_root) / "include"))
        include_paths.append(tk_root)

    extra_cuda_cflags = [
        "-DTK_COMPILE_FUSED_LAYERNORM",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-DKITTENS_HOPPER",
    ]

    try:
        return load(
            name="tk_layernorm_ext",
            sources=[str(source)],
            extra_include_paths=include_paths,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=False,
        )
    except Exception as exc:
        print(f"[WARN] Could not build/load TK extension: {exc}")
        return None


def _time_cuda_ms(fn: Callable[[], None], warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _throughput_tflops(b: int, n: int, d: int, ms: float) -> float:
    # Approximate fused residual + layernorm forward FLOPs per element.
    # Residual add + reductions + normalise + affine ~= 9 FLOPs/elem.
    flops = 9.0 * b * n * d
    return flops / (ms * 1e-3) / 1e12


def _check_close(name: str, out, ref, atol=2e-2, rtol=2e-2):
    ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
    max_abs = (out - ref).abs().max().item()
    print(f"  {name:16s} allclose={ok} max_abs_err={max_abs:.6f}")
    return ok


def run(args):
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    tk_mod = _load_tk_extension() if args.with_tk else None
    if args.with_tk and tk_mod is None:
        print("[WARN] TK benchmark disabled because extension failed to load.")

    batch = 16
    seq_lens = [1024, 2048, 4096, 8192, 16384]
    d_values = [1024, 2048]
    eps = 1e-5

    results = {
        1024: {"pytorch": [], "triton": [], "tk": []},
        2048: {"pytorch": [], "triton": [], "tk": []},
    }

    for d in d_values:
        print(f"\n=== D={d} ===")
        for n in seq_lens:
            print(f"B={batch}, N={n}, D={d}")
            x = torch.randn(batch, n, d, device=device, dtype=dtype)
            residual = torch.randn_like(x)
            weight = torch.randn(d, device=device, dtype=dtype)
            bias = torch.randn(d, device=device, dtype=dtype)

            ref_y, ref_r = pytorch_fused_residual_layer_norm_fwd(x, residual, weight, bias, eps=eps)

            tri_y, tri_r = triton_fused_residual_layer_norm_fwd(x, residual, weight, bias, eps=eps)
            tri_ok_y = _check_close("triton:y", tri_y, ref_y)
            tri_ok_r = _check_close("triton:resid", tri_r, ref_r)

            tk_ok_y = False
            tk_ok_r = False
            tk_available = tk_mod is not None and d == 1024
            if tk_available:
                tk_y, tk_r = tk_mod.fused_layernorm(x, residual, weight, bias)
                tk_ok_y = _check_close("tk:y", tk_y, ref_y)
                tk_ok_r = _check_close("tk:resid", tk_r, ref_r)
            else:
                print("  tk              skipped (extension unavailable or D != 1024)")

            pt_ms = _time_cuda_ms(
                lambda: pytorch_fused_residual_layer_norm_fwd(x, residual, weight, bias, eps=eps),
                warmup=args.warmup,
                iters=args.iters,
            )
            tri_ms = _time_cuda_ms(
                lambda: triton_fused_residual_layer_norm_fwd(x, residual, weight, bias, eps=eps),
                warmup=args.warmup,
                iters=args.iters,
            )
            if tk_available and tk_ok_y and tk_ok_r:
                tk_ms = _time_cuda_ms(
                    lambda: tk_mod.fused_layernorm(x, residual, weight, bias),
                    warmup=args.warmup,
                    iters=args.iters,
                )
            else:
                tk_ms = float("nan")

            pt_tflops = _throughput_tflops(batch, n, d, pt_ms)
            tri_tflops = _throughput_tflops(batch, n, d, tri_ms)
            tk_tflops = _throughput_tflops(batch, n, d, tk_ms) if tk_ms == tk_ms else float("nan")

            print(
                f"  throughput TFLOP/s -> "
                f"pytorch={pt_tflops:.3f}, triton={tri_tflops:.3f}, tk={tk_tflops if tk_tflops == tk_tflops else 'nan'}"
            )

            results[d]["pytorch"].append(pt_tflops)
            results[d]["triton"].append(tri_tflops)
            results[d]["tk"].append(tk_tflops)

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, d in zip(axes, d_values):
            ax.plot(seq_lens, results[d]["pytorch"], marker="o", label="PyTorch")
            ax.plot(seq_lens, results[d]["triton"], marker="o", label="Triton")
            ax.plot(seq_lens, results[d]["tk"], marker="o", label="TK")
            ax.set_title(f"Fused Residual+LayerNorm Throughput (D={d})")
            ax.set_xlabel("Sequence length (N)")
            ax.set_ylabel("Throughput (TFLOP/s)")
            ax.grid(True, alpha=0.3)
        axes[0].legend()
        fig.tight_layout()
        out_path = Path(args.plot_path)
        fig.savefig(out_path, dpi=180)
        print(f"\nSaved plot to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs Triton vs TK fused LayerNorm forward.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-tk", action="store_true", help="Attempt to compile and benchmark TK CUDA extension.")
    parser.add_argument("--plot", action="store_true", help="Save throughput line plot.")
    parser.add_argument("--plot-path", type=str, default="benchmark_layernorm.png")
    run(parser.parse_args())
