import argparse
import os
from pathlib import Path
from typing import Callable

import torch
from torch.utils.cpp_extension import load

from baselines import (
    pytorch_fused_residual_rms_norm_fwd,
    triton_fused_residual_rms_norm_fwd,
)


def _load_tk_extension():
    this_dir = Path(__file__).resolve().parent
    source = this_dir / "tk_rms.cu"
    tk_kernels_dir = this_dir.parent

    include_paths = [str(tk_kernels_dir)]
    tk_root = os.environ.get("THUNDERKITTENS_ROOT", "")
    if tk_root:
        include_paths.append(str(Path(tk_root) / "include"))
        include_paths.append(tk_root)
        prototype_dir = Path(tk_root) / "prototype"
        if (prototype_dir / "prototype.cuh").exists():
            include_paths.append(str(prototype_dir))

    extra_cuda_cflags = [
        "-DTK_COMPILE_FUSED_RMSNORM",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        "-DKITTENS_HOPPER",
    ]

    try:
        return load(
            name="tk_rmsnorm_ext",
            sources=[str(source)],
            extra_include_paths=include_paths,
            extra_cflags=["-std=c++20", "-DKITTENS_HOPPER"],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=["-lcuda"],
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
    # Residual + mean square + rsqrt + scale + weight: ~7 FLOPs/elem (rough).
    flops = 7.0 * b * n * d
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
    d = 1024
    seq_lens = [1024, 2048, 4096, 8192, 16384]
    eps = 1e-5
    results = {"pytorch": [], "triton": [], "tk": []}

    for n in seq_lens:
        print(f"\nB={batch}, N={n}, D={d}")
        x = torch.randn(batch, n, d, device=device, dtype=dtype)
        residual = torch.randn_like(x)
        weight = torch.randn(d, device=device, dtype=dtype)

        ref_y, ref_r = pytorch_fused_residual_rms_norm_fwd(x, residual, weight, eps=eps)

        tri_y, tri_r = triton_fused_residual_rms_norm_fwd(x, residual, weight, eps=eps)
        tri_ok_y = _check_close("triton:y", tri_y, ref_y)
        tri_ok_r = _check_close("triton:resid", tri_r, ref_r)

        tk_ok_y = False
        tk_ok_r = False
        tk_available = tk_mod is not None
        if tk_available:
            tk_y, tk_r = tk_mod.fused_rmsnorm(x, residual, weight)
            tk_ok_y = _check_close("tk:y", tk_y, ref_y, atol=2e-1, rtol=2e-1)
            tk_ok_r = _check_close("tk:resid", tk_r, ref_r)
        else:
            print("  tk              skipped (extension unavailable)")

        pt_ms = _time_cuda_ms(
            lambda: pytorch_fused_residual_rms_norm_fwd(x, residual, weight, eps=eps),
            warmup=args.warmup,
            iters=args.iters,
        )
        tri_ms = _time_cuda_ms(
            lambda: triton_fused_residual_rms_norm_fwd(x, residual, weight, eps=eps),
            warmup=args.warmup,
            iters=args.iters,
        )
        if tk_available and tk_ok_y and tk_ok_r:
            tk_ms = _time_cuda_ms(
                lambda: tk_mod.fused_rmsnorm(x, residual, weight),
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
            f"pytorch={pt_tflops:.3f}, triton={tri_tflops:.3f}, "
            f"tk={tk_tflops if tk_tflops == tk_tflops else 'nan'}"
        )

        results["pytorch"].append(pt_tflops)
        results["triton"].append(tri_tflops)
        results["tk"].append(tk_tflops)

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(seq_lens, results["pytorch"], marker="o", label="PyTorch")
        ax.plot(seq_lens, results["triton"], marker="o", label="Triton")
        ax.plot(seq_lens, results["tk"], marker="o", label="TK")
        ax.set_xticks(seq_lens)
        ax.set_xticklabels([str(s) for s in seq_lens])
        ax.set_title(f"Fused Residual+RMSNorm Throughput (D={d})")
        ax.set_xlabel("Sequence length (N)")
        ax.set_ylabel("Throughput (TFLOP/s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = Path(args.plot_path)
        fig.savefig(out_path, dpi=180)
        print(f"\nSaved plot to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs Triton vs TK fused RMSNorm forward.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-tk", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", type=str, default="benchmark_rmsnorm.png")
    run(parser.parse_args())
