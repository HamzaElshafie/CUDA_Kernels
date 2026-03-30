"""
PyTorch reference and Triton baseline for fused residual + RMSNorm (forward).
Triton reuses the LayerNorm directory's _layer_norm_fwd (is_rms_norm=True).
"""
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_layernorm_baselines():
    ln_path = Path(__file__).resolve().parent.parent / "TK_layernorm" / "baselines.py"
    if not ln_path.is_file():
        raise FileNotFoundError(f"Need TK_layernorm/baselines.py for Triton baseline: {ln_path}")
    spec = importlib.util.spec_from_file_location("tk_ln_baselines", ln_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {ln_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pytorch_fused_residual_rms_norm_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
):
    """
    r = x + residual
    y = RMSNorm(r; weight) — no bias.
    Returns (y, r).
    """
    r = x + residual
    if hasattr(F, "rms_norm"):
        y = F.rms_norm(r, (r.shape[-1],), weight, eps)
    else:
        rw = r.float()
        var = rw.pow(2).mean(dim=-1, keepdim=True)
        y = (rw * torch.rsqrt(var + eps) * weight.float()).to(dtype=r.dtype)
    return y, r


def triton_fused_residual_rms_norm_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
):
    """Triton via TK_layernorm's _layer_norm_fwd with is_rms_norm=True, bias=None."""
    m = _load_layernorm_baselines()
    orig_shape = x.shape
    x2d = x.reshape(-1, x.shape[-1])
    res2d = residual.reshape(-1, residual.shape[-1])
    y, _, _, _, r_out, _, _, _ = m._layer_norm_fwd(
        x2d,
        weight,
        None,
        eps,
        residual=res2d,
        is_rms_norm=True,
    )
    return y.reshape(orig_shape), r_out.reshape(orig_shape)
