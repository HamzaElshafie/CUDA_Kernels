import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def pytorch_fused_residual_layer_norm_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
):
    """
    PyTorch reference that matches the TK/Triton fused forward:
      r = x + residual
      y = LayerNorm(r; weight, bias)
    Returns (y, r).
    """
    r = x + residual
    y = F.layer_norm(r, (r.shape[-1],), weight=weight, bias=bias, eps=eps)
    return y, r


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["N", "HAS_BIAS"],
)
@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.jit
def _fused_residual_layer_norm_fwd_kernel(
    X,              # [M, N]
    RESIDUAL,       # [M, N]
    W,              # [N]
    B,              # [N] or None
    Y_RESIDUAL,     # [M, N]
    Y,              # [M, N]
    MEAN,           # [M]
    RSTD,           # [M]
    stride_x_row,
    stride_res_row,
    stride_y_res_row,
    stride_y_row,
    M,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x_ptr = X + row * stride_x_row + cols
    residual_ptr = RESIDUAL + row * stride_res_row + cols
    y_residual_ptr = Y_RESIDUAL + row * stride_y_res_row + cols
    y_ptr = Y + row * stride_y_row + cols

    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr, mask=mask, other=0.0).to(tl.float32)
    r = x + residual

    tl.store(y_residual_ptr, r, mask=mask)

    mean = tl.sum(r, axis=0) / N
    tl.store(MEAN + row, mean)
    r_centered = tl.where(mask, r - mean, 0.0)
    var = tl.sum(r_centered * r_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(RSTD + row, rstd)

    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
        y = r_centered * rstd * w + b
    else:
        y = r_centered * rstd * w
    tl.store(y_ptr, y, mask=mask)


def triton_fused_residual_layer_norm_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
    return_stats: bool = False,
):
    """
    Triton forward-only fused residual + LayerNorm:
      r = x + residual
      y = LayerNorm(r; weight, bias)
    Returns (y, y_residual), or (y, y_residual, mean, rstd) if return_stats=True.
    """
    assert x.is_cuda and residual.is_cuda and weight.is_cuda, "All tensors must be CUDA tensors"
    assert x.ndim >= 2, "x must have at least 2 dims"
    assert x.shape == residual.shape, "x and residual must have same shape"
    assert x.stride(-1) == 1 and residual.stride(-1) == 1, "Last dim must be contiguous"

    orig_shape = x.shape
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    assert weight.shape == (N,), "weight must have shape [N]"
    assert weight.stride(-1) == 1, "weight must be contiguous"
    if bias is not None:
        assert bias.shape == (N,), "bias must have shape [N]"
        assert bias.stride(-1) == 1, "bias must be contiguous"
        assert bias.is_cuda, "bias must be a CUDA tensor"

    x2d = x.reshape(M, N)
    residual2d = residual.reshape(M, N)

    y = torch.empty_like(x2d)
    y_residual = torch.empty_like(x2d)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

    max_fused_size = 65536 // x.element_size()
    block_n = min(max_fused_size, triton.next_power_of_2(N))
    if N > block_n:
        raise RuntimeError("This fused LayerNorm only supports feature dim < 64KB")

    with torch.cuda.device(x.device):
        _fused_residual_layer_norm_fwd_kernel[(M,)](
            x2d,
            residual2d,
            weight,
            bias,
            y_residual,
            y,
            mean,
            rstd,
            x2d.stride(0),
            residual2d.stride(0),
            y_residual.stride(0),
            y.stride(0),
            M,
            N,
            eps,
            BLOCK_N=block_n,
        )

    y = y.reshape(orig_shape)
    y_residual = y_residual.reshape(orig_shape)
    if return_stats:
        return y, y_residual, mean, rstd
    return y, y_residual


class FusedResidualLayerNormTriton(torch.nn.Module):
    def __init__(self, hidden_size: int = 1024, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, residual: torch.Tensor):
        return triton_fused_residual_layer_norm_fwd(
            x=x,
            residual=residual,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
            return_stats=False,
        )
