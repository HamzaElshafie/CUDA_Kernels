import torch
import triton
import triton.language as tl
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Initialise matrices
size = 1 << 12
x = torch.rand((size, size), device=DEVICE)
y = torch.rand((size, size), device=DEVICE)
x_cpu = torch.rand((size, size), device="cpu")
y_cpu = torch.rand((size, size), device="cpu")

@triton.jit
def matrix_add_kernel(
    x_ptr, y_ptr, output_ptr, 
    M, N, # num_rows, num_columns
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, # block_size_rows
    BLOCK_SIZE_N: tl.constexpr # block_size_columns
    ):
  pass

def matrix_add(x, y):
  output = torch.empty_like(x)

  assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE, "Tensors must be on CUDA"
  assert x.shape == y.shape and x.shape == output.shape, "Tensors must have identical dimension"

  M, N = output.shape

  stride_xm, stride_xn = x.stride()
  stride_ym, stride_yn = y.stride()
  stride_om, stride_on = output.stride()

  grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), 
                       triton.cdiv(N, meta["BLOCK_SIZE_N"]))
  
  matrix_add_kernel[grid](
      x, y, output, 
      M, N,
      stride_xm, stride_xn,
      stride_ym, stride_yn,
      stride_om, stride_on,
      BLOCK_SIZE_M=32,
      BLOCK_SIZE_N=32
      )

  return output

# Warm up and cache kernel

# Measure Triton execution time

# Measure Pytorch GPU execution time

# Measure Pytorch CPU execution time