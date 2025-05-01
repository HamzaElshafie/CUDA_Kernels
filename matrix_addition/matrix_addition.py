import torch
import triton
import triton.language as tl
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Initialise matrices
size = 1 << 13
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
  
  pid_m = tl.program_id(0) # x
  pid_n = tl.program_id(1) # y

  row_start = pid_m * BLOCK_SIZE_M
  column_start = pid_n * BLOCK_SIZE_N

  rows = row_start + tl.arange(0, BLOCK_SIZE_M)
  columns = column_start + tl.arange(0, BLOCK_SIZE_N)

  # Compute flat memory offsets for each (row, col) pair in the tile
  # using broadcasting to generate the full 2D grid of indices
  offsets_x = rows.expand_dims(1) * stride_xm + columns.expand_dims(0) * stride_xn
  offsets_y = rows.expand_dims(1) * stride_ym + columns.expand_dims(0) * stride_yn
  offsets_o = rows.expand_dims(1) * stride_om + columns.expand_dims(0) * stride_on

  # Allow access to elements where both row and column indices are in bounds
  mask = (rows.expand_dims(1) < M) & (columns.expand_dims(0) < N)

  # Load elements of the tiles from DRAM, masking out out-of-bound elements with 0.0
  x_tile = tl.load(x_ptr + offsets_x, mask=mask, other=0.0)
  y_tile = tl.load(y_ptr + offsets_y, mask=mask, other=0.0)

  output_tile = x_tile + y_tile

  # Write result back to DRAM
  tl.store(output_ptr + offsets_o, output_tile, mask=mask)


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
_ = matrix_add(x, y)

# Measure Triton execution time
torch.cuda.synchronize()
start = time.perf_counter()
output_triton = matrix_add(x, y)
torch.cuda.synchronize() 
end = time.perf_counter()
triton_time = (end - start) * 1000

# Measure PyTorch GPU execution time
torch.cuda.synchronize()
start = time.perf_counter()
output_torch_gpu = x + y
torch.cuda.synchronize() 
end = time.perf_counter()
pytorch_gpu_time = (end - start) * 1000

# Measure Pytorch CPU execution time
start = time.perf_counter()
output_torch_cpu = x_cpu + y_cpu
end = time.perf_counter()
pytorch_cpu_time = (end - start) * 1000

# Check correctness
max_diff = torch.max(torch.abs(output_triton - output_torch_gpu))
assert torch.allclose(output_triton, output_torch_gpu, atol=1e-5), "Mismatch with PyTorch!"

print(f"Triton time:       {triton_time:.3f} ms")
print(f"PyTorch GPU time:  {pytorch_gpu_time:.3f} ms")
print(f"PyTorch CPU time:  {pytorch_cpu_time:.3f} ms")
print(f"Max absolute diff: {max_diff.item():.6e}")


# cuda:0
# Triton time:       3.964 ms
# PyTorch GPU time:  3.548 ms
# PyTorch CPU time:  42.756 ms
# The maximum difference between torch and triton is 0.0