import torch
import triton
import triton.language as tl
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

@triton.jit
def matmul_naive_kernel(
  # Pointers to matrices
  a_ptr, b_ptr, c_ptr,
  # Matrix dimensions
  M, N, K,
  # Matrix strides
  stride_am, stride_ak,
  stride_bk, stride_bn,
  stride_cm, stride_cn,
  # Block sizes
  BLOCK_SIZE_M: tl.constexpr,
  BLOCK_SIZE_N: tl.constexpr,
):
  # Program ID
  pid_m = tl.program_id(0)  # row
  pid_n = tl.program_id(1)  # column
  
  # Calculate starting indices
  row_start = pid_m * BLOCK_SIZE_M
  col_start = pid_n * BLOCK_SIZE_N
  
  # Generate row and column indices for this block
  rows = row_start + tl.arange(0, BLOCK_SIZE_M)
  cols = col_start + tl.arange(0, BLOCK_SIZE_N)
  
  # Create masks for boundary checking
  row_mask = rows < M
  col_mask = cols < N
  mask = row_mask[:, None] & col_mask[None, :]
  
  # Initialize accumulator for results
  c_values = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
  
  # Compute output indices (only need to do this once)
  c_index = rows[:, None] * stride_cm + cols[None, :] * stride_cn
  
  # Naive matrix multiplication - similar to your CUDA implementation
  for k in range(K):
      # Load one column of A (for current k)
      a_index = rows * stride_am + k * stride_ak
      a_values = tl.load(a_ptr + a_index, mask=row_mask, other=0.0)
      
      # Load one row of B (for current k)
      b_index = k * stride_bk + cols * stride_bn
      b_values = tl.load(b_ptr + b_index, mask=col_mask, other=0.0)
      
      # Compute the outer product for this 'k' and accumulate
      # Use proper broadcasting to keep dimensions consistent
      c_values += (a_values[:, None] * b_values[None, :])
  
  # Write results to memory
  tl.store(c_ptr + c_index, c_values, mask=mask)

def matmul_naive(a, b):
  # Check constraints
  assert a.shape[1] == b.shape[0], "Incompatible dimensions"
  M, K = a.shape
  K, N = b.shape
  
  # Allocate output
  c = torch.empty((M, N), device=a.device, dtype=a.dtype)
  
  # Launch kernel with 2D grid
  grid = lambda meta: (
      triton.cdiv(M, meta["BLOCK_SIZE_M"]),
      triton.cdiv(N, meta["BLOCK_SIZE_N"]),
  )
  
  matmul_naive_kernel[grid](
      a, b, c,
      M, N, K,
      a.stride(0), a.stride(1),
      b.stride(0), b.stride(1),
      c.stride(0), c.stride(1),
      BLOCK_SIZE_M=32,
      BLOCK_SIZE_N=32,
  )
  
  return c

# Test the kernel
def test_matmul():
  # Create test matrices
  M, K, N = 256, 256, 1024  # Same as your CUDA example
  a = torch.rand((M, K), device=DEVICE, dtype=torch.float32)
  b = torch.rand((K, N), device=DEVICE, dtype=torch.float32)
  
  # Warm up and cache kernel
  _ = matmul_naive(a, b)
  
  # Measure Triton execution time
  torch.cuda.synchronize()
  start = time.perf_counter()
  c_triton = matmul_naive(a, b)
  torch.cuda.synchronize()
  end = time.perf_counter()
  triton_time = (end - start) * 1000
  
  # Measure PyTorch execution time
  torch.cuda.synchronize()
  start = time.perf_counter()
  c_torch = torch.matmul(a, b)
  torch.cuda.synchronize()
  end = time.perf_counter()
  pytorch_time = (end - start) * 1000
  
  # Check correctness
  assert torch.allclose(c_triton, c_torch, atol=1e-5), "Results don't match!"
  
  print(f"Triton naive matmul time: {triton_time:.3f} ms")
  print(f"PyTorch matmul time: {pytorch_time:.3f} ms")
  print(f"Max absolute difference: {torch.max(torch.abs(c_triton - c_torch)).item():.6e}")


test_matmul()