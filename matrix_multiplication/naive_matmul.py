import triton
import triton.language as tl
import torch
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# torch.manual_seed()

num_rows_a = 1 << 10 # M
num_columns_a = 1 << 10 # N
num_rows_b = 1 << 10 # N
num_columns_b = 1 << 11 # K

a = torch.rand(num_rows_a, num_columns_a, device=DEVICE)
b = torch.rand(num_rows_b, num_columns_b, device=DEVICE)
a_cpu = torch.rand(num_rows_a, num_columns_a, device="cpu")
b_cpu = torch.rand(num_rows_b, num_columns_b, device="cpu")

@triton.jit
def matmulKernel(
  a_ptr, b_ptr, c_ptr,
  M, N, K,
  stride_am, stride_an,
  stride_bn, stride_bk,
  stride_cm, stride_ck,
  BLOCK_SIZE_M: tl.constexpr,
  BLOCK_SIZE_K: tl.constexpr
  ):
  
  # Get program coordinates
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)

  row_start = pid_m * BLOCK_SIZE_M
  column_start = pid_k * BLOCK_SIZE_K

  rows = row_start + tl.arange(0, BLOCK_SIZE_M)
  columns = column_start + tl.arange(0, BLOCK_SIZE_K)

  # Define accumulator
  accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

  # Define masks
  row_mask = rows < M
  column_mask = columns < K

  # Loop over shared dimension
  for n in range(N):
    # Get vertical slice of A, broadcast and mask
    offsets_a = rows.expand_dims(1) * stride_am + n * stride_an # Shape -> (BLOCK_SIZE_M, 1)
    a_slice = tl.load(a_ptr + offsets_a, mask=row_mask.expand_dims(1), other=0.0)
    
    # Get horizontal slice of B, broadcast and mask
    offsets_b = n * stride_bn + columns.expand_dims(0) * stride_bk # Shape -> (1, BLOCK_SIZE_K)
    b_slice = tl.load(b_ptr + offsets_b, mask=column_mask.expand_dims(0), other=0.0)

    # Multiply outer product and add to accumulator
    accumulator += a_slice * b_slice

  # Compute flat memory offsets for each (row, col) pair in the tile
  offsets_c = rows.expand_dims(1) * stride_cm + columns.expand_dims(0) * stride_ck

  # store accumulator results to c
  c_mask = row_mask.expand_dims(1) & column_mask.expand_dims(0)
  tl.store(c_ptr + offsets_c, accumulator, mask=c_mask)


def matmul(a, b):
  # Assert shapes and device
  assert a.shape[1] == b.shape[0], "Matrix dimensions do not match"

  # Get matrices dimensions
  M, N = a.shape
  K = b.shape[1]

  # Define output tensor
  c = torch.empty(M, K, device=DEVICE)

  assert a.device == DEVICE and b.device == DEVICE and c.device == DEVICE

  # Define the strides of the tensors
  stride_am, stride_an = a.stride()
  stride_bn, stride_bk = b.stride()
  stride_cm, stride_ck = c.stride()

  # Get grid dimenions
  grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(K, meta["BLOCK_SIZE_K"]))

  # Launch grid
  matmulKernel[grid](
      a, b, c,
      M, N, K,
      stride_am, stride_an,
      stride_bn, stride_bk,
      stride_cm, stride_ck,
      BLOCK_SIZE_M=32,
      BLOCK_SIZE_K=32
  )

  return c

# Warmup kernel
_ = matmul(a, b)

# Measure triton kernel execution time
torch.cuda.synchronize()
start = time.perf_counter()
output_triton = matmul(a, b)
torch.cuda.synchronize()
end = time.perf_counter()
triton_time = (end - start) * 1000

# Measure Pytorch GPU execution time
torch.cuda.synchronize()
start = time.perf_counter()
torch_gpu_output = torch.matmul(a, b)
torch.cuda.synchronize()
end = time.perf_counter()
torch_gpu_time = (end - start) * 1000

# Measure Pytorch CPU execution time
start = time.perf_counter()
torch_cpu_output = torch.matmul(a_cpu, b_cpu)
end = time.perf_counter()
torch_cpu_time = (end - start) * 1000

# Check correctness
max_diff = torch.max(torch.abs(output_triton - torch_gpu_output))
assert torch.allclose(output_triton, torch_gpu_output, atol=1e-4, rtol=1e-5), "Mismatch with PyTorch!"

# Calculate total FLOPs for matrix multiplication: 2 * M * N * K
total_flops = 2.0 * num_rows_a * num_columns_a * num_columns_b

# Convert GPU time to seconds
triton_time_sec = triton_time / 1000.0

# Throughput in TFLOPs/s
tflops = total_flops / (triton_time_sec * 1e12)

print(f"Triton time:       {triton_time:.3f} ms")
print(f"Triton kernel throughput: {tflops} TFLOPs/s")
print(f"PyTorch GPU time:  {torch_gpu_time:.3f} ms")
print(f"PyTorch CPU time:  {torch_cpu_time:.3f} ms")
print(f"Max absolute diff: {max_diff.item():.6e}")


# cuda:0
# Triton time:       2.549 ms
# Triton kernel throughput: 1.6847395933583826 TFLOPs/s
# PyTorch GPU time:  204.987 ms
# PyTorch CPU time:  47.192 ms
# Max absolute diff: 6.103516e-04