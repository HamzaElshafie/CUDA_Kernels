import triton
import triton.language as tl
import torch
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# torch.manual_seed()

num_rows_a = 1 << 8 # M
num_columns_a = 1 << 8 # N
num_rows_b = 1 << 8 # N
num_columns_b = 1 << 10 # K

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

  # Compute flat memory offsets for each (row, col) pair in the tile
  offsets_c = rows.expand_dims(1) * stride_cm + rows.expand_dims(0) * stride_ck

  # Define accumulator
  accumulator = tl.zeros(BLOCK_SIZE_M, BLOCK_SIZE_K)

  # Define mask
  row_mask = rows < M
  column_mask = columns < K
  mask = row_mask.expand_dims(1) & column_mask.expand_dims(0)

  # Calculate offsets
  offsets_a = rows.expand_dims(1) * stride_am + tl.arange(0, N).expand_dims(0) * stride_a  


def matmul(a, b):
  # Assert shapes and device
  assert a.shape[1] == b.shape[0], (f"Matrix multiplication requires the number of columns in the first matrix "
    f"({a.shape[1]}) to match the number of rows in the second matrix ({b.shape[0]}).")
  
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

# Measure triton kernel execution time

# Measure Pytorch GPU execution time

# Measure Pytorch CPU execution time

# Check correctness