import torch
import triton
import triton.language as tl
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# torch.manual_seed(0)

size = 1 << 20
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
x_cpu = torch.rand(size, device=torch.device("cpu"))
y_cpu = torch.rand(size, device=torch.device("cpu"))

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
  # Get program index
  pid = tl.program_id(axis=0) # blockIdx.x
  block_start = pid * BLOCK_SIZE # blockIdx.x * blockDim.x
  # Generate the range of global indices this program is responsible for
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < num_elements # Guard against out of bound invalid operations

  # Load vectors from DRAM, masking out any extra elements in case the input is not a
  # multiple of the block size.
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y 

  # Write result back to DRAM
  tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.tensor, y: torch.tensor):
  output = torch.empty_like(x)
  assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

  num_elements = output.numel()
  grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]), )
  add_kernel[grid](x, y, output, num_elements, BLOCK_SIZE=1024)

  return output


# Warmup and cache kernel
_ = add(x, y)

# Measure Triton execution time
torch.cuda.synchronize()
start = time.perf_counter()
output_triton = add(x, y)
torch.cuda.synchronize()
end = time.perf_counter()
triton_time = (end - start) * 1000

# Measure PyTorch GPU execution time
torch.cuda.synchronize()
start = time.perf_counter()
output_torch = x + y
torch.cuda.synchronize()
end = time.perf_counter()
pytorch_time = (end - start) * 1000

# Measure PyTorch CPU execution time
start = time.perf_counter()
output_torch_cpu = x_cpu + y_cpu
end = time.perf_counter()
pytorch_time_cpu = (end - start) * 1000

print(f"PyTorch CPU execution time: {pytorch_time_cpu:.5f}ms")
print(f"PyTorch GPU execution time: {pytorch_time:.5f}ms")
print(f"Triton  execution time: {triton_time:.5f}ms")
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

# PyTorch CPU execution time: 15.63703ms
# PyTorch GPU execution time: 1.85400ms
# Triton  execution time: 1.99292ms
# The maximum difference between torch and triton is 0.0