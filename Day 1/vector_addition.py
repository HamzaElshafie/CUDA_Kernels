import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def add_kernel(x_ptr, y_ptr, output_ptr, num_elements, BLOCK_SIZE=tl.constexpr):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE) 

def add(x: torch.tensor, y: torch.tensor):
  output = torch.empty_like(x)
  assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

  num_elements = output.numel()
  grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]), )
  add_kernel[grid](x, y, output, num_elements, BLOCK_SIZE=1024)

  return output
