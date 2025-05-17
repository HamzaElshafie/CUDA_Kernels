# CUDA_Kernels

This repository is a personal log of my journey through the 100 Days of CUDA challenge. Alongside this, I am also studying Programming Massively Parallel Processors (PMPP) to deepen my understanding of GPU architecture and parallel programming concepts.

Mentor: https://github.com/hkproj/

| Day        | File / Topic                  | Summary |
|------------|-------------------------------|---------|
| 1          | `vector_addition.cu`               | CUDA vector addition on CPU and GPU; ~26.9× speedup. Started Triton version. Read Chapter 1. |
| 2          | `vector_addition.py`        | Triton vector addition; ~4–8× speedup over PyTorch CPU. Read Chapter 2 (2.1–2.4). |
| 3          | `matrix_addition.cu`               | CUDA matrix addition (2D); ~100× speedup over CPU. |
| 5–6        | `matrix_addition.py`        | Triton matrix addition. Compared with PyTorch. Finished Chapter 2. |
| 7          | `utils.h`                     | Wrote header for timing, error checking, and array initialisation using CUDA events. |
| 8          | `relu.cu`                     | CUDA ReLU kernel for 2D matrices; ~168.07× speedup over CPU. Read Chapter 3 (3.1–3.2). |
| 9          | `naive_matmul.cu`             | Naive CUDA matrix multiplication; ~2409.27× speedup. Supports non-square matrices. |
| 10         | `reading`                     | Read PMPP sections 4.1–4.4. |
| 11         | `naive_matmul.py`      | Started naive matrix multiplication in Triton. Initial struggle with tiling concepts. |
| 12         | `naive_matmul.py`      | Completed Triton naive matmul. Understood outer product accumulation. Learned to calculate throughput. |
| 13–14      | `N/A`                  | Finished Chapter 4 and sections 5.1–5.2. Studied CUDA memory types and tradeoffs. |
| 15         | `tiled_matmul.cu`             | Tiled CUDA matrix multiplication using shared memory; ~1.26× speedup over naive CUDA. |
| 16         | `online_softmax.cu`           | Implemented CUDA online softmax kernel. Fused max and normalisation in a single pass. |
| 17         | `smem_online_softmax.cu`      | Started shared memory version. Used block-per-row layout and reduction. Watched GPU Mode Lecture 8. |
| 18         | `smem_online_softmax.cu`      | Finalised and tested shared memory version. Observed noticeable execution time improvement. |
