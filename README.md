# CUDA_Kernels

This repository is a personal log of my journey through the 100 Days of CUDA challenge. Alongside this, I am also studying Programming Massively Parallel Processors (PMPP) to deepen my understanding of GPU architecture and parallel programming concepts.

Mentor: https://github.com/hkproj/

## Day 1 – CUDA Vector Addition kernel

- Implemented vector addition on both **CPU** and **GPU** using CUDA  
- Achieved ~**26.9× speedup** with CUDA kernel over CPU
- Started translating the kernel into Triton (in progress, will continue tomorrow)
- Read Chapter 1

## Day 2 – Triton Vector Addition kernel

- Implemented vector addition again, but using Triton
- Achieved ~4–8× speedup with Triton kernel over PyTorch CPU operation
- Read Chapter 2 from 2.1-2.4

## Day 3 – CUDA Matrix Addition kernel

- Implemented matrix addition on both **CPU** and **GPU** using CUDA (Supports 2D matrices only)
- Achieved ~**100× speedup** with CUDA kernel over CPU

## Day 5 & 6 – Triton Matrix Addition
- Implemented same matrix addition in Triton and ran tests to compare with Pytorch native ops.
- Finished Chapter 2

## Day 7 - Utils.h
- While not a CUDA kernel, today I implemented a reusable utility header file.
- It includes functions for timing CPU execution and CUDA kernel execution using CUDA events for accurate measurement.
- Also added a CUDA_CHECK macro for robust error checking and a function to initialise arrays with random values.

## Day 8 - CUDA ReLU kernel
- Implemented a CUDA kernel for the ReLU activation function that operates on 2D matrices.
- Compared the results against a CPU implementation for correctness and performance.
- Achieved an approximate ~168.07× speedup with the CUDA kernel over the CPU version.
- Read sections 3.1 and 3.2 from chapter 3.

## Day 9 - CUDA Naive matrix multiplication kernel
-	Implemented a naive CUDA kernel for matrix multiplication and compared its performance against CPU version.
- The implementation supports both square and non-square matrices, as long as the dimensions are valid for matrix multiplication (i.e., A: M×N, B: N×K).
- Achieved approximately 2409.27× speedup over the CPU version. Will experiment with shared memory and tiling next to further optimise performance.
