# Matrix transpose: column major to row major (2D)

## Problem statement

We are given a 2D matrix of floats stored in **column major** memory layout.

- Logical shape: `M × K`
- Storage: column major (columns are contiguous in memory)

Our goal is to compute the **transpose** of this matrix and store the result in **row major** layout.

- Output shape: `K × M`
- Storage: row major (rows are contiguous in memory)

Mathematically, the operation is:

$$
B(j, i) = A(i, j)
$$

where:
- `A` is the input matrix (`M × K`, column major)
- `B` is the output matrix (`K × M`, row major)


## Memory layout reminder

For the input matrix `A` (column major):

$$
A(i, j) \rightarrow A[j \cdot M + i]
$$

For the output matrix `B` (row major):

$$
B(j, i) \rightarrow B[j \cdot M + i]
$$


## Kernel mapping

- Each thread handles one matrix element `(i, j)`
- Threads load one value from column major memory
- Threads store one value into row major memory
- Bounds checks ensure `i < M` and `j < K`

This kernel is intentionally **naive** and serves as a baseline:
- No shared memory
- No tiling
- No vectorisation

The purpose is correctness and clarity, not peak performance.


## What this kernel demonstrates

- Understanding of row major vs column major layout
- Correct index mapping for transpose
- Awareness of memory stride and access patterns
- Clean 2D grid and thread indexing

This is a common building block and a frequent CUDA interview question.