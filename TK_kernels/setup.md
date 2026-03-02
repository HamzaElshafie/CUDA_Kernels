# ThunderKittens Setup (H100)

## 1) Clone ThunderKittens
```bash
git clone https://github.com/HazyResearch/ThunderKittens.git ~/ThunderKittens
```

## 2) Set env vars
```bash
export THUNDERKITTENS_ROOT=~/ThunderKittens
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
```

## 3) Build from `TK_kernels`
```bash
cd /path/to/your/repo/TK_kernels
cmake -S . -B build
cmake --build build -j
```

## 4) Run a kernel binary
Each `.cu` file in `TK_kernels` builds to one executable with the same name.

Example: if you have `hello_tk.cu`, run:
```bash
./build/hello_tk
```

## 5) Rebuild after edits
```bash
cmake --build build -j
```
