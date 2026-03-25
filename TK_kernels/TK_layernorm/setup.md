# ThunderKittens Setup (H100)

## 1) Clone repos
```bash
git clone <https://github.com/HamzaElshafie/CUDA_Kernels.git>
git clone https://github.com/HazyResearch/ThunderKittens.git ~/ThunderKittens
```

## 2) Install prerequisites

### CMake (required: >= 3.24)
```bash
apt-get update
apt-get install -y gpg wget
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' > /etc/apt/sources.list.d/kitware.list
apt-get update
apt-get install -y cmake
```

### CUDA Toolkit 12.8 (nvcc)
```bash
apt-get update
apt-get install -y wget gpg
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-toolkit-12-8
```

### Python + Triton (for `baselines.py`)
```bash
apt-get update
apt-get install -y python3 python3-pip
python3 -m pip install --upgrade pip
python3 -m pip install torch triton
```

## 3) Set environment
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export THUNDERKITTENS_ROOT=~/ThunderKittens
```

## 4) Verify environment
```bash
nvidia-smi
nvcc --version
cmake --version
python3 -c "import torch, triton; print(torch.__version__, triton.__version__)"
```

## 5) Configure and build
```bash
cd ~/CUDA_Kernels/TK_kernels/TK_layernorm
rm -rf build
cmake -S . -B build -DCUDAToolkit_ROOT=$CUDA_HOME -DTHUNDERKITTENS_ROOT=$THUNDERKITTENS_ROOT
cmake --build build -j
```

## 6) Run kernels
```bash
# CUDA/TK kernel
./build/tk_layernorm

# Triton forward kernel (import test)
python3 -c "from baselines import triton_fused_residual_layer_norm_fwd; print('Triton LayerNorm ready')"
```
