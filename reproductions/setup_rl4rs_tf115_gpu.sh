#!/usr/bin/env bash
# Upgrade RL4RS online stack: NVIDIA TF 1.15.5+nv23.02 (sm_89) + PyTorch cu118.
# Large wheels go to /root/autodl-tmp to avoid filling the 30G root overlay.
set -euo pipefail

export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
WHEEL_DIR=/root/autodl-tmp/wheels
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$WHEEL_DIR"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rl4rs-tf115

pip_install() {
  echo ""
  echo "========== pip install: $* =========="
  pip install --no-cache-dir "$@"
}

wget_wheel() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "[skip] already have $(basename "$out")"
    return 0
  fi
  echo ""
  echo "========== downloading $(basename "$out") =========="
  wget -c --progress=bar:force:noscroll -O "$out" "$url"
}

echo "Disk: $(df -h / /root/autodl-tmp | tail -2)"

# --- NVIDIA TensorFlow wheel (sm_89, CUDA 12) ---
TF_WHL="$WHEEL_DIR/nvidia_tensorflow-1.15.5+nv23.02-7195399-cp38-cp38-linux_x86_64.whl"
wget_wheel \
  "https://developer.nvidia.cn/w/compute/redist/nvidia-tensorflow/nvidia_tensorflow-1.15.5%2Bnv23.02-7195399-cp38-cp38-linux_x86_64.whl" \
  "$TF_WHL"

# --- TF 1.15 deps from PyPI (avoid broken NVIDIA tensorboard redirect) ---
pip_install \
  'numpy>=1.22.0,<1.24' 'h5py==2.10.0' \
  'tensorboard==1.15.0' 'tensorflow-estimator==1.15.1' 'protobuf>=3.6.1,<4' \
  'gast==0.3.3' 'astor==0.8.1' 'astunparse==1.6.3' 'absl-py' \
  'keras-applications' 'keras-preprocessing' 'opt-einsum' 'six' 'google-pasta' \
  grpcio termcolor wrapt wheel

# --- NVIDIA CUDA companion libs (pin cudnn 8.7 for nv23.02) ---
pip_install \
  nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 'nvidia-cudnn-cu11==8.7.0.84' \
  nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-nccl-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvcc-cu12

pip_install --no-deps "$TF_WHL"

# --- PyTorch (sm_89): local wheel + Tsinghua mirror for deps (avoid 663MB cudnn re-download) ---
TORCH_WHL="$WHEEL_DIR/torch-2.4.1+cu118-cp38-cp38-linux_x86_64.whl"
wget_wheel "https://download.pytorch.org/whl/cu118/torch-2.4.1%2Bcu118-cp38-cp38-linux_x86_64.whl" "$TORCH_WHL"

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
echo "========== torch --no-deps (reuse existing nvidia-cudnn-cu11) =========="
pip_install --no-deps "$TORCH_WHL"
pip_install filelock typing-extensions sympy networkx jinja2 fsspec mpmath
# skip nvidia-cudnn-cu11 (663MB); env already has 9.10 from TF stack
pip_install \
  nvidia-cuda-runtime-cu11==11.8.89 nvidia-cuda-nvrtc-cu11==11.8.89 \
  nvidia-cuda-cupti-cu11==11.8.87 nvidia-cufft-cu11==10.9.0.58 \
  nvidia-curand-cu11==10.3.0.86 nvidia-cusolver-cu11==11.4.1.48 \
  nvidia-cusparse-cu11==11.7.5.86 nvidia-nccl-cu11==2.20.5 nvidia-nvtx-cu11==11.8.86
# triton optional for small MLP Q-net; install if needed:
# pip_install triton==3.0.0
unset PIP_INDEX_URL

# --- RL4RS project deps (py3.8 compatible) ---
pip_install \
  'pandas==1.1.5' 'scikit-learn==0.24.2' 'gym==0.19.0' 'tqdm' 'pyyaml' \
  'deepctr==0.9.0' 'd3rlpy==1.1.1' 'opencv-python-headless==4.3.0.36'

# --- RLlib DQN (official online stack) ---
pip_install 'ray==1.5.1' 'dm-tree' 'tabulate' 'lz4' 'aiohttp==3.7.4'

echo ""
echo "========== GPU smoke test =========="
python - <<'PY'
import tensorflow as tf
print("TF:", tf.__version__)
print("TF GPU:", tf.test.is_gpu_available())
import torch
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(4, 4, device="cuda")
    print("Torch matmul OK:", (x @ x).shape)
PY

echo ""
echo "Done. Activate with: conda activate rl4rs-tf115"
