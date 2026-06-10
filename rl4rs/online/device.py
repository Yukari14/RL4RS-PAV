import os

import torch
import torch.nn as nn


def _cuda_kernels_compatible():
    if not torch.cuda.is_available():
        return False
    try:
        layer = nn.Linear(8, 8).cuda()
        x = torch.randn(2, 8, device="cuda")
        layer(x)
        return True
    except RuntimeError:
        return False


def resolve_torch_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if _cuda_kernels_compatible():
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        print(
            "WARNING: GPU visible but PyTorch in rl4rs env lacks kernels for this GPU "
            "(e.g. RTX 4090 needs torch>=1.13). Using CPU for Q-network/PAV; "
            "use --batch-size 32+ to speed up simulator.",
            flush=True,
        )
    return torch.device("cpu")


def simulator_config_gpu(use_cpu=False):
    """RecSimBase: gpu=True -> TF sim on CPU; gpu=False -> TF tries GPU (TF1.15 often still CPU)."""
    if use_cpu or not torch.cuda.is_available():
        return True
    return False


def configure_runtime(force_cpu=False):
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
