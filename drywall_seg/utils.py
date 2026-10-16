import json
import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # bilinear upsampling has no deterministic CUDA backward; warn instead of crashing
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """DataLoader worker_init_fn: derive numpy/random seeds from torch's per-worker seed."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model):
    tensors = [*model.parameters(), *model.buffers()]
    return sum(t.numel() * t.element_size() for t in tensors) / 2**20


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
