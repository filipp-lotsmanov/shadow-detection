"""GPU selection helper. Must be called BEFORE any `import torch` statement."""

from __future__ import annotations

import os
import subprocess


def select_free_gpu(min_free_mb: int = 10000) -> int:
    """Pick the GPU with the most free VRAM and set CUDA_VISIBLE_DEVICES.

    Must be called before `import torch` - otherwise PyTorch caches the device
    list and the env var has no effect.

    Returns the index of the selected GPU.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvidia-smi not available - leaving CUDA_VISIBLE_DEVICES untouched.")
        return -1

    best_gpu, best_free = 0, 0
    for line in result.stdout.strip().split("\n"):
        idx_s, free_s, _util_s = (x.strip() for x in line.split(","))
        idx, free_mb = int(idx_s), int(free_s)
        if free_mb > best_free:
            best_free = free_mb
            best_gpu = idx

    if best_free < min_free_mb:
        print(f"Warning: best GPU {best_gpu} has only {best_free}MB free")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
    print(f"Selected GPU {best_gpu} ({best_free}MB free)")
    return best_gpu
