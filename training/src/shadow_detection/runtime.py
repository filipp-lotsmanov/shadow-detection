"""Runtime device detection.

Single source of truth for "do we have CUDA available?" - used to toggle AMP,
DataLoader pinning, worker counts, and autocast device on a per-machine basis.

The expected runtime flow:
  1. gpu.select_free_gpu() runs at script start (before `import torch`).
  2. After torch imports, code calls get_device() / use_amp() / dataloader_kwargs()
     to pick up the right settings for THIS machine.
"""

from __future__ import annotations

import os
import platform
from typing import Any

import torch


def get_device() -> torch.device:
    """Return cuda if available, else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def use_amp(requested: bool = True) -> bool:
    """AMP only works on CUDA. Return False on CPU regardless of config request."""
    return bool(requested) and torch.cuda.is_available()


def dataloader_kwargs(num_workers: int = 8) -> dict[str, Any]:
    """Return DataLoader kwargs appropriate for the current machine.

    On Windows, multiprocessing for DataLoader workers is fragile (especially with
    persistent_workers + Hydra + Jupyter). On CPU we also gain little from many
    workers since the bottleneck shifts elsewhere. So:
      - Windows: force num_workers=0, no persistent workers, no pinning
      - Linux without CUDA: keep workers but disable CUDA-specific perf flags
      - Linux with CUDA: full speed (workers + pin + persistent)
    """
    is_windows = platform.system() == "Windows"
    has_cuda = torch.cuda.is_available()

    if is_windows:
        return {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }

    if not has_cuda:
        return {
            "num_workers": min(num_workers, 2),
            "pin_memory": False,
            "persistent_workers": min(num_workers, 2) > 0,
        }

    return {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }


def print_runtime_summary() -> None:
    """One-line printout describing what we'll be running on."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Runtime: CUDA on {name}")
    else:
        sysname = platform.system()
        print(f"Runtime: CPU only ({sysname}). Training will be much slower.")
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
            print("  (CUDA_VISIBLE_DEVICES is set but no CUDA runtime is reachable)")
