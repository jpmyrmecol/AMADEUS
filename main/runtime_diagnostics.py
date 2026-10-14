# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Environment facts shared by startup output and tracking run logs.

GPU availability means PyTorch can access an accelerator. It does not mean
that every tracking stage is configured to execute on that accelerator.
"""

from __future__ import annotations

import platform
import sys

import torch

from main.compute_backend import Backend, detect_backend
from main.version_info import installed_version, version_label


def runtime_details() -> dict[str, str]:
    """Collect the versions and accelerator state of the running interpreter."""
    backend = detect_backend()
    if backend in (Backend.NVIDIA_CUDA, Backend.AMD_ROCM):
        devices = ", ".join(
            f"{index}: {torch.cuda.get_device_name(index)}"
            for index in range(torch.cuda.device_count())
        )
        automatic_device = "cuda:0"
    elif backend == Backend.APPLE_MPS:
        devices = "Apple Silicon GPU"
        automatic_device = "mps"
    else:
        devices = "none"
        automatic_device = "cpu"

    return {
        "amadeus_version": installed_version(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_runtime": str(torch.version.cuda or "none"),
        "rocm_runtime": str(torch.version.hip or "none"),
        "gpu_backend": backend.value,
        "gpu_available": "true" if backend != Backend.CPU else "false",
        "gpu_devices": devices,
        "automatic_torch_device": automatic_device,
    }


def runtime_summary_lines(details: dict[str, str]) -> list[str]:
    """Format the same diagnostic facts for interactive launch and batch logs."""
    available = "available" if details["gpu_available"] == "true" else "unavailable"
    return [
        f"[AMADEUS] Version: {version_label(details['amadeus_version'])}",
        f"[AMADEUS] Python: {details['python_version']} ({details['python_executable']})",
        f"[AMADEUS] PyTorch: {details['torch_version']} "
        f"(CUDA runtime: {details['cuda_runtime']}, ROCm runtime: {details['rocm_runtime']})",
        f"[AMADEUS] GPU: {details['gpu_devices']} ({details['gpu_backend']}, {available})",
        f"[AMADEUS] Automatic PyTorch device: {details['automatic_torch_device']}",
    ]
