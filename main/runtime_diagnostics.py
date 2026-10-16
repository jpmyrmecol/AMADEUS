# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Environment facts shared by startup output and tracking run logs.

GPU availability means PyTorch can access an accelerator. It does not mean
that every tracking stage is configured to execute on that accelerator.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys

import torch

from main.compute_backend import Backend, detect_backend
from main.version_info import installed_version, version_label


def accelerator_memory_details(backend: Backend) -> dict[str, str]:
    """Report accelerator capacity before training; never confuse MPS RAM with VRAM."""
    gib = 1024 ** 3
    if backend in (Backend.NVIDIA_CUDA, Backend.AMD_ROCM):
        capacities = []
        for index in range(torch.cuda.device_count()):
            try:
                total = torch.cuda.get_device_properties(index).total_memory
                value = f"{total / gib:.2f}"
            except Exception:
                value = "unavailable"
            capacities.append((index, value))
        return {
            # The automatic PyTorch device is cuda:0; other GPUs are listed separately.
            "vram_total_device": "cuda:0" if capacities else "unavailable",
            "vram_total_gib": capacities[0][1] if capacities else "unavailable",
            "vram_by_device_gib": ", ".join(f"cuda:{idx}:{size}" for idx, size in capacities)
            if capacities else "unavailable",
        }
    if backend == Backend.APPLE_MPS:
        try:
            value = f"{torch.mps.recommended_max_memory() / gib:.2f}"
        except Exception:
            value = "unavailable"
        # Apple GPUs use unified memory; this is a recommended budget, not VRAM.
        return {"mps_recommended_gib": value}
    return {}


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
        **accelerator_memory_details(backend),
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


def system_details() -> dict[str, str]:
    """Collect small, non-identifying hardware facts for reproducible run logs."""
    cpu_model = platform.processor() or "unknown"
    if sys.platform == "darwin":
        try:
            value = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL, text=True, timeout=2,
            ).strip()
            if value:
                cpu_model = value
        except (OSError, subprocess.SubprocessError):
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as stream:
                for line in stream:
                    if line.startswith("model name"):
                        cpu_model = line.partition(":")[2].strip()
                        break
        except OSError:
            pass
    elif sys.platform == "win32":
        cpu_model = os.environ.get("PROCESSOR_IDENTIFIER") or cpu_model

    import psutil

    gib = 1024 ** 3
    memory = psutil.virtual_memory()
    battery = psutil.sensors_battery()
    return {
        "cpu_model": cpu_model,
        "cpu_cores": f"{psutil.cpu_count(logical=False) or 'n/a'} physical/"
                     f"{psutil.cpu_count(logical=True) or 'n/a'} logical",
        "ram_total_gib": f"{memory.total / gib:.1f}",
        "ram_used_gib": f"{(memory.total - memory.available) / gib:.2f}",
        "swap_used_gib": f"{psutil.swap_memory().used / gib:.2f}",
        "power_source": (
            "unknown" if battery is None
            else "AC" if battery.power_plugged else "battery"
        ),
    }
