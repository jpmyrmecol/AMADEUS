# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Runtime identity and capabilities, independent of installer and OS policy.

``cuda`` is a PyTorch API/device namespace shared by CUDA and HIP. Never use
it as a vendor identifier. Optional measurements return None, not zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import sys


class Backend(str, Enum):
    NVIDIA_CUDA = "NVIDIA CUDA"
    AMD_ROCM = "AMD ROCm"
    APPLE_MPS = "Apple MPS"
    CPU = "CPU"


@dataclass(frozen=True)
class Capabilities:
    torch_device: str
    memory_model: str
    telemetry: str | None
    configured_workers: bool
    # Permission to request AMP; Ultralytics retains its own capability checks.
    amp: bool


CAPABILITIES = {
    Backend.NVIDIA_CUDA: Capabilities("cuda", "dedicated", "nvidia", True, True),
    Backend.AMD_ROCM: Capabilities("cuda", "dedicated", None, True, True),
    Backend.APPLE_MPS: Capabilities("mps", "unified", None, False, True),
    Backend.CPU: Capabilities("cpu", "system", None, False, False),
}


def _torch():
    import torch
    return torch


def mps_available() -> bool:
    try:
        torch = _torch()
        if not torch.backends.mps.is_available():
            return False
        value = torch.ones(1, device="mps") + 1
        torch.mps.synchronize()
        return value.item() == 2
    except Exception:
        return False


def detect_backend() -> Backend:
    try:
        torch = _torch()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            if getattr(torch.version, "hip", None):
                return Backend.AMD_ROCM
            if getattr(torch.version, "cuda", None):
                return Backend.NVIDIA_CUDA
        if mps_available():
            return Backend.APPLE_MPS
    except Exception:
        pass
    return Backend.CPU


@dataclass(frozen=True)
class Runtime:
    backend: Backend
    index: int = 0

    @property
    def capabilities(self) -> Capabilities:
        return CAPABILITIES[self.backend]

    @property
    def wddm(self) -> bool:
        return self.backend == Backend.NVIDIA_CUDA and sys.platform == "win32"

    @property
    def torch_device(self) -> str:
        kind = self.capabilities.torch_device
        return f"cuda:{self.index}" if kind == "cuda" else kind

    def empty_cache(self) -> None:
        kind = self.capabilities.torch_device
        if kind == "cuda":
            _torch().cuda.empty_cache()
        elif kind == "mps":
            try:
                _torch().mps.empty_cache()
            except Exception:
                pass

    def synchronize(self) -> None:
        kind = self.capabilities.torch_device
        if kind == "cuda":
            _torch().cuda.synchronize(self.index)
        elif kind == "mps":
            try:
                _torch().mps.synchronize()
            except Exception:
                pass

    def memory(self) -> tuple[float | None, float | None]:
        """Allocator reserved / device total, or MPS driver / working-set budget.

        MPS's second value is a recommended budget, never dedicated VRAM.
        """
        try:
            torch = _torch()
            if self.capabilities.memory_model == "dedicated":
                return (float(torch.cuda.memory_reserved(self.index)),
                        float(torch.cuda.get_device_properties(self.index).total_memory))
            if self.capabilities.memory_model == "unified":
                return (float(torch.mps.driver_allocated_memory()),
                        float(torch.mps.recommended_max_memory()))
        except Exception:
            pass
        return None, None

    def allocated_memory(self) -> float | None:
        try:
            if self.capabilities.torch_device == "cuda":
                return float(_torch().cuda.memory_allocated(self.index))
            if self.capabilities.torch_device == "mps":
                return float(_torch().mps.current_allocated_memory())
        except Exception:
            pass
        return None


def runtime_for(device=None) -> Runtime:
    """Resolve identity for an already selected device; unavailable => CPU."""
    value = str(device).strip().lower()
    if value == "cpu":
        return Runtime(Backend.CPU)
    if value in {"mps", "mps:0"}:
        # Explicit device was validated by resolve_device; no per-batch probe.
        return Runtime(Backend.APPLE_MPS)
    backend = detect_backend()
    if value in {"none", "auto", ""}:
        return Runtime(backend)
    indices = cuda_indices(value)
    if indices and CAPABILITIES[backend].torch_device == "cuda":
        if max(indices) < _torch().cuda.device_count():
            return Runtime(backend, indices[0])
    return Runtime(Backend.CPU)


def cuda_indices(device) -> list[int]:
    value = str(device).strip().lower()
    if value == "cuda":
        return [0]
    if value.startswith("cuda:"):
        value = value[5:]
    parts = value.split(",")
    return [int(p.strip()) for p in parts] if all(p.strip().isdigit() for p in parts) else []


def resolve_device(device="auto", purpose="YOLO") -> str:
    requested = str(device).strip()
    value = requested.lower()
    runtime = runtime_for(device)
    if value in {"auto", "none", ""}:
        result = "0" if runtime.capabilities.torch_device == "cuda" else runtime.torch_device
    elif value == "cpu":
        result = "cpu"
    elif value in {"mps", "mps:0"}:
        result = "mps" if mps_available() else "cpu"
        runtime = Runtime(Backend.APPLE_MPS if result == "mps" else Backend.CPU)
    elif cuda_indices(value) or value.startswith("cuda"):
        result = requested if runtime.capabilities.torch_device == "cuda" else "cpu"
    else:
        return requested  # Leave invalid/custom inputs to Ultralytics validation.
    if runtime.backend == Backend.AMD_ROCM and result != "cpu":
        try:
            torch = _torch()
            for index in cuda_indices(result):
                value = torch.ones(1, device=f"cuda:{index}") + 1
                torch.cuda.synchronize(index)
                if value.item() != 2:
                    raise RuntimeError("Unexpected HIP computation result")
        except Exception as exc:
            print(f"[WARN] AMD ROCm cannot execute on device={result}: {exc}; using CPU.")
            result, runtime = "cpu", Runtime(Backend.CPU)
    print(f"[INFO] {purpose} device: {result} ({runtime.backend.value})")
    return result
