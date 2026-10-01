# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Versioned wheel/support policy; runtime backend identity has no versions."""
from dataclasses import dataclass
import platform
import sys

from main.compute_backend import Backend

TORCH_VERSION = "2.14.0"
TORCHVISION_VERSION = "0.29.0"


@dataclass(frozen=True)
class WheelProfile:
    backend: Backend
    suffix: str
    index: str | None
    runtime_version: str | None = None
    torch_version: str = TORCH_VERSION
    torchvision_version: str = TORCHVISION_VERSION
    package_extra: str | None = None
    install_dependencies: bool = False
    python_minor: str | None = None
    sync_extra: str | None = None


PROFILES = {
    "cpu": WheelProfile(Backend.CPU, "cpu", "https://download.pytorch.org/whl/cpu"),
    "macos": WheelProfile(Backend.APPLE_MPS, "", None),
    # CUDA 13.2 is the primary NVIDIA profile for Turing (CC 7.5) and newer.
    # It is installed outside uv.lock so legacy CUDA 12.6 remains available.
    "cu132": WheelProfile(
        Backend.NVIDIA_CUDA,
        "cu132",
        "https://download.pytorch.org/whl/cu132",
        "13.2",
        sync_extra="cpu",
    ),
    # PyTorch 2.14 + CUDA 12.6 is the final prebuilt line retaining
    # Maxwell/Pascal/Volta support.
    "cu126": WheelProfile(Backend.NVIDIA_CUDA, "cu126", "https://download.pytorch.org/whl/cu126", "12.6"),
    "rocm72": WheelProfile(Backend.AMD_ROCM, "rocm7.2", "https://download.pytorch.org/whl/rocm7.2", "7.2"),
    # AMD ROCm 10 wheels are distributed from AMD's wheel index. device-all
    # installs the GPU runtime components required by supported hardware.
    "rocm100": WheelProfile(
        Backend.AMD_ROCM,
        "rocm10.0.0",
        "https://stable.repo.amd.com/rocm/whl-next/",
        None,
        torch_version="2.13.0",
        torchvision_version="0.28.0",
        package_extra="device-all",
        install_dependencies=True,
        python_minor="3.12",
        sync_extra="cpu",
    ),
    "rocmwin100": WheelProfile(
        Backend.AMD_ROCM,
        "rocm10.0.0",
        "https://stable.repo.amd.com/rocm/whl-next/",
        None,
        torch_version="2.13.0",
        torchvision_version="0.28.0",
        package_extra="device-all",
        install_dependencies=True,
        python_minor="3.12",
        sync_extra="cpu",
    ),
}


def check_support(profile: str) -> None:
    if profile not in PROFILES:
        raise RuntimeError(f"Unknown PyTorch profile: {profile}")
    if profile in {"rocm72", "rocm100"} and not (
        sys.platform.startswith("linux") and platform.machine().lower() in {"x86_64", "amd64"}
    ):
        raise RuntimeError(f"The {profile} wheel profile requires Linux x86_64.")
    if profile == "rocmwin100" and not (
        sys.platform == "win32" and platform.machine().lower() in {"x86_64", "amd64"}
    ):
        raise RuntimeError("The rocmwin100 wheel profile requires Windows x86_64.")
    if profile == "macos" and sys.platform != "darwin":
        raise RuntimeError("The macos wheel profile requires macOS.")
    if profile in {"cu126", "cu132"} and sys.platform == "darwin":
        raise RuntimeError(f"The {profile} wheel profile is unavailable on macOS.")


def verify_build(profile, torch, torchvision) -> None:
    """Reject wrong vendors, runtime versions and mismatched torchvision wheels."""
    check_support(profile)
    expected = PROFILES[profile]
    if expected.python_minor:
        running_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
        if running_minor != expected.python_minor:
            raise RuntimeError(
                f"{profile} requires Python {expected.python_minor}, found Python {running_minor}"
            )
    for module, version in (
        (torch, expected.torch_version),
        (torchvision, expected.torchvision_version),
    ):
        if module.__version__.split("+")[0] != version:
            raise RuntimeError(f"Expected {version}, found {module.__version__}")
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    if expected.backend == Backend.NVIDIA_CUDA:
        valid = not hip and cuda == expected.runtime_version
    elif expected.backend == Backend.AMD_ROCM:
        valid = bool(hip) and not cuda
        if valid and expected.runtime_version:
            valid = str(hip).split(".")[:2] == expected.runtime_version.split(".")
    else:
        valid = not hip and not cuda
    if not valid:
        raise RuntimeError(f"Wrong PyTorch runtime for {profile}: CUDA={cuda}, HIP={hip}")
    suffix = expected.suffix if sys.platform != "darwin" else ""
    for module in (torch, torchvision):
        # Official Linux aarch64 torchvision CPU/CUDA wheels have no suffix.
        arm_vision = (module is torchvision and sys.platform.startswith("linux")
                      and platform.machine() in {"aarch64", "arm64"})
        if suffix and not arm_vision and module.__version__.partition("+")[2] != suffix:
            raise RuntimeError(f"Expected +{suffix}, found {module.__version__}")


def verify_execution(profile, torch, torchvision) -> None:
    verify_build(profile, torch, torchvision)
    expected = PROFILES[profile]
    if expected.backend not in {Backend.NVIDIA_CUDA, Backend.AMD_ROCM}:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(f"{expected.backend.value} cannot access a GPU; use the cpu profile if necessary.")
    # Availability alone is insufficient for unsupported AMD/NVIDIA architectures.
    value = torch.ones(1, device="cuda:0") + 1
    torch.cuda.synchronize(0)
    if value.item() != 2:
        raise RuntimeError("Accelerator computation returned an unexpected result")
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda:0")
    scores = torch.tensor([0.9, 0.8], device="cuda:0")
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    torch.cuda.synchronize(0)
    if keep.numel() != 1:
        raise RuntimeError("Accelerator torchvision NMS verification failed")
