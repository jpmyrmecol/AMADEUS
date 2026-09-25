# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Versioned wheel/support policy; runtime backend identity has no versions."""
from dataclasses import dataclass
import platform
import sys

from main.compute_backend import Backend

TORCH_VERSION = "2.7.1"
TORCHVISION_VERSION = "0.22.1"


@dataclass(frozen=True)
class WheelProfile:
    backend: Backend
    suffix: str
    index: str | None
    runtime_version: str | None = None


PROFILES = {
    "cpu": WheelProfile(Backend.CPU, "cpu", "https://download.pytorch.org/whl/cpu"),
    "macos": WheelProfile(Backend.APPLE_MPS, "", None),
    "cu128": WheelProfile(Backend.NVIDIA_CUDA, "cu128", "https://download.pytorch.org/whl/cu128", "12.8"),
    "rocm63": WheelProfile(Backend.AMD_ROCM, "rocm6.3", "https://download.pytorch.org/whl/rocm6.3", "6.3"),
}


def check_support(profile: str) -> None:
    if profile not in PROFILES:
        raise RuntimeError(f"Unknown PyTorch profile: {profile}")
    if profile == "rocm63" and not (
        sys.platform.startswith("linux") and platform.machine().lower() in {"x86_64", "amd64"}
    ):
        raise RuntimeError("The rocm63 wheel profile requires Linux x86_64.")
    if profile == "macos" and sys.platform != "darwin":
        raise RuntimeError("The macos wheel profile requires macOS.")
    if profile == "cu128" and sys.platform == "darwin":
        raise RuntimeError("The cu128 wheel profile is unavailable on macOS.")


def verify_build(profile, torch, torchvision) -> None:
    """Reject wrong vendors, runtime versions and mismatched torchvision wheels."""
    check_support(profile)
    expected = PROFILES[profile]
    for module, version in ((torch, TORCH_VERSION), (torchvision, TORCHVISION_VERSION)):
        if module.__version__.split("+")[0] != version:
            raise RuntimeError(f"Expected {version}, found {module.__version__}")
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    if expected.backend == Backend.NVIDIA_CUDA:
        valid = not hip and cuda == expected.runtime_version
    elif expected.backend == Backend.AMD_ROCM:
        valid = bool(hip) and str(hip).split(".")[:2] == expected.runtime_version.split(".") and not cuda
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
