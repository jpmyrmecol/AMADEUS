# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Optional external providers; missing tools never prevent training.

AMD currently uses allocator measurements. An AMD SMI provider can be added
here without changing training or making its installation mandatory.
"""
import os
import shutil
import subprocess
from pathlib import Path
try:
    from .compute_backend import runtime_for
except ImportError:
    from compute_backend import runtime_for


def _nvidia_smi_id(logical_index: int) -> str:
    """Map a PyTorch logical index through CUDA_VISIBLE_DEVICES when set."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [part.strip() for part in visible.split(",")]
        if 0 <= logical_index < len(devices) and devices[logical_index]:
            return devices[logical_index]
    return str(logical_index)


def nvidia_smi_candidates() -> list[str]:
    """Return usable nvidia-smi candidates on Windows, native Linux and WSL2."""
    candidates: list[Path | str] = []
    discovered = shutil.which("nvidia-smi")
    if discovered:
        candidates.append(discovered)
    else:
        # Keep the command name as a last-resort PATH lookup. This also makes
        # diagnostics resilient when PATH changes after Python starts.
        candidates.append("nvidia-smi")

    if os.name == "nt":
        windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        program_files = Path(
            os.environ.get("ProgramW6432", os.environ.get("ProgramFiles", r"C:\Program Files"))
        )
        candidates.extend(
            [
                windows_dir / "System32" / "nvidia-smi.exe",
                program_files / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
            ]
        )
        driver_store = windows_dir / "System32" / "DriverStore" / "FileRepository"
        if driver_store.is_dir():
            candidates.extend(driver_store.glob("nv*/*nvidia-smi.exe"))
    else:
        # /usr/lib/wsl/lib is the standard WSL2 GPU bridge location and is not
        # guaranteed to be on PATH. /usr/bin covers normal distro packages.
        candidates.extend(
            [
                Path("/usr/bin/nvidia-smi"),
                Path("/usr/local/bin/nvidia-smi"),
                Path("/usr/lib/wsl/lib/nvidia-smi"),
            ]
        )

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = str(candidate)
        if os.path.isabs(value) and not Path(value).is_file():
            continue
        key = os.path.normcase(os.path.abspath(value)) if os.path.isabs(value) else value
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def query_nvidia_driver_info() -> tuple[str, str, str] | None:
    """Return (GPU name, driver version, nvidia-smi executable) for the first working GPU."""
    for executable in nvidia_smi_candidates():
        try:
            out = subprocess.check_output(
                [
                    executable,
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3.0,
            ).strip()
        except Exception:
            continue
        first = out.splitlines()[0].strip() if out else ""
        if not first:
            continue
        parts = [part.strip() for part in first.split(",", 1)]
        if len(parts) != 2 or not all(parts):
            continue
        return parts[0], parts[1], executable
    return None


def query_nvidia_smi(device) -> tuple[float | None, float | None, float | None]:
    """Return (gpu_util_percent, vram_used_gib, vram_total_gib)."""
    runtime = runtime_for(device)
    if runtime.capabilities.telemetry != "nvidia":
        return None, None, None
    idx = runtime.index
    if idx is None:
        return None, None, None
    for executable in nvidia_smi_candidates():
        try:
            out = subprocess.check_output(
                [
                    executable,
                    f"--id={_nvidia_smi_id(idx)}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=2.0,
            ).strip()
            first = out.splitlines()[0]
            util_s, used_s, total_s = [x.strip() for x in first.split(",")[:3]]
            return float(util_s), float(used_s) / 1024.0, float(total_s) / 1024.0
        except Exception:
            continue
    return None, None, None


def query_wddm_non_local_usage(pid: int | None = None, device=None) -> float | None:
    """Return one process's WDDM 'Non Local Usage' (GiB): the GPU
    memory segment Windows' WDDM driver keeps off the GPU adapter itself
    (for a discrete GPU, system RAM reached over the bus rather than
    on-die VRAM), summed across every adapter/engine instance this
    process holds. Defaults to this process; the CLI supervisor supplies its
    CUDA child's PID so monitoring remains live even when that child holds the
    GIL inside a long CUDA call.

    This is the same pool NVIDIA's driver falls back to under the "CUDA
    Sysmem Fallback Policy" when a dedicated-VRAM allocation can't be
    satisfied: rather than raising an out-of-memory error, the allocation
    silently lands here instead, which is drastically slower (PCIe-speed
    access instead of on-die VRAM) -- this is the real mechanism behind a
    training process that neither errors nor visibly runs out of VRAM, yet
    slows to a crawl. Windows-only (WDDM has no equivalent on Linux/macOS);
    returns None there, and on any query failure (counter unavailable, no
    matching instance for this PID, timeout, etc.) -- callers must treat
    None as "unknown", never as "zero usage".
    """
    if not runtime_for(device).wddm:
        return None
    target_pid = os.getpid() if pid is None else int(pid)
    script = (
        "(Get-Counter -Counter '\\GPU Process Memory(*)\\Non Local Usage' "
        "-ErrorAction SilentlyContinue).CounterSamples "
        f"| Where-Object {{ $_.InstanceName -like 'pid_{target_pid}_*' }} "
        "| Measure-Object -Property CookedValue -Sum "
        "| Select-Object -ExpandProperty Sum"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5.0,
        ).strip()
        if not out:
            return 0.0
        return float(out) / (1024.0 ** 3)
    except Exception:
        return None


def sample_accelerator(device):
    runtime = runtime_for(device)
    if runtime.capabilities.telemetry == "nvidia":
        util, used, total = query_nvidia_smi(device)
        if used is not None and total:
            return util, used, total
    used, total = runtime.memory()
    gib = 1024.0 ** 3
    return None, None if used is None else used / gib, None if total is None else total / gib


def print_device_summary(device=None):
    runtime = runtime_for(device)
    util, _, _ = sample_accelerator(device)
    print("[AMADEUS] Backend:", runtime.backend.value)
    print("[AMADEUS] PyTorch device:", runtime.torch_device)
    print("[AMADEUS] GPU utilization:", "unavailable" if util is None else f"{util}%")
