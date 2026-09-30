# Compute backends

AMADEUS separates runtime identity (`main/compute_backend.py`), optional external
telemetry (`main/compute_telemetry.py`), batch-policy dispatch (`main/batch_utils.py`),
and versioned installation/support policy (`tools/runtime_profiles.py`). The two
training entry points share these modules. Existing `batch_utils` imports remain
compatible; `_accelerator_type()` describes the **PyTorch API namespace**, not the
vendor. Low-level embedding allocator/RNG calls using `torch.cuda` are valid for
both CUDA and HIP and intentionally retain their existing training semantics.

| Runtime backend | Current installation scope | PyTorch / YOLO device | Memory model | External telemetry | WDDM |
| --- | --- | --- | --- | --- | --- |
| NVIDIA CUDA | Windows, Linux | `cuda:0` / `0` | reserved / dedicated total | optional NVIDIA SMI | Windows only |
| AMD ROCm | supported Linux x86_64 and Windows x86_64 systems | `cuda:0` / `0` | reserved / dedicated total | allocator fallback; AMD SMI not required | disabled |
| Apple MPS | macOS Apple Silicon | `mps` | driver allocation / recommended working-set budget | allocator fallback | disabled |
| CPU | CPU fallback | `cpu` | existing system-RAM batch policy | existing psutil system telemetry | disabled |

Detection requires available CUDA-API devices and checks `torch.version.hip`
**before** `torch.version.cuda`; an unknown build is not classified as NVIDIA.
Next comes MPS availability plus a small execution probe, then CPU. `DEVICE=auto`
selects logical GPU 0, MPS, or CPU in that order. Explicit CPU remains CPU; MPS
and CUDA-style indices are validated. If a requested GPU or MPS device is
unavailable, AMADEUS prints a warning and falls back to CPU rather than failing
the tracking run. ROCm selection additionally executes a tiny operation on each
requested device: merely detecting an AMD PCI vendor or a HIP build is not a
support verdict. No `rocm:0` device is generated. Logs and the training progress
display retain backend identity separately from the device string. Invalid or
custom device strings are still left to Ultralytics validation.

The capability table owns the API namespace, memory model, telemetry provider,
worker policy and permission to request AMP. Ultralytics still performs its own
AMP checks. WDDM is an OS capability requiring **NVIDIA CUDA AND Windows**;
both the sampler and CLI supervisor gate it. Missing external measurements are
unknown (`None`), never evidence of zero memory pressure. On NVIDIA, SMI device
usage remains preferred; if unavailable, allocator reserved memory is used.
Allocator memory does not include other processes and must not be interpreted
as complete board utilization. MPS retains driver/working-set semantics; its
recommended budget is not dedicated VRAM. NVIDIA SMI queries map PyTorch's
logical device index through `CUDA_VISIBLE_DEVICES`, so a masked or reordered
Linux GPU is not confused with another card.

Each of CUDA, ROCm, MPS and CPU has a separate batch-policy entry in
`BATCH_POLICIES`. CUDA and ROCm use the established total-device-memory
estimator, preserving the pre-`6af3f83` NVIDIA batch behavior. This does not
assert that CUDA and ROCm have the same optimal batch. Runtime OOM, WDDM and
MPS recovery remain responsible for correcting an unsafe automatic batch;
fixed user-specified batches are preserved. MPS and CPU keep their separate
policies. Resume, optimizer state, checkpoint selection and recovery
thresholds are unchanged. The inactive preflight calibration module is not
re-enabled.

For the subsequent MPS-only training policy revision, see
[MPS training policy](mps-training-policy.md). The historical validation below
describes the original backend separation, before that revision.

## Installation and migration

The current profiles are `cpu`, `macos`, `cu126`, `rocm72`, and
`rocmwin100`. CPU, macOS, NVIDIA and Linux AMD use torch **2.14.0** and
torchvision **0.29.0**. NVIDIA uses the official CUDA 12.6 wheel index; Linux
AMD uses the official ROCm 7.2 wheel index and pins **triton-rocm 3.8.0** from
that same index.

Windows AMD is intentionally separate. The `rocmwin100` profile uses AMD's
official Windows ROCm 10.0 wheel index with Python **3.12**, torch
**2.13.0+rocm10.0.0**, and torchvision **0.28.0+rocm10.0.0**. The
`device-all` extras install the required AMD runtime components for supported
hardware. The normal locked AMADEUS dependencies are synced first while
torch/torchvision remain separately managed. Entering or leaving the Windows AMD
profile rebuilds the managed Windows virtual environment so CUDA and ROCm
runtime packages are not mixed.

Runtime identity does not depend on profile names. PyTorch HIP continues to use
the `cuda` API/device namespace, so the existing AMD runtime backend is shared
between Linux and Windows.

Setup chooses macOS wheels on macOS, then an NVIDIA driver candidate, then a Linux
AMD candidate (`/dev/kfd` plus AMD DRM vendor), then a Windows AMD display
adapter, otherwise CPU. Windows AMD selects `rocmwin100` rather than CPU. NVIDIA discovery
checks PATH plus common native-Linux locations and the WSL2 GPU bridge at
`/usr/lib/wsl/lib/nvidia-smi`; runtime telemetry uses the same candidate list.
When CUDA is selected, setup reports the detected GPU, driver version and
`nvidia-smi` path. A failed CUDA verification includes the same driver diagnostic
and the verifier output so driver/runtime incompatibilities are distinguishable
from package errors. This is only wheel selection. Build/runtime/suffix/version
verification, GPU tensor computation and torchvision GPU NMS must succeed before
setup writes a ready marker. Unsupported AMD hardware is never accepted solely
from its vendor or `is_available()`.
This smoke verification cannot certify every operator on every GPU: the AMD GPU,
driver and OS compatibility matrix still applies. No compatibility override such
as `HSA_OVERRIDE_GFX_VERSION` is installed.

Override selection when necessary, for example:

```sh
AMADEUS_TORCH_PROFILE=rocm72 ./AMADEUS.sh
AMADEUS_TORCH_PROFILE=cpu ./AMADEUS.sh
```

An unusable requested GPU profile fails verification and does not become ready;
use the CPU profile to install a CPU fallback environment. AMD SMI is optional
and is currently not installed or queried. Its future provider belongs in
`compute_telemetry.py`, without changing model training.

Setup preserves the existing locked, inexact sync and isolated torch/torchvision
repair. CPU, macOS, CUDA and Linux ROCm keep the existing `--no-deps` pair
repair after locked runtime dependencies are synced. Windows AMD instead installs
AMD's official `device-all` runtime dependencies together with its exact
torch/torchvision pair from the AMD index. The new JSON ready marker stores the
profile and hashes of the installation inputs. Old empty/timestamp-only markers
require one full setup. The Windows fast path also verifies the actual installed
wheel pair and execution every time, so swapping wheels after a successful setup
cannot silently pass. POSIX launchers already run setup every time. This adds
verification latency to the former Windows timestamp-only fast path.

## Validation

Windows NVIDIA has been validated on actual hardware. Linux NVIDIA, Linux AMD,
macOS Apple Silicon, and Windows AMD use separate runtime profiles as described
above. Windows AMD support should still be validated on a supported AMD GPU with
a representative tracking and training workload.

## Changed paths

- New: `main/compute_backend.py`, `main/compute_telemetry.py`, `tools/runtime_profiles.py`.
- Integration: `main/batch_utils.py`, both `obb_detector_training.py` files,
  `gui/gui_easy_tracking.py`, `tools/setup_environment.py`, `AMADEUS.bat`.
- Packaging/update: `pyproject.toml`, `uv.lock`, `tools/update_manifest.txt`.
- Documentation: this document, `README.md`.

## Specification references

- [PyTorch HIP semantics](https://docs.pytorch.org/docs/2.7/notes/hip.html): HIP uses the `cuda` namespace; inspect build metadata for identity.
- [Official previous-version wheels](https://pytorch.org/get-started/previous-versions/#v271): PyTorch 2.7.1 / torchvision 0.22.1 with ROCm 6.3 on Linux.
- [AMD ROCm 6.3.3 compatibility](https://rocm.docs.amd.com/en/docs-6.3.3/compatibility/compatibility-matrix.html): GPU, OS and framework compatibility are separate requirements; official PyTorch wheel availability alone is not AMD certification of every combination.
- [PyTorch MPS APIs](https://docs.pytorch.org/docs/2.7/mps.html): allocator and recommended working-set measurements.
- [Pinned Ultralytics device selection](https://github.com/ultralytics/ultralytics/blob/v8.3.185/ultralytics/utils/torch_utils.py): CUDA-style devices reach the shared PyTorch API.
- [AMD Windows PyTorch installation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html): official wheel and Python requirements differ from this repository's locked Linux ROCm profile.
- [AMD Windows compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html): supported GPU and OS matrix for the Windows PyTorch distribution.
