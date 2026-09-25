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
| AMD ROCm | supported Linux x86_64 systems | `cuda:0` / `0` | reserved / dedicated total | allocator fallback; AMD SMI not required | disabled |
| Apple MPS | macOS Apple Silicon | `mps` | driver allocation / recommended working-set budget | allocator fallback | disabled |
| CPU | CPU fallback | `cpu` | existing system-RAM batch policy | existing psutil system telemetry | disabled |

Detection requires available CUDA-API devices and checks `torch.version.hip`
**before** `torch.version.cuda`; an unknown build is not classified as NVIDIA.
Next comes MPS availability plus a small execution probe, then CPU. `DEVICE=auto`
selects logical GPU 0, MPS, or CPU in that order. Explicit CPU remains CPU; MPS
and CUDA-style indices are validated and fall back to CPU when unavailable.
ROCm selection additionally executes a tiny operation on each requested device:
merely detecting an AMD PCI vendor or a HIP build is not a support verdict.
No `rocm:0` device is generated. Logs and the training progress display retain
backend identity separately from the device string.

The capability table owns the API namespace, memory model, telemetry provider,
worker policy and permission to request AMP. Ultralytics still performs its own
AMP checks. WDDM is an OS capability requiring **NVIDIA CUDA AND Windows**;
both the sampler and CLI supervisor gate it. Missing external measurements are
unknown (`None`), never evidence of zero memory pressure. On NVIDIA, SMI device
usage remains preferred; if unavailable, allocator reserved memory is used.
Allocator memory does not include other processes and must not be interpreted
as complete board utilization. MPS retains driver/working-set semantics; its
recommended budget is not dedicated VRAM.

Each of CUDA, ROCm, MPS and CPU has a separate batch-policy entry in
`BATCH_POLICIES`. Initially they reuse the established estimator where applicable;
this does not assert that CUDA and ROCm have the same optimal batch. No new batch
constants, MPS thresholds or performance workarounds were introduced. Resume,
optimizer state, checkpoint selection and recovery thresholds are unchanged.
The inactive preflight calibration module is not re-enabled by this refactor.

## Installation and migration

The existing `cpu`, `macos`, and `cu128` extras retain their names. New `rocm63`
uses official torch **2.7.1+rocm6.3**, torchvision **0.22.1+rocm6.3**, and their
**pytorch-triton-rocm 3.3.1** dependency. The latter is explicitly sourced from
PyTorch's shared wheel index. Existing lockfile package versions are unchanged.
Runtime identity does not depend on these names or versions. The Linux restriction
belongs to this wheel profile, not to the AMD runtime backend.

Setup chooses macOS wheels on macOS, then an NVIDIA driver candidate, then a Linux
AMD candidate (`/dev/kfd` plus AMD DRM vendor), otherwise CPU. This is only wheel
selection. Build/runtime/suffix/version verification, GPU tensor computation and
torchvision GPU NMS must succeed before setup writes a ready marker. Unsupported
AMD hardware is never accepted solely from its vendor or `is_available()`.
This smoke verification cannot certify every operator on every GPU: the AMD GPU,
driver and OS compatibility matrix still applies. No compatibility override such
as `HSA_OVERRIDE_GFX_VERSION` is installed.

Override selection when necessary, for example:

```sh
AMADEUS_TORCH_PROFILE=rocm63 ./AMADEUS.sh
AMADEUS_TORCH_PROFILE=cpu ./AMADEUS.sh
```

An unusable requested GPU profile fails verification and does not become ready;
use the CPU profile to install a CPU fallback environment. AMD SMI is optional
and is currently not installed or queried. Its future provider belongs in
`compute_telemetry.py`, without changing model training.

Setup preserves the existing locked, inexact sync and isolated `--no-deps`
torch/torchvision repair. Runtime dependencies (including ROCm Triton) are synced
from the lock before repairing that pair. The new JSON ready marker stores the
profile and hashes of the installation inputs. Old empty/timestamp-only markers
require one full setup. The Windows fast path also verifies the actual installed
wheel pair and execution every time, so swapping wheels after a successful setup
cannot silently pass. POSIX launchers already run setup every time. This adds
verification latency to the former Windows timestamp-only fast path.

## Validation (2026-09-25)

Base: repository main `07e6775`. No test files were present in that checkout
(despite historical test paths in the update manifest). Twenty focused unittest tests provide
coverage includes detection, device selection, vendor/OS telemetry isolation,
both training samplers, memory/cache/sync, batch dispatch, OOM retries, setup
selection, incompatible wheels, repair and ready-marker rejection.

```sh
python -m unittest discover -s tests -v
uv lock --check --offline
python tests/smoke_compute_backend.py auto
python tests/smoke_compute_backend.py cpu
```

Actual hardware: Windows, NVIDIA GeForce RTX 4070; existing environment with
PyTorch 2.7.1+cu128 and torchvision 0.22.1+cu128. GPU and explicit CPU passed
synthetic Ultralytics forward/backward, optimizer step, torchvision NMS, memory,
cache and synchronization. The CPU smoke used the CPU device of that CUDA build,
not a separately installed CPU wheel. Exact CUDA profile verification passed.
NVIDIA SMI returned values; WDDM returned unknown on this machine, so actual spill
recovery was **not** reproduced. No sustained/full dataset training is claimed.

Batch estimates were additionally compared against the unchanged main version
for 288 CPU/MPS/CUDA combinations of RAM, device budget, image size and mode;
all matched. Locked installer dry-runs passed for Windows CPU/CUDA, Linux
CUDA/ROCm and macOS arm64 profiles; the ROCm plan contains ROCm Triton and no
NVIDIA packages. Nine resume/checkpoint/monitor definitions in each training module
were checked structurally unchanged. ROCm, MPS and Linux NVIDIA are covered by
mock/unit verification only; hardware training and full installer execution on
those systems remain unverified. A real supported AMD machine should run the
manual smoke and a representative training/resume workload before rollout.

## Changed paths

- New: `main/compute_backend.py`, `main/compute_telemetry.py`, `tools/runtime_profiles.py`.
- Integration: `main/batch_utils.py`, both `obb_detector_training.py` files,
  `gui/gui_easy_tracking.py`, `tools/setup_environment.py`, `AMADEUS.bat`.
- Packaging/update: `pyproject.toml`, `uv.lock`, `tools/update_manifest.txt`.
- Validation/docs: `tests/test_compute_backend.py`, `tests/test_runtime_profiles.py`,
  `tests/smoke_compute_backend.py`, this document, `README.md`.

## Specification references

- [PyTorch HIP semantics](https://docs.pytorch.org/docs/2.7/notes/hip.html): HIP uses the `cuda` namespace; inspect build metadata for identity.
- [Official previous-version wheels](https://pytorch.org/get-started/previous-versions/#v271): PyTorch 2.7.1 / torchvision 0.22.1 with ROCm 6.3 on Linux.
- [AMD ROCm 6.3.3 compatibility](https://rocm.docs.amd.com/en/docs-6.3.3/compatibility/compatibility-matrix.html): GPU, OS and framework compatibility are separate requirements; official PyTorch wheel availability alone is not AMD certification of every combination.
- [PyTorch MPS APIs](https://docs.pytorch.org/docs/2.7/mps.html): allocator and recommended working-set measurements.
- [Pinned Ultralytics device selection](https://github.com/ultralytics/ultralytics/blob/v8.3.185/ultralytics/utils/torch_utils.py): CUDA-style devices reach the shared PyTorch API.
