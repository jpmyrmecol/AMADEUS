# MPS training policy

This change is based on AMADEUS `ae1ba8f` (training code identical to the
Windows NVIDIA golden baseline `c0e2483`). It changes MPS training only. CUDA,
HIP, CPU, inference batch estimation, dependency versions and environment
variables retain their existing policy.

## Decision

Use **A: upstream-compatible default batch 16** for fresh MPS `BATCH_SIZE=auto`.
This is a predictable starting point, **not a throughput optimum or a guarantee
that every model fits**. Explicit batch and checkpoint batch still take priority.
Real OOM retains the existing recovery path. Workers remain zero on MPS.
There is no candidate search or epoch-wise memory-driven batch resizing.

| Option | Assessment |
| --- | --- |
| A. Upstream-like default | Adopted: small change, reproducible, no profiling overhead or candidate state contamination |
| B. RAM/density/working-set estimate | Rejected for MPS training: available unified RAM is not a throughput predictor; short batch-24 evidence did not establish sustained stability |
| C. Sustained candidate search | Deferred: expensive, requires independent processes/order control, and previous runs did not identify a reliable winner |
| D. Larger fixed default or per-batch cache clearing | No supporting sustained evidence; not adopted |

Removed MPS training decisions: RAM 60% / default cap 24 / density-based initial
estimate; driver/recommended >=95% batch reduction; system RAM percentage-only
batch reduction; <40% memory-headroom batch increase. The old estimator remains
for inference; CPU's 50% / cap 16 policy is unchanged. No 7 GiB free-RAM rule,
90% driver rule, or new ROCm constants were introduced.

PyTorch owns allocator limits. AMADEUS still owns explicit configuration,
logging, checkpoint selection and existing OOM/retry handling. Removing a
ratio-only recovery does not establish that swap growth is harmless: sustained
benchmarks must inspect actual pressure, swap trend and throughput and stop
safely when the experiment ceases to be informative. This change does not add a
new production macOS pressure daemon or claim protection against all system-wide
memory exhaustion. Explicit batch retains existing failure semantics.

## Pinned upstream evidence

- [Ultralytics 8.3.185 AutoBatch](https://github.com/ultralytics/ultralytics/blob/v8.3.185/ultralytics/utils/autobatch.py)
  returns its default (16) on CPU/MPS without CUDA profiling. Current upstream
  [AutoBatch](https://github.com/ultralytics/ultralytics/blob/0ea465b1016fbb1b88b24f521af41fe4a0352545/ultralytics/utils/autobatch.py)
  keeps the CPU/MPS fallback.
- [Pinned trainer](https://github.com/ultralytics/ultralytics/blob/v8.3.185/ultralytics/engine/trainer.py)
  calls `_clear_memory(0.5)` before validation and after each fit epoch.
  For MPS the fraction is **system RAM percentage**, not dedicated GPU memory.
- [PR 24038](https://github.com/ultralytics/ultralytics/pull/24038), referencing
  [issue 22621](https://github.com/ultralytics/ultralytics/issues/22621), makes those
  calls unconditional on MPS. AMADEUS backports this by overriding
  `_clear_memory` in its two existing OBB trainers and passing `None` only on
  MPS. Existing invocation boundaries, garbage collection and cache clearing
  remain upstream-owned. No per-batch `empty_cache` or `synchronize` is added.
  CUDA/HIP/CPU receive the original threshold unchanged. Dependency upgrade
  would involve many unrelated trainer changes and is unnecessary for this fix.
- [Issue 24065](https://github.com/ultralytics/ultralytics/issues/24065) reports
  continuing swap growth on large datasets. An upstream cache fix is not proof
  that all MPS memory or performance issues are solved.
- [PyTorch 2.7.1 allocator header](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/mps/MPSAllocator.h)
  and [implementation](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/mps/MPSAllocator.mm)
  define high watermark **1.7**, unified-memory low watermark **1.4** (discrete
  **1.0**). Recommended working set is not hard dedicated VRAM capacity. Low
  watermark triggers garbage collection/adaptive commit; high watermark limits
  allocation. These are verified at 2.7.1, not inferred from current docs.

## Environment and fallback

The [2.7.1 environment documentation](https://github.com/pytorch/pytorch/blob/v2.7.1/docs/source/mps_environment_variables.rst)
confirms HIGH/LOW_WATERMARK control the allocator; PREFER_METAL selects Metal
rather than MPSGraph for matmul; FAST_MATH can change numerical precision; and
ENABLE_MPS_FALLBACK enables generic unsupported-op CPU fallback. None is set or
overridden by this change. In particular high watermark is not disabled and
FAST_MATH is not enabled for speed.

[2.7.1 Indexing.mm](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/mps/operations/Indexing.mm)
contains an operator-specific CPU fallback for `nonzero` on macOS 14 when the
input has at least **2^24 elements** (also other unsupported input cases).
This path does not require ENABLE_MPS_FALLBACK. OBB TaskAlignedAssigner uses
boolean indexing of batch × GT × anchors arrays, making this workload relevant.
`TORCH_WARN_ONCE` means one warning is not one fallback invocation. Warning
counts alone cannot quantify its runtime contribution. No replacement assigner,
shape padding or numerical/model changes are introduced here.

## What the existing M1 evidence establishes

Earlier batch 24/32/40/44/48 runs degraded with time. In the batch-32 diagnostic,
a cache clear reduced driver allocation from 15.62 to 3.59 GiB without restoring
speed (3.51 to 3.29 images/s for adjacent windows). Other runs slowed while
current/driver allocations were approximately flat. Thus allocator accumulation
is not a demonstrated sufficient explanation for all observed slowdown.
Transient allocation, graph/backend behavior, CPU work, thermal effects and
system activity remain possible contributors. The epoch-boundary backport must
not be advertised as an established fix for within-epoch throughput collapse.

If finer attribution is needed, use a **separate instrumented benchmark**:
[PyTorch MPS profiler/signposts](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/mps/profiler.py)
and Instruments, or synchronized boundaries around
loading/transfer/forward-loss/backward/optimizer-EMA. Python dispatch durations
alone do not measure asynchronous GPU stages. Do not add those synchronization
barriers to production training. Independent process, same seed/data order and
fresh model/optimizer state are required for batch comparisons; fresh process
alone does not reset OS swap/thermal state.

## Regression contract

`python -m unittest discover -s tests -v` includes 288 CUDA and 288 CPU
batch-estimator comparisons with the exact `c0e2483` source; MPS policy tests;
CUDA/CPU initial-load verdict comparisons; MPS-only cache threshold forwarding;
and structural checks for all other training definitions/constants in both
entry points. These cover unchanged resume/save, optimizer/AMP arguments,
worker/WDDM/throughput/fallback/OOM paths. The baseline must be in Git history
(use a full clone or fetch it in shallow CI).

The backend/profile/telemetry modules and lockfile are byte-identical to golden
main. ROCm still provisionally uses the CUDA memory formula and `torch.cuda`
allocator/OOM API namespace, as PyTorch HIP requires. No Linux NVIDIA or ROCm
performance changes are intended. Mock contracts are not RTX4070/ROCm hardware
validation; this Mac cannot certify those devices.

Actual MPS measurements and limitations are recorded in the accompanying
[benchmark report](mps-benchmark-2026-09-28.md).
