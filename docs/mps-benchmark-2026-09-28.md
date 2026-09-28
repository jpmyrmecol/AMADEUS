# M1 Pro verification, 2026-09-28

## Scope and result

Base main: `ae1ba8f61f46e518e9251dde85d0a48228bdb8a7`; its training source
matches golden `c0e2483b8e0d486bdfb860da240fd6e89974c107`. Proposed runs used
that base plus the MPS-only patch recorded in each preflight. Dependencies:
Python 3.12.10, torch 2.7.1, torchvision 0.22.1, Ultralytics 8.3.185;
macOS 14.5, Apple M1 Pro, 32 GiB RAM, MPS available, recommended 21.33 GiB.
No MPS tuning environment variables were enabled.

**The change is a policy simplification and upstream cache backport, not a
verified sustained speedup. Neither full-data candidate completed one epoch.**
Both showed substantial slowdown and were stopped with their evidence retained.
Consequently no five-epoch run was started. The 12-minute/full-epoch goal is
unverified. Historical 13.92-hour epoch 1 and 92.23-minute resumed epoch 2 cannot
be used to calculate a controlled speedup against these partial runs.

## Full-data diagnostics

Each process loaded the existing **9,500 training + 500 validation** dataset,
imgsz 576, yolo11n-obb initial weights, SGD, seed 0, deterministic mode, original
AMADEUS augmentation (including mosaic=1, close_mosaic=0, no flips), workers=0,
no resume, configured epochs=1. New run-local image/label symlinks and caches
preserve existing datasets and results. No process shared a model/optimizer
with another candidate. Run configuration, source diff, system versions, logs
and five-second current/driver/RAM/swap samples were saved separately.

| Measurement | Old main auto | Proposed auto |
| --- | ---: | ---: |
| Selected batch | 24 | 16 |
| First 20 batches, images/s (includes warmup) | 6.65 | 5.19 |
| Batches 21–40, images/s | 4.53 | 3.35 |
| Batches 41–60, images/s | 3.71 | 2.66 |
| Completed batches at stop | 60 | 61 |
| Process wall time through stop, seconds | 344.05 | 324.39 |
| Sampled peak current allocation, GiB | 4.75 | 3.20 |
| Sampled peak driver allocation, GiB | 10.67 | 7.48 |
| Peak driver / recommended | 50.0% | 35.1% |
| Sampled minimum available RAM, GiB | 9.53 | 10.99 |
| Swap delta, MiB | -8 | 0 |
| Observed OOM | No | No |
| `nonzero` fallback warning occurrences | 1 | 0 |
| Validation reached | No | No |

Old main's estimated remaining training time at the 60-batch decision checkpoint
was 2,174 seconds (36 minutes); proposed was approximately 50 minutes. Continuing
unconditionally would provide little evidence that either reached the requested
sustained speed. The proposed run used SIGINT outside checkpoint/save I/O; the
baseline raised the diagnostic stop at a batch boundary. Logs/config/profiles
were retained; no completed epoch checkpoint exists for these partial runs.

Sampled peaks are not allocator high-water marks: events between samples may
be missed. `memory_pressure -Q` raw outputs were retained; their free percentage
is not a categorical macOS pressure-level reading. The small runs also recorded
`kern.memorystatus_vm_pressure_level=1`. System swap was already about 2 GiB at
start and did not grow during these comparisons. Free space within the currently
allocated swap files is not a fixed system swap capacity.

Background screen recording/remote desktop/WindowServer processes were active.
The OS was not rebooted between cases; order was not randomized or repeated.
Therefore this is not a clean-system hardware performance certification. Lower
memory consumption at 16 does **not** establish better throughput: it was slower
in the measured full-data windows.

## Completed one-epoch functional comparison

Separate fresh processes used the same 128 real training images and 32 real
validation images at imgsz 576. Old main was explicitly batch 16 (upstream-like
starting batch); proposed used auto, which selected 16. All other training
settings were identical. These are small functional checks, not full-data
benchmarks.

| Measurement, seconds unless specified | Old main, explicit 16 | Proposed auto 16 |
| --- | ---: | ---: |
| Training | 24.06 | 24.15 |
| Epoch validation | 20.69 | 20.49 |
| Checkpoint save | 1.77 | 1.71 |
| Trainer epoch wall time | 46.66 | 46.52 |
| Separate final best.pt validation | 14.67 | 15.30 |
| Total invocation wall time | 83.91 | 84.73 |
| Train images/s | 5.32 | 5.30 |
| Epoch validation images/s | 1.55 | 1.56 |
| Sampled peak current / driver, GiB | 2.97 / 5.44 | 2.83 / 5.63 |
| Sampled minimum available RAM, GiB | 15.63 | 15.68 |
| Swap delta | 0 | 0 |
| OOM / nonzero warning | 0 / 0 | 0 / 0 |
| NMS time-limit warnings | 4 | 4 |

Proposed pre-validation clearing reduced driver allocation 5.63 → 2.61 GiB.
At fit-epoch end the old threshold call skipped clearing; the backport executed
it (2.13 → 2.10 GiB). This verifies the intended boundary behavior, not a cure
for within-epoch slowdown. There are no additional per-batch synchronizations in
production. The diagnostic harness synchronizes at train/validation ends only.

The saved epoch1.pt EMA state dictionaries had **541/541 tensors exactly equal**
at the same batch. This checks that the cache-policy backport did not alter this
small run's trained weights. It does not certify all workloads or changing the
batch itself. Each run logged `NMS time limit 2.800s exceeded` twice in epoch
validation and twice in final validation. Those warnings can truncate NMS;
reported validation throughput and metrics are not evidence of complete,
accurate validation. NMS behavior was not changed in this patch.

## Isolated operator diagnostic

A separate baseline process profiled exactly two batch-24 steps with
`torch.profiler` CPU activities and input shapes, then stopped at the callback.
No production instrumentation or environment variable changes were made.
There was one warning, but 132 recorded `aten::nonzero` events in total. Twenty
had shapes `(24,109,6804)` or `(24,111,6804)`, above the macOS-14 2^24 threshold.
Removing nested same-thread events leaves **10 outermost large-input calls**
across two batches, consistent with the operator-specific fallback path. Their
combined CPU-observed inclusive duration was **0.333 s**. The 132-event aggregate
(including nested calls) was 0.816 s and must not be added to the outer duration.

These are CPU-side observed durations, potentially including synchronization and
transfers, not isolated GPU execution time or a full-run fallback penalty. The
profile includes warmup and changes timing. DataLoader CPU total was 0.895 s;
other CPU dispatch work was substantial. This short trace does not locate the
cause of the later slowdown. Trace, per-op table, shape counts and nesting
summary are in `baseline-b24-dispatch/`. No OOM or swap growth occurred.

## Tests and limitations

28 focused unit/contract tests pass, including 288 CUDA and 288 CPU batch cases
against c0e2483, both trainers, MPS-only thresholds, MPS OOM/cache retry and
structural invariance of unrelated training definitions/constants. Diff
whitespace checks pass. No RTX4070, Linux NVIDIA or ROCm hardware was available;
forward/backward/NMS/recovery on those devices were **not rerun**. CUDA/ROCm/CPU
runtime/profile/telemetry and lockfiles are unchanged. ROCm still uses the
provisional CUDA batch formula and the supported torch.cuda API namespace.

Full-dataset validation, five-epoch stability, thermal/GPU utilization attribution
and an optimal MPS batch remain unverified. Flat or recoverable allocation
counters plus continuing slowdown do not identify a unique root cause.

## Preserved local artifacts

Root: `~/benchmark/mps-policy-review/`.

- `baseline-full-auto/`, `proposed-full-auto-r2/`: full-data diagnostic configs,
  preflight/source diffs, events.jsonl, report.json; sibling `.log` files.
- `upstream-like-subset-b16/`, `proposed-subset-auto/`: completed checks; each has
  `results/fresh/args.yaml`, `results.csv`, and `weights/{epoch1,last,best}.pt`.
- `summary.json`, `same-batch-weight-comparison.json`, `tests.txt`,
  `background-processes.txt`, `thermal.txt`, `original-cache-hashes.json`.
- `run_benchmark.py`, `summarize.py`: external instrumentation; not production code.
- `proposed-full-auto/`: excluded first attempt, interrupted after a telemetry
  logging error. It is retained and must not be used for memory conclusions.

See [policy/source assessment](mps-training-policy.md) for official links and
reasons for preferring the minimal backport over a dependency upgrade/autotuner.
