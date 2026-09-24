# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""GPU batch size and CPU worker heuristics, plus OOM-retry utilities.

Usage
-----
from batch_utils import resolve_batch_size, run_with_oom_retry, resolve_num_workers

batch = resolve_batch_size(
    cfg_value,
    image_size,
    device,
    mode="train",
    labels_dir="path/to/train/labels",
    task="obb",
)
result = run_with_oom_retry(lambda bs: model.train(batch=bs, ...), batch)
workers = resolve_num_workers(cfg_value, task="process")
"""
from __future__ import annotations

import os

TQDM_BAR_FORMAT = "{l_bar}{bar:20}{r_bar}"


def tqdm(*args, **kwargs):
    from tqdm import tqdm as _tqdm

    if "ascii" not in kwargs:
        kwargs["ascii"] = " -"
    if "bar_format" not in kwargs:
        kwargs["bar_format"] = TQDM_BAR_FORMAT
    if "colour" not in kwargs:
        kwargs["colour"] = "green"
    return _tqdm(*args, **kwargs)


def _resolve_cuda_index(device) -> tuple[bool, int]:
    """Return (is_cuda, device_index)."""
    s = str(device).strip().lower()
    if s == "cpu":
        return False, 0
    if s.isdigit():
        return True, int(s)
    if s.startswith("cuda:"):
        tail = s[5:]
        return True, int(tail) if tail.isdigit() else 0
    if s == "cuda":
        return True, 0
    return False, 0


def _resolve_cuda_indices(device) -> list[int]:
    """Return requested CUDA indices for YOLO-style device strings."""
    s = str(device).strip().lower()
    if s == "cuda":
        return [0]
    if s.startswith("cuda:"):
        s = s[5:]
    if not s:
        return []
    if all(part.strip().isdigit() for part in s.split(",")):
        return [int(part.strip()) for part in s.split(",") if part.strip()]
    return []


def _cuda_device_count() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _mps_available() -> bool:
    """Return whether the PyTorch MPS backend can execute a small operation."""
    try:
        import torch

        if not torch.backends.mps.is_available():
            return False
        value = torch.ones(1, device="mps") + 1
        torch.mps.synchronize()
        return value.item() == 2
    except Exception:
        return False


def resolve_device(device="auto", purpose: str = "YOLO") -> str:
    """Resolve a configured device to a value accepted by Ultralytics.

    Existing configs commonly store ``0`` as the default device.  On machines
    without a CUDA-visible NVIDIA GPU, passing that value directly to
    Ultralytics raises before training or prediction starts, so CUDA-like
    requests fall back to CPU when Torch cannot use them.

    "auto" prefers CUDA, then Apple Silicon MPS, then CPU. An explicit "mps"
    request is validated the same way explicit CUDA requests already are,
    falling back to CPU with a warning when MPS isn't actually usable.
    Ultralytics accepts "mps" as a device string natively, so the resolved
    value is passed straight through -- no separate MPS code path is needed
    downstream for YOLO itself.
    """
    requested = str(device).strip()
    requested_lower = requested.lower()
    if requested_lower in {"", "auto", "none"}:
        count = _cuda_device_count()
        if count > 0:
            resolved = "0"
            backend = f"cuda:{resolved}"
        elif _mps_available():
            resolved = "mps"
            backend = "mps"
        else:
            resolved = "cpu"
            backend = "cpu"
        print(f"[INFO] {purpose} device auto-selected: {resolved} ({backend})")
        return resolved

    if requested_lower == "cpu":
        return "cpu"

    if requested_lower == "mps":
        if _mps_available():
            return "mps"
        print(f"[WARN] {purpose} device='mps' requested, but torch.backends.mps.is_available() is False; using CPU.")
        return "cpu"

    cuda_indices = _resolve_cuda_indices(requested)
    requests_cuda = requested_lower == "cuda" or requested_lower.startswith("cuda:") or bool(cuda_indices)
    if not requests_cuda:
        return requested

    count = _cuda_device_count()
    if count <= 0:
        print(
            f"[WARN] {purpose} device='{requested}' requests CUDA, "
            "but torch.cuda.is_available() is False; using CPU."
        )
        return "cpu"

    invalid = [idx for idx in cuda_indices if idx < 0 or idx >= count]
    if invalid:
        print(
            f"[WARN] {purpose} device='{requested}' requests unavailable CUDA index "
            f"{invalid}; torch sees {count} CUDA device(s). Using CPU."
        )
        return "cpu"

    return requested


# --- Accelerator utilities (CUDA/MPS/CPU) ------------------------------------
#
# Small, explicit helpers so device-specific memory/cache/sync calls live in
# one place instead of being repeated as raw torch.cuda.* calls throughout
# the codebase. CUDA behavior is always the exact pre-existing torch.cuda.*
# call; MPS is the only backend these helpers add support for; CPU is always
# a no-op.


def _accelerator_type(device=None) -> str:
    """Return 'cuda', 'mps', or 'cpu' for a device spec, torch.device, or None.

    None resolves to whichever accelerator is actually active (CUDA first,
    then MPS), matching resolve_device()'s auto-selection order.
    """
    try:
        import torch
    except Exception:
        return "cpu"

    if isinstance(device, torch.device):
        return device.type

    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    s = str(device).strip().lower()
    if s == "mps":
        return "mps"
    if s == "cpu":
        return "cpu"
    is_cuda, _idx = _resolve_cuda_index(device)
    if is_cuda or _resolve_cuda_indices(device):
        return "cuda"
    return "cpu"


def effective_dataloader_workers(device, configured_workers: int) -> int:
    """Return the workers Ultralytics actually uses for this backend.

    Ultralytics 8.3.185 forces ``args.workers`` to zero for CPU and MPS
    devices.  Keep the configured value available for retry bookkeeping, but
    use this value for diagnostics and memory-pressure decisions.
    """
    configured = max(0, int(configured_workers))
    return 0 if _accelerator_type(device) in {"cpu", "mps"} else configured


def empty_accelerator_cache(device=None) -> None:
    """Clear the CUDA or MPS caching allocator; a no-op on CPU.

    CUDA keeps its exact previous torch.cuda.empty_cache() behavior; the only
    thing this adds is the MPS branch (torch.mps.empty_cache()), so an
    OOM-retry that used to silently skip cache-clearing on Apple Silicon now
    actually frees memory before the next attempt, same as it already did
    for CUDA.
    """
    try:
        import torch
    except Exception:
        return

    kind = _accelerator_type(device)
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif kind == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def synchronize_accelerator(device=None) -> None:
    """torch.cuda.synchronize() / torch.mps.synchronize(); a no-op on CPU."""
    try:
        import torch
    except Exception:
        return

    kind = _accelerator_type(device)
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif kind == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def get_accelerator_memory(device=None) -> tuple[float | None, float | None]:
    """Return (used_bytes, budget_bytes) for the active accelerator, or (None, None).

    CUDA: (reserved, total) via torch.cuda.memory_reserved()/device properties
    -- unchanged from what callers computed inline before.
    MPS: (driver_allocated_memory(), recommended_max_memory()), the pair
    Apple/PyTorch expose for judging unified-memory headroom; this is a
    distinct signal from plain system RAM usage (which callers already get
    separately via psutil) since not all RAM pressure comes from the MPS
    allocator.
    """
    try:
        import torch
    except Exception:
        return None, None

    kind = _accelerator_type(device)
    if kind == "cuda" and torch.cuda.is_available():
        try:
            is_cuda, idx = _resolve_cuda_index(device if device is not None else "cuda")
            idx = idx if is_cuda else torch.cuda.current_device()
            used = float(torch.cuda.memory_reserved(idx))
            total = float(torch.cuda.get_device_properties(idx).total_memory)
            return used, total
        except Exception:
            return None, None
    if kind == "mps":
        try:
            used = float(torch.mps.driver_allocated_memory())
            total = float(torch.mps.recommended_max_memory())
            return used, total
        except Exception:
            return None, None
    return None, None


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _cuda_total_gib(device) -> float | None:
    """Return CUDA device total memory in GiB."""
    try:
        import torch

        is_cuda, idx = _resolve_cuda_index(device)
        if not is_cuda or not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(idx).total_memory / (1024.0 ** 3)
    except Exception:
        return None


def _mps_recommended_gib(device) -> float | None:
    """Return the MPS recommended working-set size in GiB when available."""
    if _accelerator_type(device) != "mps":
        return None
    _used_bytes, recommended_bytes = get_accelerator_memory(device)
    if recommended_bytes is None or recommended_bytes <= 0:
        return None
    return recommended_bytes / (1024.0 ** 3)


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile of `values` (0 <= pct <= 100).

    pct=100 is equivalent to the maximum value; this lets a single statistic
    cover both the "percentile" and "maximum" density options via one
    environment variable (AMADEUS_LABEL_DENSITY_PERCENTILE).
    """
    if not values:
        return 0.0
    s = sorted(values)
    idx = (pct / 100.0) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _estimate_yolo_label_density(
    labels_dir: str | None, max_files: int = 2000, percentile: float = 95.0
) -> tuple[float, float, int, int] | None:
    """Estimate per-image label density from YOLO .txt label files.

    Returns (percentile_labels_per_image, mean_labels_per_image,
    total_label_files, sampled_label_files).

    The percentile statistic (not the mean) is what feeds the batch-size cost
    model: a dataset can have a low mean instance count while still containing
    a minority of label-dense images, and because of mosaic augmentation a
    single such image landing in a batch is enough to spike VRAM usage
    regardless of the dataset-wide average.
    """
    if not labels_dir or not os.path.isdir(labels_dir):
        return None

    files = sorted(
        os.path.join(labels_dir, name)
        for name in os.listdir(labels_dir)
        if name.lower().endswith(".txt")
    )
    if not files:
        return None

    if len(files) > max_files:
        step = max(1, len(files) // max_files)
        sampled = files[::step][:max_files]
    else:
        sampled = files

    counts: list[int] = []
    for path in sampled:
        with open(path, "r", encoding="utf-8") as f:
            counts.append(sum(1 for line in f if line.strip()))

    mean_labels = sum(counts) / max(1, len(counts))
    pct_labels = _percentile(counts, percentile)
    return pct_labels, mean_labels, len(files), len(sampled)


def _batch_size_from_memory(
    total_gib: float,
    image_size: int,
    mode: str,
    density: float | None,
    task: str,
) -> tuple[int, float, float]:
    """Unified per-sample VRAM cost model.

        per_sample_cost = area_factor(image_size) * mode_factor(mode) * density_factor(density, mode, task)
        batch           = floor(total_gib / per_sample_cost)

    area_factor(image_size) = (image_size / 640)^2
        Pixel-count scaling, calibrated to 1.0 at the reference resolution
        (640 px).

    mode_factor(mode) = 1/3 (train) or 1/6 (infer)
        Reciprocal of the previous per-GiB batch throughput constants
        (factor=3 for train, factor=6 for infer), so that
        total_gib / (area_factor * mode_factor) reproduces the same value the
        image-size-only heuristic used to give at density=0.

    density_factor(density, mode, task) = 1 + density / AMADEUS_LABEL_DENSITY_HALF
        Applies only when mode == "train" and task in {detect, obb};
        identically 1.0 otherwise (including whenever density is None, e.g.
        no labels_dir was supplied). AMADEUS_LABEL_DENSITY_HALF (default 200
        labels/image) is the density at which density_factor doubles, halving
        the resulting batch. The mode/task gate is enforced inside this
        function itself, not left to the caller, so a direct call with an
        inference mode or a non-detection task cannot accidentally apply a
        density penalty that was never intended for it.

    Because density enters the same product as image resolution and GPU
    memory, instance count is a first-class term in the per-sample memory
    cost rather than a correction applied after the fact -- doubling label
    density has the same mathematical status as doubling image resolution.

    Returns (batch, per_sample_cost, density_factor) for logging/verification.
    """
    area_factor = (float(image_size) / 640.0) ** 2
    mode_factor = 1.0 / 3.0 if str(mode).strip().lower() == "train" else 1.0 / 6.0

    density_factor = 1.0
    if (
        density is not None
        and str(mode).strip().lower() == "train"
        and str(task).strip().lower() in {"detect", "obb"}
    ):
        half_life_density = _float_env("AMADEUS_LABEL_DENSITY_HALF", 200.0)
        density_factor = 1.0 + max(0.0, density) / half_life_density

    per_sample_cost = area_factor * mode_factor * density_factor
    batch = max(1, int(total_gib / per_sample_cost))
    return batch, per_sample_cost, density_factor


def auto_batch_size(
    image_size: int,
    device,
    mode: str = "train",
    labels_dir: str | None = None,
    task: str = "detect",
) -> int:
    """Estimate batch size from a unified GPU-memory / image-size / label-density cost model.

    Batch size is the floor of total VRAM divided by an estimated per-sample
    memory cost. That cost is the product of an image-resolution term and,
    for training on detect/obb tasks, a label-density term derived from a
    high percentile of per-image instance counts (default p95) rather than
    the dataset mean, so that a minority of instance-dense images cannot
    cause VRAM overflow that a dataset-wide average would mask. See
    `_batch_size_from_memory` for the exact formula.

    Environment overrides:
    - AMADEUS_LABEL_DENSITY_PERCENTILE: percentile used as the density statistic
      (default 95; use 100 for the maximum observed value)
    - AMADEUS_LABEL_DENSITY_HALF: labels/image at which the batch size is halved
      relative to density=0 (default 200)
    - AMADEUS_MAX_AUTO_BATCH: optional absolute cap on the resulting batch size
    - AMADEUS_MAX_AUTO_CPU_BATCH: optional cap for CPU and MPS batch sizing
    """
    device_kind = _accelerator_type(device)
    total_gib = _cuda_total_gib(device)
    mps_recommended_gib = _mps_recommended_gib(device)
    mode_lower = str(mode).strip().lower()

    if total_gib is None:
        # CPU and MPS use the existing conservative system-RAM heuristic. MPS
        # contributes only an additional upper bound, never a replacement for
        # that heuristic or the CUDA VRAM cost model.
        try:
            import psutil

            available_gib = psutil.virtual_memory().available / (1024.0 ** 3)
            physical_cpu = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        except Exception:
            available_gib = None
            physical_cpu = os.cpu_count() or 1

        area_factor = (float(image_size) / 640.0) ** 2
        per_sample_gib = _float_env(
            "AMADEUS_CPU_BATCH_GIB_PER_SAMPLE",
            0.50 if mode_lower == "train" else 0.20,
        ) * area_factor
        density_stat = mean_labels = total_files = sampled_files = None
        density_factor = 1.0
        if device_kind == "mps" and mode_lower == "train" and str(task).strip().lower() in {"detect", "obb"}:
            percentile = _float_env("AMADEUS_LABEL_DENSITY_PERCENTILE", 95.0)
            estimated = _estimate_yolo_label_density(labels_dir, percentile=percentile)
            if estimated is not None:
                density_stat, mean_labels, total_files, sampled_files = estimated
                _unused_batch, _unused_cost, density_factor = _batch_size_from_memory(
                    1.0, image_size, mode, density_stat, task
                )
                per_sample_gib *= density_factor
        # MPS shares unified memory with the CPU. The local M1 Pro profile
        # measured batch=24 below the MPS working-set limit with system RAM
        # headroom, so let MPS use 60% of currently available RAM by default.
        # CPU-only behavior retains its existing 50% default.
        default_ram_fraction = 0.60 if device_kind == "mps" else 0.50
        ram_fraction = min(
            0.80,
            max(0.10, _float_env("AMADEUS_CPU_BATCH_RAM_FRACTION", default_ram_fraction)),
        )
        # Keep the measured MPS batch available while preserving the existing
        # CPU default. The memory estimate and MPS recommended limit still cap
        # the selected value.
        default_max_cpu_batch = 24 if device_kind == "mps" else 16
        max_cpu_batch = max(
            1, int(_float_env("AMADEUS_MAX_AUTO_CPU_BATCH", default_max_cpu_batch))
        )

        if available_gib is None:
            batch = min(8, max_cpu_batch)
            ram_batch = None
            memory_budget_gib = mps_recommended_gib if device_kind == "mps" else None
            if mps_recommended_gib is not None:
                batch = min(batch, max(1, int(mps_recommended_gib / max(per_sample_gib, 0.01))))
        else:
            ram_budget_gib = max(1.0, available_gib * ram_fraction)
            memory_budget_gib = ram_budget_gib
            if device_kind == "mps" and mps_recommended_gib is not None:
                memory_budget_gib = min(memory_budget_gib, mps_recommended_gib)
            ram_batch = max(1, int(memory_budget_gib / max(per_sample_gib, 0.01)))
            cpu_batch_multiplier = 3 if device_kind == "mps" else 2
            cpu_batch_cap = max(4, int(physical_cpu) * cpu_batch_multiplier)
            raw_batch = min(ram_batch, cpu_batch_cap, max_cpu_batch)
            # Stable, conventional batch sizes avoid awkward final batches and
            # make comparisons between runs easier. A 24-sample MPS batch was
            # profiled directly; select it only when the memory estimate can
            # support at least 24 samples.
            if device_kind == "mps" and raw_batch >= 24:
                batch = 24
            else:
                batch = 1 << max(0, int(raw_batch).bit_length() - 1)

        ram_text = "unknown" if available_gib is None else f"{available_gib:.1f}GiB"
        ram_batch_text = "unknown" if ram_batch is None else str(ram_batch)
        backend = "mps" if device_kind == "mps" else "cpu"
        recommended_text = (
            "unavailable" if mps_recommended_gib is None else f"{mps_recommended_gib:.1f}GiB"
        )
        budget_text = "unknown" if memory_budget_gib is None else f"{memory_budget_gib:.1f}GiB"
        density_text = "unavailable" if density_stat is None else f"p{percentile:.0f}={density_stat:.1f}"
        print(
            f"[INFO] auto {backend} batch: "
            f"available_ram={ram_text}, mps_recommended_memory={recommended_text}, "
            f"memory_budget={budget_text}, ram_fraction={ram_fraction:.2f}, "
            f"label_density={density_text}, density_factor={density_factor:.3f}, "
            f"estimated_gib_per_sample={per_sample_gib:.3f}, "
            f"ram_batch_cap={ram_batch_text}, physical_cpu={physical_cpu}, "
            f"max_cpu_batch={max_cpu_batch}, selected_batch={batch}"
        )
        return batch

    percentile = _float_env("AMADEUS_LABEL_DENSITY_PERCENTILE", 95.0)

    density_stat = mean_labels = total_files = sampled_files = None
    if mode_lower == "train":
        estimated = _estimate_yolo_label_density(labels_dir, percentile=percentile)
        if estimated is not None:
            density_stat, mean_labels, total_files, sampled_files = estimated

    batch, per_sample_cost, density_factor = _batch_size_from_memory(
        total_gib, image_size, mode, density_stat, task
    )

    max_auto_batch = os.environ.get("AMADEUS_MAX_AUTO_BATCH")
    capped = False
    if max_auto_batch is not None and str(max_auto_batch).strip() != "":
        new_batch = min(batch, max(1, int(max_auto_batch)))
        capped = new_batch != batch
        batch = new_batch

    suffix = " (capped)" if capped else ""
    if density_stat is None:
        print(
            "[INFO] auto batch cost: "
            f"total_vram={total_gib:.1f}GiB, per_sample_cost={per_sample_cost:.5f}, "
            f"density_factor={density_factor:.3f}, selected_batch={batch}{suffix}"
        )
    else:
        print(
            "[INFO] auto batch cost: "
            f"total_vram={total_gib:.1f}GiB, p{percentile:.0f}_labels_per_image={density_stat:.1f}, "
            f"mean_labels_per_image={mean_labels:.1f}, sampled={sampled_files}/{total_files}, "
            f"per_sample_cost={per_sample_cost:.5f}, density_factor={density_factor:.3f}, "
            f"selected_batch={batch}{suffix}"
        )

    return batch


def resolve_batch_size(
    value,
    image_size: int,
    device,
    mode: str = "train",
    labels_dir: str | None = None,
    task: str = "detect",
) -> int:
    """Return batch size as int; compute automatically when value is 'auto'/None/''."""
    if value is None or str(value).strip().lower() in {"", "auto"}:
        bs = auto_batch_size(
            image_size=image_size,
            device=device,
            mode=mode,
            labels_dir=labels_dir,
            task=task,
        )
        print(f"[INFO] auto batch_size={bs} (mode={mode}, image_size={image_size}, task={task})")
        return bs
    return max(1, int(value))


_OOM_MARKERS = (
    "out of memory",
    "not enough memory",
    "cannot allocate memory",
    "can't allocate memory",
    "defaultcpuallocator",
    "bad allocation",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)


def _is_oom_error(exc: BaseException) -> tuple[bool, str]:
    """Return (is_oom, backend_label) for an exception raised during training.

    backend_label is "CUDA", "MPS", or "CPU/RAM" and is only meaningful when
    is_oom is True. Shared by the calibration candidate search and by
    run_with_oom_retry so both agree on what counts as an OOM.
    """
    message = str(exc).lower()
    if not isinstance(exc, MemoryError) and not any(marker in message for marker in _OOM_MARKERS):
        return False, ""
    if any(x in message for x in ("cuda", "cublas", "cudnn")):
        backend = "CUDA"
    elif "mps" in message:
        backend = "MPS"
    else:
        backend = "CPU/RAM"
    return True, backend


def run_with_oom_retry(fn, initial_batch: int):
    """Call fn(batch_size) once; on CUDA/CPU-memory OOM, halve the batch and
    call fn again with the smaller batch.

    fn must accept a single positional int argument (batch size) and must
    build any model/trainer/optimizer state it needs from scratch on every
    call -- this helper does not reuse state across retries itself, so a
    retry only gets a clean state if fn constructs one. Raises RuntimeError
    if memory exhaustion persists at batch=1.
    """
    batch = max(1, int(initial_batch))
    while True:
        try:
            return fn(batch)
        except Exception as exc:
            is_oom, backend = _is_oom_error(exc)
            if not is_oom:
                raise
            empty_accelerator_cache()
            new_batch = batch // 2
            if new_batch < 1:
                raise RuntimeError(f"{backend} memory exhaustion even at batch=1; cannot proceed.") from exc
            print(f"[WARN] {backend} memory exhaustion (batch={batch}) - retrying with batch={new_batch}")
            batch = new_batch


# --- Real-training starting-epoch sanity check --------------------------------
#
# There is no more preflight calibration (a separate throwaway probe before
# real training). Batch size comes from the closed-form heuristic below
# (auto_batch_size) and dataloader workers default to a fixed count; real
# training then starts immediately. _InitialLoadCheck owns RAM pressure, WDDM
# observation, and VRAM headroom during the starting epoch; the separate
# _PerformanceMonitor watches measured training/validation throughput for the
# whole run. Once the starting epoch finishes cleanly, its own full
# training-only throughput becomes the official later-epoch baseline.

# Dataloader workers default to this fixed count rather than a RAM/CPU-aware
# heuristic. If that's wrong for a given machine, the starting-epoch check
# catches the resulting RAM pressure/DataLoader failure and steps workers
# down. The recovery resumes a valid checkpoint when ACCEPT_RESUME permits it
# and starts fresh only when resume is disabled or unavailable.
TRAIN_DEFAULT_WORKERS = int(_float_env("AMADEUS_TRAIN_DEFAULT_WORKERS", 8))

# Rolling-window duration _PerformanceMonitor measures real throughput over
# before comparing it against the starting epoch's baseline (see
# TRAIN_COLLAPSE_RATIO). Completed training and validation windows are measured
# by their own batch callbacks, so validation/checkpoint-save time can never
# contaminate training throughput. The same duration is the minimum evidence
# for an in-flight batch stall; the supervised CLI can terminate that CUDA
# child even when its callback never returns.
TRAIN_COLLAPSE_WINDOW_SECONDS = _float_env("AMADEUS_TRAIN_COLLAPSE_WINDOW_SECONDS", 60.0)

# System RAM fraction, observed at any point during the starting epoch,
# considered clear pressure -- retry with fewer dataloader workers.
TRAIN_INITIAL_RAM_TARGET = _float_env("AMADEUS_TRAIN_INITIAL_RAM_TARGET", 0.90)

# If measured training or validation throughput falls below this fraction of
# its training reference, that's a collapse candidate. obb_detector_obb_detector_training.py lowers
# batch only when it also confirms both WDDM non-local growth and VRAM pressure
# (including the pre-eviction pressure-equivalent working set); otherwise it
# warns and continues. Windows/WDDM-independent recovery remains limited to
# actual CUDA OOM and Ultralytics' silent CUDA-OOM CPU fallback.
TRAIN_COLLAPSE_RATIO = _float_env("AMADEUS_TRAIN_COLLAPSE_RATIO", 0.5)

# VRAM fraction (peak, over the whole starting epoch) below which VRAM is
# considered clearly underused -- batch may be raised once via
# next_batch_candidate(batch, "up") (~1.25x, not a closed-form estimate: VRAM
# use isn't reliably linear in batch size, so extrapolating a target
# utilization from one data point risks overshooting straight into the next
# OOM). Never re-evaluated after that single recovery, so this can't turn
# into an iterative max-batch search, and is permanently disabled for the
# rest of a run's recovery series the moment a batch has ever been lowered
# (see main()) -- otherwise a lower/raise pair could oscillate indefinitely
# around the same unsafe batch.
TRAIN_INITIAL_VRAM_HEADROOM_TRIGGER = _float_env("AMADEUS_TRAIN_INITIAL_VRAM_HEADROOM_TRIGGER", 0.40)


_DATALOADER_FAILURE_MARKERS = (
    "dataloader worker",
    "workers exited unexpectedly",
    "killed by signal",
    "pin memory thread exited unexpectedly",
)


def _is_dataloader_failure(exc: BaseException) -> bool:
    """Return True for a PyTorch DataLoader worker crash/failure -- a clear,
    unambiguous signal (like OOM) rather than a heuristic, so it's treated
    the same way regardless of when during training it happens."""
    message = str(exc).lower()
    return any(marker in message for marker in _DATALOADER_FAILURE_MARKERS)


def next_batch_candidate(batch: int, direction: str, max_batch: int | None = None) -> int:
    """Return the next batch size to try during automatic recovery.

    direction="up" escalates ~1.25-1.5x (at least +1); direction="down"
    (used after a hard limit: OOM or GPU->CPU fallback) halves -- a single
    substantial cut, not a 1-at-a-time search. Always clipped to >= 1 and, if
    given, to AMADEUS_MAX_AUTO_BATCH-style caps via max_batch. Training's
    lower-batch recovery uses next_batch_candidate_soft_down instead.
    """
    batch = max(1, int(batch))
    if direction == "up":
        new_batch = max(batch + 1, int(round(batch * 1.25)))
    elif direction == "down":
        new_batch = batch // 2
    else:
        raise ValueError(f"Unknown direction: {direction!r}")
    new_batch = max(1, new_batch)
    if max_batch is not None:
        new_batch = min(new_batch, max(1, int(max_batch)))
    return new_batch


def next_batch_candidate_soft_down(batch: int) -> int:
    """Return training recovery's conservative ~2/3 batch reduction.

    Used after OOM, GPU-to-CPU fallback, WDDM spill, or a throughput collapse
    confirmed by VRAM pressure. Strictly decreases for batch >= 2 and floors
    at 1.
    """
    batch = max(1, int(batch))
    return max(1, round(batch * 2 / 3))


def next_worker_candidate(workers: int) -> int:
    """Halve dataloader worker count on RAM pressure: 8->4->2->1->0."""
    workers = max(0, int(workers))
    if workers <= 0:
        return 0
    return workers // 2


def auto_num_workers(
    task: str = "process",
    image_size: int = 640,
    device=None,
) -> int:
    try:
        import psutil

        mem_gb = psutil.virtual_memory().available / (1024 ** 3)
        physical_cpu = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except Exception:
        mem_gb = None
        physical_cpu = os.cpu_count() or 1

    logical_cpu = os.cpu_count() or 1
    is_cpu_device = str(device).strip().lower() == "cpu"

    if task == "process":
        mem_cap = max(1, int(mem_gb // 2.5)) if mem_gb is not None else 4
        n = max(1, min(12, logical_cpu - 1, mem_cap))
    elif task == "dataloader":
        area_factor = (float(image_size) / 640.0) ** 2
        per_worker_gib = 2.5 * area_factor
        mem_cap = max(1, int(mem_gb // per_worker_gib)) if mem_gb is not None else 4
        if is_cpu_device:
            # The training process itself needs the physical cores for oneDNN
            # convolution. Reserve roughly half for model compute and use the
            # remainder for decoding/augmentation.
            worker_cap = max(1, int(physical_cpu) // 2)
            n = max(1, min(8, worker_cap, mem_cap))
        else:
            n = max(1, min(8, logical_cpu, mem_cap))
    else:
        mem_cap = max(1, int(mem_gb // 2.5)) if mem_gb is not None else 4
        n = max(1, min(12, logical_cpu, mem_cap))

    mem_text = "unknown" if mem_gb is None else f"{mem_gb:.1f}GiB"
    print(
        f"[INFO] auto workers detail: task={task}, device={device}, "
        f"logical_cpu={logical_cpu}, physical_cpu={physical_cpu}, "
        f"available_ram={mem_text}, mem_cap={mem_cap}, selected={n}"
    )
    return n


def resolve_num_workers(
    value,
    task: str = "process",
    image_size: int = 640,
    device=None,
) -> int:
    if value is None or str(value).strip().lower() in {"", "auto"}:
        n = auto_num_workers(task, image_size=image_size, device=device)
        print(
            f"[INFO] auto num_workers={n} "
            f"(task={task}, image_size={image_size}, device={device})"
        )
        return n
    return max(0, int(value))
