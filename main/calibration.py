# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Calibrate CUDA training batch size and data-loader workers.

Probe augmented samples from the instance-dense tail, verify proposed batch
sizes with forward/backward steps, and estimate workers from measured rates.
Reject OOM, shared-memory spill, and memory-budget violations. Sampled probes
do not bound every batch in an epoch; obb_detector_obb_detector_training.py handles later overloads.
Return None on CPU/MPS or calibration failure so callers use their existing
batch-size heuristic."""

from __future__ import annotations

import gc
import math
import os
# Every duration measured here -- one augmented sample, one training step -- is
# tens of milliseconds, and is then divided into another such duration to size
# the worker count. time.monotonic() cannot measure that on Windows: its
# resolution there is 15.625ms, so a 90ms sample carries up to 17% error and
# the ratio of two of them is noise. time.perf_counter() is sub-microsecond on
# every platform, which is why it is used throughout this module.
import time
from dataclasses import dataclass, field

from batch_utils import (
    TRAIN_COLLAPSE_RATIO,
    TRAIN_INITIAL_RAM_TARGET,
    TRAIN_WDDM_GROWTH_MIN_GIB,
    _is_oom_error,
    _resolve_cuda_index,
    empty_accelerator_cache,
    query_wddm_non_local_usage,
)

# Fraction of the training set, taken from its instance-dense end, that
# calibration probes are built from. 1% keeps the sample clearly in the tail
# that drives memory while staying wide enough that one mislabeled image cannot
# define the worst case on its own.
CALIBRATION_TOP_PERCENT = float(os.environ.get("AMADEUS_CALIBRATION_TOP_PERCENT", "1.0"))

# Fraction of measured free VRAM left unused by the extrapolated batch. This
# covers allocator fragmentation and the driver-side growth that the caching
# allocator's own counters do not see; it only sets where the search starts,
# since the chosen batch is then verified by a real probe.
CALIBRATION_VRAM_MARGIN = float(os.environ.get("AMADEUS_CALIBRATION_VRAM_MARGIN", "0.10"))

# Worst-case samples generated before probing; the densest of them becomes the
# sample every probe batch is built from. This is a sample of a distribution
# whose upper tail is the whole point, and an epoch draws thousands of times
# more mosaics than calibration can afford to: too few samples here and the
# probe is calibrated against a batch milder than the ones the epoch will
# actually produce, which shows up as a spill in the first epoch instead. The
# cost is linear and small (one augmented sample each), so this errs on the
# generous side.
CALIBRATION_POOL_SAMPLES = int(float(os.environ.get("AMADEUS_CALIBRATION_POOL_SAMPLES", "12")))

# Random images loaded before the pool, to give mosaic a realistic set of
# partner images to compose each worst-case sample from (see
# _WorstCaseSamplePool). Four is what one mosaic consumes.
CALIBRATION_PRIMING_SAMPLES = int(float(os.environ.get("AMADEUS_CALIBRATION_PRIMING_SAMPLES", "6")))

# Seed batch sizes for the memory fit, approached from below. Overshooting is
# what costs time here -- a batch that spills runs at PCIe speed and takes
# seconds to unwind -- so the first two points are the two cheapest ones that
# can define a line, and everything above them is extrapolated and then
# checked one step at a time.
CALIBRATION_PROBE_BATCHES = (1, 2)

# Upper bound on the verification retries (each one cuts the batch to 2/3).
CALIBRATION_MAX_VERIFY_ATTEMPTS = 4

# Training steps timed for the steady-state rate the worker count is derived
# from. The first is always discarded and the rest are taken as a median:
# individual steps scatter by tens of percent, and that scatter would
# otherwise land directly in the worker count.
CALIBRATION_STEADY_ITERATIONS = 6


@dataclass
class CalibrationResult:
    """Outcome of a successful calibration.

    batch/workers are what training should start with; the remaining fields are
    the measurements they were derived from, kept for the log line so a run's
    chosen values can always be traced back to what was observed.
    """

    batch: int
    workers: int
    worst_case_instances: float
    fixed_bytes: float
    per_sample_bytes: float
    usable_bytes: float
    gpu_seconds_per_image: float
    cpu_seconds_per_image: float
    workers_needed: float
    notes: list[str] = field(default_factory=list)


@dataclass
class _ProbeResult:
    ok: bool
    peak_bytes: float = 0.0
    step_seconds: float = 0.0
    spilled: bool = False
    reason: str = ""


def _log(message: str) -> None:
    print(f"[CAL] {message}", flush=True)


def calibrate_training(
    *,
    model_path: str,
    data_yaml: str,
    dataset_dir: str,
    image_size: int,
    device,
    train_overrides: dict | None = None,
    task: str = "obb",
    max_batch: int | None = None,
    worker_cap: int | None = None,
) -> CalibrationResult | None:
    """Measure a safe batch size and a matching worker count, or return None.

    None means "not calibrated" -- a non-CUDA device, or any failure at all --
    and the caller is expected to fall back to its closed-form heuristics.
    Returning None is always preferable to raising: nothing in here is worth
    failing a training run over.
    """
    started = time.perf_counter()
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        _log(f"skipped: torch unavailable ({exc})")
        return None

    is_cuda, cuda_index = _resolve_cuda_index(device)
    if not is_cuda or not torch.cuda.is_available():
        # The probe measures CUDA reserved memory and WDDM spill; neither
        # exists on CPU, and MPS reports memory through a different API whose
        # numbers are not comparable. Both keep the closed-form path.
        _log(f"skipped: device={device} is not CUDA; using the closed-form estimate instead")
        return None

    try:
        result = _calibrate_cuda(
            torch=torch,
            cuda_index=cuda_index,
            model_path=model_path,
            data_yaml=data_yaml,
            dataset_dir=dataset_dir,
            image_size=int(image_size),
            device=device,
            train_overrides=dict(train_overrides or {}),
            task=task,
            max_batch=max_batch,
            worker_cap=worker_cap,
        )
    except Exception as exc:
        _log(f"failed: {type(exc).__name__}: {exc}; using the closed-form estimate instead")
        return None
    finally:
        gc.collect()
        empty_accelerator_cache(device)

    if result is not None:
        _log(f"calibration took {time.perf_counter() - started:.1f}s")
    return result


def _calibrate_cuda(
    *,
    torch,
    cuda_index: int,
    model_path: str,
    data_yaml: str,
    dataset_dir: str,
    image_size: int,
    device,
    train_overrides: dict,
    task: str,
    max_batch: int | None,
    worker_cap: int | None,
) -> CalibrationResult | None:
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    notes: list[str] = []

    # The probe must augment exactly the way training will, or its timings and
    # its memory both describe a different job than the one being calibrated.
    overrides = {
        "task": task,
        "mode": "train",
        "data": data_yaml,
        "imgsz": int(image_size),
        "device": str(device),
    }
    for key in ("mosaic", "close_mosaic", "fliplr", "flipud", "scale", "degrees", "translate", "cache", "rect"):
        if key in train_overrides:
            overrides[key] = train_overrides[key]
    args = get_cfg(overrides=overrides)

    data = check_det_dataset(data_yaml)
    train_images = os.path.join(dataset_dir, "train", "images")
    dataset = build_yolo_dataset(
        args,
        train_images,
        CALIBRATION_PROBE_BATCHES[-1],
        data,
        mode="train",
        stride=32,
    )

    counts = [int(len(label.get("cls", []))) for label in dataset.labels]
    if not counts:
        _log("skipped: the training set has no labels to profile")
        return None
    dense_indices, worst_case_instances = _dense_indices(counts)
    _log(
        f"worst case: top {CALIBRATION_TOP_PERCENT:g}% of {len(counts)} training images "
        f"({len(dense_indices)} images), mean {worst_case_instances:.0f} instances/image "
        f"vs {sum(counts) / len(counts):.0f} dataset-wide"
    )

    model = YOLO(model_path)
    net = model.model
    net.args = args
    net.to(f"cuda:{cuda_index}")
    net.train()
    # Weights come off a checkpoint in inference state; the real trainer
    # re-enables gradients the same way before its first step, and without it
    # backward would allocate nothing and the probe would measure a forward
    # pass rather than a training step.
    for param in net.parameters():
        if param.dtype.is_floating_point:
            param.requires_grad_(True)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda")

    # Building the pool is also the augmentation-cost measurement.
    pool = _WorstCaseSamplePool(
        dataset,
        dense_indices,
        CALIBRATION_POOL_SAMPLES,
        CALIBRATION_PRIMING_SAMPLES,
    )
    _log(
        f"probe sample: {pool.worst_instances()} instances after mosaic "
        f"(pool of {CALIBRATION_POOL_SAMPLES}: {sorted(pool.instance_counts(), reverse=True)}), "
        f"augmentation {pool.seconds_per_sample() * 1000:.0f}ms/sample"
    )

    empty_accelerator_cache(device)
    torch.cuda.reset_peak_memory_stats(cuda_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_index)
    reserved_bytes = float(torch.cuda.memory_reserved(cuda_index))
    # What the run may reserve in total: everything currently free, plus what
    # this process has already reserved (its own pool is reusable), minus a
    # margin for fragmentation and driver-side allocations.
    usable_bytes = (float(free_bytes) + reserved_bytes) * (1.0 - CALIBRATION_VRAM_MARGIN)
    _log(
        f"VRAM: total={_gib(total_bytes)}, free={_gib(free_bytes)}, "
        f"already reserved by this process={_gib(reserved_bytes)}, "
        f"usable for training={_gib(usable_bytes)} "
        f"(margin {CALIBRATION_VRAM_MARGIN:.0%})"
    )

    # Rolling baseline: each probe is judged against the reading left by the
    # one before it. A fixed baseline would be wrong from the first spill
    # onward -- once the driver has moved allocations into system memory the
    # counter stays high, and every later probe would inherit that growth and
    # be rejected for a spill it never caused.
    wddm_state = {"baseline": query_wddm_non_local_usage()}

    def probe(
        batch: int, samples=None, check_spill: bool = True, iterations: int = 2
    ) -> _ProbeResult:
        result = _probe_batch(
            torch=torch,
            cuda_index=cuda_index,
            device=device,
            net=net,
            optimizer=optimizer,
            scaler=scaler,
            samples=pool.worst_batch(batch) if samples is None else samples,
            dataset=dataset,
            batch=batch,
            wddm_state=wddm_state if check_spill else None,
            iterations=iterations,
        )
        # Running without raising is not the same as fitting. A batch whose
        # peak reserve is above the usable budget is one the driver only got
        # through by consuming the headroom that fragmentation, validation and
        # every other process on this GPU also need -- the state that turns
        # into a spill (and the throughput cliff that follows) once real
        # training varies the batch composition.
        if result.ok and result.peak_bytes > usable_bytes:
            return _ProbeResult(
                ok=False,
                peak_bytes=result.peak_bytes,
                step_seconds=result.step_seconds,
                reason=(
                    f"peak reserve {_gib(result.peak_bytes)} exceeds the "
                    f"{_gib(usable_bytes)} usable budget"
                ),
            )
        return result

    # --- measure, fit, extrapolate, verify, repeat -----------------------------
    #
    # Every probe that survives becomes another point of the affine memory
    # model, and the model is refitted before proposing the next batch. That
    # matters because the first two points bracket a small range: the fit over
    # them alone tends to overstate the marginal cost per sample and stop well
    # short of the real capacity. Each accepted batch is a batch that has
    # actually run, so the loop can only ever return a size that was verified,
    # never one that was merely predicted.
    points: list[tuple[int, float]] = []
    accepted: list[tuple[int, float]] = []  # (batch, seconds per image)
    tried: set[int] = set()
    largest_ok = 0
    # No batch at or above a size that has already failed is ever proposed
    # again, however optimistic a refitted model becomes.
    ceiling = float(max_batch) if max_batch else math.inf
    fixed_bytes = per_sample_bytes = 0.0
    seeds = list(CALIBRATION_PROBE_BATCHES)
    candidate = seeds.pop(0)
    probes_left = len(CALIBRATION_PROBE_BATCHES) + CALIBRATION_MAX_VERIFY_ATTEMPTS

    while probes_left > 0:
        if math.isfinite(ceiling):
            candidate = int(min(candidate, ceiling))
        if candidate < 1 or candidate in tried or (largest_ok and candidate <= largest_ok):
            break

        probes_left -= 1
        tried.add(candidate)
        result = probe(candidate)

        if result.ok:
            points.append((candidate, result.peak_bytes))
            accepted.append((candidate, result.step_seconds / float(candidate)))
            largest_ok = max(largest_ok, candidate)
            _log(
                f"batch={candidate} fits: peak reserved={_gib(result.peak_bytes)} of "
                f"{_gib(usable_bytes)} usable, step={result.step_seconds * 1000:.0f}ms "
                f"({result.step_seconds / candidate * 1000:.0f}ms/img)"
            )
            fixed_bytes, per_sample_bytes = _fit_memory_model(points)
            if seeds:
                candidate = seeds.pop(0)
            else:
                # Never more than double what has already been shown to run:
                # the fit is an extrapolation, and a wildly oversized probe
                # risks a long driver stall rather than the clean rejection a
                # modest step gives.
                candidate = min(
                    _capacity(usable_bytes, fixed_bytes, per_sample_bytes, max_batch),
                    largest_ok * 2,
                )
                _log(
                    f"memory model: {_gib(fixed_bytes)} fixed + "
                    f"{_gib(per_sample_bytes)}/sample -> next candidate is batch={candidate}"
                )
        else:
            notes.append(f"batch={candidate} rejected: {result.reason}")
            _log(f"batch={candidate} rejected: {result.reason}")
            ceiling = candidate - 1
            seeds = []
            # The same 2/3 cut real training applies when it hits this wall.
            candidate = max(1, round(candidate * 2 / 3))

    if not accepted:
        _log("no batch size could be verified; falling back to batch=1")
        batch = 1
    else:
        # Take the largest batch that fit, unless its throughput collapsed
        # against the best rate measured -- the same criterion, and the same
        # ratio, that the runtime watchdog uses to call a slowdown a memory
        # problem rather than normal variation (TRAIN_COLLAPSE_RATIO). Probe
        # timings scatter by a good fraction between repeats, so anything
        # milder than a collapse would just be reading noise.
        best_rate = max(1.0 / seconds for _batch, seconds in accepted)
        healthy = [
            (candidate_batch, seconds)
            for candidate_batch, seconds in accepted
            if (1.0 / seconds) >= best_rate * TRAIN_COLLAPSE_RATIO
        ]
        # A ratio of 1 or more (only reachable by overriding
        # AMADEUS_TRAIN_COLLAPSE_RATIO) can exclude everything, including the
        # fastest probe itself. Fall back to what was measured rather than
        # failing calibration over a threshold setting.
        batch = max(healthy or accepted, key=lambda item: item[0])[0]
        if batch != largest_ok:
            notes.append(
                f"batch={largest_ok} fit, but its throughput collapsed against batch={batch}"
            )

    # --- workers from the measured producer/consumer rates --------------------
    #
    # One more step at the chosen batch, on ordinary samples this time: what
    # the worker count has to keep up with is the rate training actually runs
    # at, not the rate of the densest batch in the dataset.
    steady = probe(
        batch,
        samples=pool.typical_batch(batch),
        check_spill=False,
        iterations=CALIBRATION_STEADY_ITERATIONS,
    )
    gpu_seconds_per_image = (
        steady.step_seconds / float(batch) if steady.ok and steady.step_seconds > 0 else 0.0
    )
    cpu_seconds_per_image = pool.seconds_per_sample()
    _log(
        f"steady state at batch={batch}: gpu {gpu_seconds_per_image * 1000:.0f}ms/img, "
        f"augmentation {cpu_seconds_per_image * 1000:.0f}ms/img "
        f"(ordinary samples, {sorted(pool.typical_instance_counts(), reverse=True)} instances)"
    )
    workers_needed = (
        cpu_seconds_per_image / gpu_seconds_per_image if gpu_seconds_per_image > 0 else 1.0
    )
    workers = _resolve_workers(
        workers_needed=workers_needed,
        per_worker_bytes=pool.per_worker_bytes(),
        worker_cap=worker_cap,
        notes=notes,
    )

    return _finish(
        batch=batch,
        workers=workers,
        worst_case_instances=worst_case_instances,
        fixed_bytes=fixed_bytes,
        per_sample_bytes=per_sample_bytes,
        usable_bytes=usable_bytes,
        gpu_seconds_per_image=gpu_seconds_per_image,
        cpu_seconds_per_image=cpu_seconds_per_image,
        workers_needed=workers_needed,
        notes=notes,
    )


def _finish(**kwargs) -> CalibrationResult:
    """Report the result.

    The probe's weights, gradients and optimizer state are released by
    calibrate_training's finally block once this frame is gone -- real
    training builds its own model right after, and would otherwise start with
    the VRAM calibration just measured already partly spent.
    """
    result = CalibrationResult(**kwargs)
    _log(
        f"calibrated: batch={result.batch}, workers={result.workers} "
        f"(gpu {result.gpu_seconds_per_image * 1000:.0f}ms/img, "
        f"augmentation {result.cpu_seconds_per_image * 1000:.0f}ms/img, "
        f"workers needed {result.workers_needed:.1f})"
    )
    for note in result.notes:
        _log(f"note: {note}")
    return result


def _fit_memory_model(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least-squares affine fit of peak memory against batch size.

    Returns (fixed_bytes, per_sample_bytes). A single point is read as
    entirely per-sample cost, which overstates the marginal cost and so can
    only ever propose a smaller batch than the truth -- the safe direction to
    be wrong in. A negative intercept (memory growing slightly faster than
    linearly, which happens because the assigner is sized by the densest
    image in the batch) is clamped to zero for the same reason.
    """
    if not points:
        return 0.0, 0.0
    if len(points) == 1:
        batch, peak = points[0]
        return 0.0, peak / float(batch)

    n = float(len(points))
    mean_batch = sum(b for b, _ in points) / n
    mean_peak = sum(p for _, p in points) / n
    covariance = sum((b - mean_batch) * (p - mean_peak) for b, p in points)
    variance = sum((b - mean_batch) ** 2 for b, _ in points)
    if variance <= 0:
        batch, peak = points[-1]
        return 0.0, peak / float(batch)
    slope = covariance / variance
    if slope <= 0:
        batch, peak = points[-1]
        return 0.0, peak / float(batch)
    intercept = mean_peak - slope * mean_batch
    return max(0.0, intercept), slope


def _capacity(
    usable_bytes: float, fixed_bytes: float, per_sample_bytes: float, max_batch: int | None
) -> int:
    """Largest batch the fitted model says fits in the usable VRAM."""
    if per_sample_bytes <= 0:
        return 1
    batch = max(1, int((usable_bytes - fixed_bytes) // per_sample_bytes))
    if max_batch is not None:
        batch = min(batch, max(1, int(max_batch)))
    return batch


def _dense_indices(counts: list[int]) -> tuple[list[int], float]:
    """Return (indices of the densest images, their mean instance count)."""
    order = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    take = max(1, int(math.ceil(len(order) * CALIBRATION_TOP_PERCENT / 100.0)))
    dense = order[:take]
    mean_instances = sum(counts[i] for i in dense) / float(len(dense))
    return dense, mean_instances


class _WorstCaseSamplePool:
    """Augmented worst-case samples, generated once and reused by every probe.

    Indexing the dataset directly is exactly the work one dataloader worker
    does per sample, so the time spent here is also the measurement of a
    worker's throughput.

    What counts as the worst case needs care, because mosaic composes each
    sample from four images. The buffer Ultralytics draws those partners from
    is primed here with images taken at random from the whole training set,
    and reset to that state before every dense sample, so a probe sample is
    one instance-dense image mosaicked with ordinary ones -- the densest thing
    real training will actually hand the model. Letting the buffer fill with
    dense images instead produces samples several times denser than any batch
    a real run can draw, which is not a safety margin but a different dataset.

    Every probe batch is then built by repeating the single densest sample of
    the pool. That is not pessimism either: the assigner pads its tensors to
    the largest instance count in the batch, so a batch holding one dense
    image and a batch holding nothing but copies of it allocate exactly the
    same memory. Repeating one sample simply removes the composition noise
    that would otherwise make measured memory look superlinear in batch size
    and corrupt the fit.
    """

    def __init__(
        self,
        dataset,
        dense_indices: list[int],
        sample_count: int,
        priming_count: int,
        seed: int = 0,
    ) -> None:
        self._dataset = dataset
        self._dense = dense_indices
        self._samples: list[dict] = []
        self._durations: list[float] = []
        self._typical: list[dict] = []
        self._typical_durations: list[float] = []
        self._rss = _process_rss_bytes()
        self._worst: dict | None = None
        self._worst_instances = 0
        self._fill(max(1, sample_count), max(1, priming_count), seed)

    def _fill(self, count: int, priming_count: int, seed: int) -> None:
        import random

        rng = random.Random(seed)
        image_count = int(getattr(self._dataset, "ni", 0)) or len(self._dense)
        # Priming loads do double duty: they give mosaic a realistic set of
        # partner images to build the worst-case samples from, and being
        # ordinary images drawn at random they are themselves the measurement
        # of what an average sample costs to produce.
        for position in range(priming_count):
            started = time.perf_counter()
            sample = self._dataset[rng.randrange(image_count)]
            elapsed = time.perf_counter() - started
            self._typical.append(sample)
            # The very first load pays for lazily imported codecs and cold
            # file access, which no later sample repeats.
            if position > 0:
                self._typical_durations.append(elapsed)
        primed_buffer = list(getattr(self._dataset, "buffer", []))

        for position in range(count):
            index = self._dense[position % len(self._dense)]
            if primed_buffer:
                self._dataset.buffer = list(primed_buffer)
            started = time.perf_counter()
            sample = self._dataset[index]
            self._durations.append(time.perf_counter() - started)
            self._samples.append(sample)
            instances = int(len(sample.get("cls", [])))
            if self._worst is None or instances > self._worst_instances:
                self._worst, self._worst_instances = sample, instances
        self._rss = _process_rss_bytes()

    def worst_batch(self, count: int) -> list[dict]:
        """A batch of `count` copies of the densest sample seen."""
        return [self._worst] * max(1, count)

    def typical_batch(self, count: int) -> list[dict]:
        """A batch of `count` ordinary samples, cycled from the priming set."""
        count = max(1, count)
        return [self._typical[i % len(self._typical)] for i in range(count)]

    def worst_instances(self) -> int:
        return self._worst_instances

    def instance_counts(self) -> list[int]:
        return [int(len(sample.get("cls", []))) for sample in self._samples]

    def typical_instance_counts(self) -> list[int]:
        return [int(len(sample.get("cls", []))) for sample in self._typical]

    def seconds_per_sample(self) -> float:
        """Median augmentation time of an ordinary sample.

        Worker count is a question about the steady state -- can the
        dataloader keep up with the GPU over an epoch -- so it is answered
        with ordinary samples on both sides, not with the worst case that
        decides memory. Instance density inflates GPU time far more than
        augmentation time, so pricing both sides off the densest sample would
        systematically ask for too few workers.
        """
        return _median(self._typical_durations) or _median(self._durations)

    def per_worker_bytes(self) -> float:
        """Resident memory one dataloader worker is expected to hold.

        A spawned worker re-creates this process's dataset, transforms and
        per-sample buffers, so this process's own RSS while doing exactly that
        work is the closest available measurement of one worker's footprint.
        """
        return max(0.0, float(self._rss))


def _probe_batch(
    *,
    torch,
    cuda_index: int,
    device,
    net,
    optimizer,
    scaler,
    samples: list[dict],
    dataset,
    batch: int,
    wddm_state: dict | None,
    iterations: int = 2,
) -> _ProbeResult:
    """Run real training steps at `batch` and report peak memory and step time.

    Never one step: the first pays for cuDNN algorithm selection and the
    allocator's initial growth, so it is always discarded. Deciding whether a
    batch fits needs only the two steps this defaults to, since that answer
    comes from peak memory; a step *time* that something is going to be
    divided by wants more, because a single step scatters far too much to
    divide by.

    The reported time is the fastest of the timed steps, not their average.
    Every iteration here does identical work on identical data, so the spread
    between them is not the model's variability -- it is whatever else the
    machine was doing, and that can only ever make a step look slower. Another
    process briefly holding the GPU during calibration would otherwise be
    read as "this GPU is slow", which in turn asks for too few dataloader
    workers for the entire run.
    """
    collated = dataset.collate_fn(list(samples))
    try:
        empty_accelerator_cache(device)
        torch.cuda.reset_peak_memory_stats(cuda_index)
        images = collated["img"].to(f"cuda:{cuda_index}", non_blocking=True).float() / 255.0
        payload = dict(collated)
        payload["img"] = images

        timings: list[float] = []
        for iteration in range(max(2, int(iterations))):
            started = time.perf_counter()
            with torch.amp.autocast("cuda"):
                loss, _items = net.loss(payload)
            scaler.scale(loss.sum()).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=False)
            torch.cuda.synchronize(cuda_index)
            if iteration > 0:
                timings.append(time.perf_counter() - started)

        step_seconds = min(timings) if timings else 0.0
        peak_bytes = float(torch.cuda.max_memory_reserved(cuda_index))
    except Exception as exc:
        is_oom, backend = _is_oom_error(exc)
        empty_accelerator_cache(device)
        if is_oom:
            return _ProbeResult(ok=False, reason=f"{backend} out of memory")
        raise
    finally:
        payload = None
        collated = None

    if wddm_state is not None and wddm_state.get("baseline") is not None:
        non_local = query_wddm_non_local_usage()
        if non_local is not None:
            growth = non_local - wddm_state["baseline"]
            wddm_state["baseline"] = non_local
            if growth >= TRAIN_WDDM_GROWTH_MIN_GIB:
                return _ProbeResult(
                    ok=False,
                    peak_bytes=peak_bytes,
                    step_seconds=step_seconds,
                    spilled=True,
                    reason=(
                        f"VRAM spilled into shared system memory (WDDM non-local usage grew by "
                        f"{growth:.2f}GiB)"
                    ),
                )

    return _ProbeResult(ok=True, peak_bytes=peak_bytes, step_seconds=step_seconds)


def _resolve_workers(
    *,
    workers_needed: float,
    per_worker_bytes: float,
    worker_cap: int | None,
    notes: list[str],
) -> int:
    """Workers enough to keep the GPU fed, clipped by CPU count and RAM."""
    try:
        import psutil

        virtual = psutil.virtual_memory()
        logical_cpu = os.cpu_count() or 1
        # The same RAM ceiling the starting-epoch watchdog enforces, so
        # calibration never picks a worker count that watchdog would
        # immediately roll back.
        ram_budget = virtual.total * TRAIN_INITIAL_RAM_TARGET - (virtual.total - virtual.available)
    except Exception:
        virtual = None
        logical_cpu = os.cpu_count() or 1
        ram_budget = None

    # One worker past break-even, not merely enough to match it. The two ways
    # of being wrong here are not symmetric: an extra worker costs one core
    # and one dataset copy in RAM, both of which are checked below, while
    # being one short starves the GPU for the entire run. Break-even itself is
    # also known to be slightly optimistic, since per-worker throughput drops
    # a little as workers contend for cores and memory bandwidth.
    workers = max(1, int(math.floor(workers_needed)) + 1)

    # One core stays with the training process itself: it runs the Python
    # training loop, the CUDA launches and the collate step.
    cpu_limit = max(1, logical_cpu - 1)
    if workers > cpu_limit:
        notes.append(f"workers clipped to {cpu_limit} by CPU count ({logical_cpu} logical)")
        workers = cpu_limit

    if ram_budget is not None and per_worker_bytes > 0:
        ram_limit = max(1, int(ram_budget // per_worker_bytes))
        if workers > ram_limit:
            notes.append(
                f"workers clipped to {ram_limit} by RAM headroom "
                f"({_gib(ram_budget)} free below the {TRAIN_INITIAL_RAM_TARGET:.0%} line, "
                f"{_gib(per_worker_bytes)} per worker)"
            )
            workers = ram_limit

    if worker_cap is not None and workers > max(0, int(worker_cap)):
        notes.append(f"workers clipped to the configured maximum {int(worker_cap)}")
        workers = max(0, int(worker_cap))

    return workers


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _process_rss_bytes() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0.0


def _gib(value: float) -> str:
    return f"{value / (1024.0 ** 3):.2f}GiB"
