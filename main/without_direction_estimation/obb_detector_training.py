# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

import sys
from pathlib import Path
_MAIN_DIR = Path(__file__).resolve().parent.parent
for _path in (str(_MAIN_DIR), str(_MAIN_DIR.parent)):
    if _path not in sys.path:
        sys.path.append(_path)


import os
import re
import csv
import gc
import itertools
import json
import logging
import mmap
import shutil
import signal
import struct
import sys
import subprocess
import tempfile
import threading
import time
from typing import Callable, List

import yaml
from ultralytics import YOLO

from compute_backend import runtime_for
from compute_telemetry import (
    query_nvidia_smi as _query_nvidia_smi,
    query_wddm_non_local_usage as _query_wddm_non_local_usage,
    sample_accelerator,
)

from batch_utils import (
    resolve_batch_size,
    resolve_device,
    resolve_num_workers as _resolve_num_workers,
    auto_batch_size,
    next_batch_candidate,
    next_batch_candidate_soft_down,
    next_worker_candidate,
    _is_oom_error,
    _is_dataloader_failure,
    empty_accelerator_cache,
    get_accelerator_memory,
    _accelerator_type,
    effective_dataloader_workers,
    TRAIN_DEFAULT_WORKERS,
    TRAIN_COLLAPSE_WINDOW_SECONDS,
    TRAIN_INITIAL_RAM_TARGET,
    TRAIN_COLLAPSE_RATIO,
    TRAIN_INITIAL_VRAM_HEADROOM_TRIGGER,
)
from without_direction_estimation.create_dataset import resolve_yolo_dataset_dir
from path_utils import resolve_config_paths
from experiment_utils import (
    DEFAULT_LR0,
    DEFAULT_LRF,
    experiment_dir_name_from_cfg,
    format_lr0_for_name,
    format_lrf_for_name,
    require_single_lr0,
    require_single_lrf,
    resolve_existing_experiment_dir_name,
)
from random_utils import normalize_seed
from without_direction_estimation.training_paths import training_project_dir

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DIRECTION_CLASS_NAMES = ['animal']

# Minimum growth (GiB) in this process's own WDDM "Non Local Usage" above its
# recorded training-start baseline before it's treated as a genuine spill
# rather than noise (see _query_wddm_non_local_usage) -- some non-local usage
# can be normal driver/runtime overhead even without genuine VRAM
# oversubscription, so only a clear rise above this process's own baseline
# counts.
#
# This growth is a necessary but not sufficient recovery condition: after the
# starting epoch, _PerformanceMonitor also requires both a measured throughput
# collapse and VRAM pressure before lowering batch. Pinned host memory used by
# the DataLoader for faster host-to-device copies raises this counter during
# perfectly healthy training, which growth alone cannot distinguish.
TRAIN_WDDM_GROWTH_MIN_GIB = 0.25

# Shared VRAM-usage threshold for both the periodic bottleneck label and
# throughput-collapse recovery. A throughput drop alone is not evidence that
# batch size is the cause; it becomes actionable only when a fresh sample also
# shows that at least this fraction of device VRAM is in use.
TRAIN_VRAM_PRESSURE_RATIO = 0.95

# Cadences for _DeviceSampler, the single background thread that owns every
# subprocess-backed measurement. Nothing on the training path ever spawns a
# process: callbacks read the sampler's cached values only, so sampling cost
# can never stall the training loop or distort the throughput it measures.
TRAIN_SAMPLER_INTERVAL_SECONDS = 5.0

# WDDM "Non Local Usage" is sampled on the same five-second cadence as VRAM.
# Each sample is expensive (a PowerShell process enumerates GPU-process
# counters), but it stays entirely in the sampler thread and this cadence is
# needed to stop an in-flight sysmem-backed CUDA batch before it can spend
# minutes consuming tens of GiB of shared GPU memory.
TRAIN_SAMPLER_WDDM_INTERVAL_SECONDS = 5.0

# Internal-only process supervision protocol. The CLI entry point keeps a
# lightweight parent alive while the actual CUDA training attempt runs in a
# child process. A watchdog can therefore terminate a single attempt even
# while its training thread is stuck inside a very slow sysmem-backed CUDA
# kernel, then let the parent restart from the persisted lower-batch state.
# These are deliberately not user-facing settings.
_TRAIN_WORKER_ENV = "_AMADEUS_TRAIN_WORKER"
_EMERGENCY_BATCH_ENV = "_AMADEUS_EMERGENCY_BATCH"
_EMERGENCY_WORKERS_ENV = "_AMADEUS_EMERGENCY_WORKERS"
_WATCHDOG_STATE_ENV = "_AMADEUS_TRAIN_WATCHDOG_STATE"
_EMERGENCY_RECOVERY_EXIT_CODE = 86
_EMERGENCY_FATAL_EXIT_CODE = 87
_EMERGENCY_RECOVERY_FILENAME = ".automatic_batch_recovery.json"
_WATCHDOG_STATE_STRUCT = struct.Struct("<Qddiiiiii")


def describe_child_return_code(return_code: int) -> str:
    """Return a stable diagnostic for a child exit or POSIX signal."""
    if return_code < 0:
        number = -int(return_code)
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = {
                4: "SIGILL",
                6: "SIGABRT",
                7: "SIGBUS",
                8: "SIGFPE",
                11: "SIGSEGV",
                13: "SIGPIPE",
                15: "SIGTERM",
            }.get(number, "unknown signal")
        return f"child terminated by signal {number} ({name})"
    return f"child exited with code {int(return_code)}"


def get_dataset_yaml_path(dataset_dir: str) -> str:
    data_yaml = os.path.join(dataset_dir, "data.yaml")
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"Missing dataset data.yaml: {data_yaml}")
    return data_yaml


def verify_data_yaml(data_yaml: str, expected_nc: int, expected_names: List[str]) -> None:
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    nc = int(data.get("nc", -1))
    names = data.get("names", [])
    if nc != int(expected_nc):
        raise ValueError(f"Invalid nc in {data_yaml}: got {nc}, expected {expected_nc}")
    if list(names) != list(expected_names):
        raise ValueError(f"Invalid names in {data_yaml}: got {names}, expected {expected_names}")


def verify_dataset_dirs(dataset_dir: str) -> None:
    required_dirs = [
        os.path.join(dataset_dir, "train", "images"),
        os.path.join(dataset_dir, "train", "labels"),
        os.path.join(dataset_dir, "test", "images"),
        os.path.join(dataset_dir, "test", "labels"),
    ]
    missing = [x for x in required_dirs if not os.path.isdir(x)]
    if missing:
        raise FileNotFoundError("Missing required dataset directories:\n" + "\n".join(missing))

def append_pt_if_missing(model_name: str) -> str:
    return model_name if model_name.endswith(".pt") else model_name + ".pt"



def resolve_model_name(model_name: str) -> str:
    model_name = append_pt_if_missing(model_name)
    stem, ext = os.path.splitext(model_name)
    return model_name if stem.endswith("-obb") else f"{stem}-obb{ext}"


def resolve_class_names(cfg: dict) -> List[str]:
    training = cfg.get("training", {}) or {}
    use_direction = bool(training.get("USE_DIRECTION_CLASSES", True))
    if not use_direction:
        return ["object"]
    names = training.get("CLASS_NAMES")
    if isinstance(names, list) and len(names) == 8:
        return [str(x) for x in names]
    return list(DIRECTION_CLASS_NAMES)


def _torch_load_checkpoint(path: str):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_epoch_and_optimizer(path: str) -> tuple[int | None, bool]:
    try:
        ckpt = _torch_load_checkpoint(path)
    except Exception as exc:
        print(f"[WARN] Could not inspect checkpoint: {path} ({exc})")
        return None, False

    if not isinstance(ckpt, dict):
        return None, False

    try:
        epoch = ckpt.get("epoch")
        epoch = int(epoch) if epoch is not None else None
    except Exception:
        epoch = None
    return epoch, ckpt.get("optimizer") is not None


def _epoch_from_filename(path: str) -> int | None:
    """Convert a 1-based AMADEUS ``epochN.pt`` name to YOLO's epoch index."""
    match = re.fullmatch(r"epoch([1-9]\d*)\.pt", os.path.basename(path))
    if not match:
        return None
    return int(match.group(1)) - 1


def _periodic_checkpoint_filename(yolo_epoch: int, save_period: int) -> str | None:
    """Return the 1-based AMADEUS filename due for a YOLO epoch index."""
    yolo_epoch = int(yolo_epoch)
    save_period = int(save_period)
    if yolo_epoch < 0 or save_period < 1:
        return None
    if (yolo_epoch + 1) % save_period != 0:
        return None
    return f"epoch{yolo_epoch + 1}.pt"


def _linear_lr_factor(epoch: int, epochs: int, lrf: float) -> float:
    return max(1 - int(epoch) / int(epochs), 0) * (1.0 - float(lrf)) + float(lrf)


def _format_resume_float(value: float) -> str:
    return f"{float(value):.10g}"


def _reconcile_results_csv_for_resume(results_csv: str, checkpoint_path: str, checkpoint_epoch: int) -> None:
    """Make results.csv consistent with the checkpoint chosen for resume,
    treating that checkpoint's completed epoch as ground truth.

    Ultralytics writes an epoch's results.csv row (save_metrics) before it
    writes that epoch's checkpoint (save_model) -- see BaseTrainer._do_train
    -- so an interruption between those two steps is a normal way to end up
    with results.csv one epoch ahead of the last checkpoint actually saved.
    That is not corruption: any such trailing rows (epochs beyond the
    checkpoint's own completed epoch) are simply discarded here so training
    can resume cleanly from the checkpoint without duplicate epoch rows.

    Only the rows up to and including the checkpoint's completed epoch have
    to be a clean, gapless, correctly-ordered 1..completed_epoch sequence
    (or the 0-based equivalent some Ultralytics releases wrote) -- if that
    prefix itself has a gap, duplicate, or ordering problem, that is real
    corruption and raises.
    """
    if not os.path.isfile(results_csv):
        raise FileNotFoundError(f"results.csv is required when resuming: {results_csv}")
    completed_epoch = int(checkpoint_epoch) + 1

    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"results.csv is empty: {results_csv}")
    header, data_rows = rows[0], rows[1:]
    epoch_col = next((i for i, name in enumerate(header) if name.strip() == "epoch"), None)
    if epoch_col is None:
        raise ValueError(f"results.csv does not contain an epoch column: {results_csv}")

    parsed: list[tuple[int, list[str]]] = []
    for row in data_rows:
        raw = (row[epoch_col] if epoch_col < len(row) else "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid epoch value in results.csv: {raw!r}") from exc
        if not value.is_integer():
            raise ValueError(f"Non-integer epoch value in results.csv: {raw!r}")
        parsed.append((int(value), row))
    if not parsed:
        raise ValueError(f"results.csv does not contain any epoch rows: {results_csv}")

    # Ultralytics releases have written either 0-based or 1-based epoch
    # columns; the convention is inferred from the first row.
    first_epoch = parsed[0][0]
    if first_epoch == 0:
        expected_prefix = list(range(0, int(checkpoint_epoch) + 1))
    elif first_epoch == 1:
        expected_prefix = list(range(1, completed_epoch + 1))
    else:
        expected_prefix = []

    prefix = parsed[: len(expected_prefix)]
    if [epoch for epoch, _ in prefix] != expected_prefix:
        raise RuntimeError(
            "results.csv epoch sequence up to the resume checkpoint is inconsistent. "
            f"results_csv={results_csv}, "
            f"checkpoint_file={os.path.basename(checkpoint_path)}, "
            f"checkpoint_internal_epoch={checkpoint_epoch}, "
            f"expected_completed_epochs=1..{completed_epoch}, "
            f"found_epochs={[epoch for epoch, _ in parsed]}"
        )

    if len(parsed) == len(prefix):
        return

    print(
        f"[WARN] results.csv has {len(parsed) - len(prefix)} row(s) beyond the resume "
        f"checkpoint's completed epoch {completed_epoch} (a normal mid-epoch interruption can "
        f"leave results.csv one epoch ahead of the last checkpoint saved); discarding them: {results_csv}"
    )
    with open(results_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(row for _, row in prefix)


def _find_resume_checkpoint(weights_dir: str, target_epochs: int) -> tuple[str | None, int | None, bool]:
    """Return (checkpoint_path, epoch_index, already_complete).

    last.pt is Ultralytics' own checkpoint, rewritten every epoch (see
    BaseTrainer.save_model); AMADEUS' 1-based epochN.pt files are just a
    plain-copy snapshot of last.pt taken only on epochs where SAVE_PERIOD
    divides evenly. That makes last.pt at least as advanced as any epochN.pt
    under normal operation, so it is preferred whenever it loads cleanly and
    carries what resuming needs (a valid epoch, and optimizer state for an
    actual resume). Only when last.pt is missing, unreadable, or missing
    that state does this fall back to the newest usable epochN.pt, exactly
    as before AMADEUS considered last.pt at all.
    """
    target_epoch_index = int(target_epochs) - 1
    if target_epoch_index < 0 or not os.path.isdir(weights_dir):
        return None, None, False

    print(f"[INFO] Looking for resumable checkpoints in: {weights_dir}")

    valid_epoch_paths: list[tuple[int, str, bool]] = []
    saw_epoch_checkpoint = False
    for filename in os.listdir(weights_dir):
        path = os.path.join(weights_dir, filename)
        if not os.path.isfile(path):
            continue
        filename_epoch = _epoch_from_filename(path)
        if filename_epoch is None:
            continue
        saw_epoch_checkpoint = True
        ckpt_epoch, has_optimizer = _checkpoint_epoch_and_optimizer(path)
        if ckpt_epoch != filename_epoch:
            print(
                "[WARN] 1-based epochN.pt filename and checkpoint epoch do not match: "
                f"path={path}, filename_yolo_epoch={filename_epoch}, "
                f"checkpoint_internal_epoch={ckpt_epoch}"
            )
            continue
        valid_epoch_paths.append((filename_epoch, os.path.abspath(path), has_optimizer))

    last_path = os.path.abspath(os.path.join(weights_dir, "last.pt"))
    has_last = os.path.isfile(last_path)
    last_epoch: int | None = None
    last_has_optimizer = False
    if has_last:
        last_epoch, last_has_optimizer = _checkpoint_epoch_and_optimizer(last_path)
        if last_epoch is None or last_epoch < 0:
            print(f"[WARN] last.pt is not usable as a resume source (unreadable or missing epoch): {last_path}")
            last_epoch = None

    has_prior_checkpoint = saw_epoch_checkpoint or has_last

    # Already complete: a valid last.pt is checked first since it reflects
    # the single most recently completed epoch under normal operation.
    if last_epoch is not None and last_epoch >= target_epoch_index:
        return last_path, last_epoch, True
    if valid_epoch_paths:
        latest_epoch, latest_epoch_path, _ = max(valid_epoch_paths, key=lambda item: item[0])
        if latest_epoch >= target_epoch_index:
            return latest_epoch_path, latest_epoch, True

    # Not complete -- pick a resume source, preferring last.pt.
    if last_epoch is not None:
        if last_has_optimizer:
            print(f"[INFO] Selected resume checkpoint: {last_path}")
            return last_path, last_epoch, False
        print(f"[WARN] last.pt is not resumable (missing optimizer): {last_path}")

    for epoch, path, has_optimizer in sorted(valid_epoch_paths, key=lambda item: item[0], reverse=True):
        if has_optimizer:
            print(f"[INFO] Selected resume checkpoint: {path}")
            return path, epoch, False
        print(f"[WARN] Checkpoint is not resumable (missing optimizer): {path}")

    if has_prior_checkpoint:
        raise RuntimeError(
            "No resumable checkpoint was found. A resumable checkpoint must be either "
            "last.pt or AMADEUS' 1-based epochN.pt (with checkpoint metadata epoch N-1), "
            "and must include optimizer state. "
            f"weights_dir={weights_dir}"
        )

    return None, None, False


def _get_epochs_obb_trainer():
    from ultralytics.models.yolo.obb import OBBTrainer

    class EpochsOBBTrainer(OBBTrainer):
        def check_resume(self, overrides):
            overrides = overrides or {}
            requested_epochs = overrides.get("epochs")
            requested_lr0 = overrides.get("lr0")
            requested_lrf = overrides.get("lrf")
            requested_save_period = overrides.get("save_period")

            # Ultralytics restores project/name/data/workers from train_args
            # embedded in the checkpoint. Old paths are unsafe when epochN.pt
            # has moved to another AMADEUS session, and a recovery-requested
            # workers value must replace the checkpoint value. Restore both
            # after upstream check_resume(), before BaseTrainer creates its
            # output directory and DataLoaders.
            requested_run_args = {
                key: overrides.get(key)
                for key in ("project", "name", "data", "exist_ok", "workers")
                if key in overrides
            }

            super().check_resume(overrides)
            if self.resume:
                for key, value in requested_run_args.items():
                    setattr(self.args, key, value)
                if requested_epochs is not None:
                    self.args.epochs = int(requested_epochs)
                    if hasattr(self, "epochs"):
                        self.epochs = int(requested_epochs)
                if requested_lr0 is not None:
                    self.args.lr0 = float(requested_lr0)
                if requested_lrf is not None:
                    self.args.lrf = float(requested_lrf)
                if requested_save_period is not None:
                    self.args.save_period = int(requested_save_period)
                    if hasattr(self, "save_period"):
                        self.save_period = int(requested_save_period)

        def resume_training(self, ckpt):
            super().resume_training(ckpt)
            if ckpt is not None and self.resume:
                ckpt_epoch = int(ckpt.get("epoch", -1))
                expected_start_epoch = ckpt_epoch + 1
                if self.start_epoch != expected_start_epoch:
                    raise RuntimeError(
                        "Resume start epoch does not match checkpoint epoch: "
                        f"start_epoch={self.start_epoch}, expected_start_epoch={expected_start_epoch}, "
                        f"checkpoint_internal_epoch={ckpt_epoch}"
                    )
                completed_epoch = ckpt_epoch + 1
                first_resumed_epoch = self.start_epoch + 1
                expected_lr_factor = _linear_lr_factor(self.start_epoch, self.epochs, self.args.lrf)
                expected_lr = float(self.args.lr0) * expected_lr_factor
                print(
                    f"[INFO] Resume state loaded: checkpoint_epoch={ckpt_epoch}, "
                    f"start_epoch={self.start_epoch}, target_epochs={self.epochs}"
                )
                print(f"[INFO] Resume checkpoint file: {os.path.basename(str(self.args.model))}")
                print(f"[INFO] Checkpoint internal epoch: {ckpt_epoch}")
                print(f"[INFO] Completed epochs: {completed_epoch}")
                print(f"[INFO] First resumed epoch: {first_resumed_epoch}")
                print(f"[INFO] Target epochs: {self.epochs}")
                print(f"[INFO] Resume output directory: {self.save_dir}")
                print(f"[INFO] Resume save period: {self.save_period}")
                print(f"[INFO] Resume lr0: {_format_resume_float(self.args.lr0)}")
                print(f"[INFO] Resume lrf: {_format_resume_float(self.args.lrf)}")
                print(f"[INFO] Expected first resumed LR factor: {_format_resume_float(expected_lr_factor)}")
                print(f"[INFO] Expected first resumed LR: {_format_resume_float(expected_lr)}")

        def save_model(self):
            # Ultralytics numbers the stored checkpoint epoch from zero and
            # evaluates save_period against that value. AMADEUS exposes both
            # the interval and filename from one, so completed epoch 5 is
            # saved as epoch5.pt while its metadata remains YOLO epoch 4.
            save_period = self.save_period
            self.save_period = -1
            try:
                super().save_model()
            finally:
                self.save_period = save_period

            checkpoint_name = _periodic_checkpoint_filename(self.epoch, save_period)
            if checkpoint_name is not None:
                shutil.copyfile(self.last, self.wdir / checkpoint_name)

    return EpochsOBBTrainer

def resolve_save_period(value):
    if value is None or value == "":
        return 5
    if isinstance(value, str) and value.strip().lower() == "best":
        return -1
    if isinstance(value, bool):
        raise ValueError("SAVE_PERIOD must be a positive integer, -1, or 'best'.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError("SAVE_PERIOD must be a positive integer, -1, or 'best'.")
        parsed = int(value)
    else:
        text = str(value).strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            raise ValueError("SAVE_PERIOD must be a positive integer, -1, or 'best'.")
        parsed = int(text)
    value = parsed
    if value != -1 and value < 1:
        raise ValueError("SAVE_PERIOD must be a positive integer, -1, or 'best'.")
    return value


def resolve_epoch_count(value, *, key: str = "EPOCHS") -> int:
    """Parse a positive epoch count without truncating decimal values."""
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer greater than or equal to 1.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{key} must be an integer greater than or equal to 1.")
        parsed = int(value)
    else:
        text = str(value).strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            raise ValueError(f"{key} must be an integer greater than or equal to 1.")
        parsed = int(text)
    if parsed < 1:
        raise ValueError(f"{key} must be an integer greater than or equal to 1.")
    return parsed


def _parse_cuda_device_index(device) -> int | None:
    s = str(device).strip().lower()
    if s == "cpu":
        return None
    if s.isdigit():
        return int(s)
    if s == "cuda":
        return 0
    if s.startswith("cuda:"):
        tail = s[5:]
        return int(tail) if tail.isdigit() else 0
    return None


def _query_torch_vram(device) -> tuple[float | None, float | None]:
    runtime = runtime_for(device)
    if runtime.capabilities.memory_model != "dedicated":
        return None, None
    allocated = runtime.allocated_memory()
    reserved, _ = runtime.memory()
    gib = 1024.0 ** 3
    return (None if allocated is None else allocated / gib,
            None if reserved is None else reserved / gib)



def _query_cpu_memory() -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Return (cpu_percent, ram_used_gib, ram_total_gib, proc_rss_gib, child_rss_gib)."""
    try:
        import psutil

        cpu_percent = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        ram_used_gib = (vm.total - vm.available) / (1024.0 ** 3)
        ram_total_gib = vm.total / (1024.0 ** 3)

        proc = psutil.Process(os.getpid())
        proc_rss = proc.memory_info().rss
        child_rss = 0
        for child in proc.children(recursive=True):
            try:
                child_rss += child.memory_info().rss
            except Exception:
                pass
        return cpu_percent, ram_used_gib, ram_total_gib, proc_rss / (1024.0 ** 3), child_rss / (1024.0 ** 3)
    except Exception:
        return None, None, None, None, None


def _query_ram_fraction() -> float | None:
    """Return system RAM usage as a fraction of total, or None if unknown.

    This is the only memory figure the per-batch starting-epoch check needs
    (see _InitialLoadCheck.record_batch), and it deliberately does not reuse
    _query_cpu_memory: that function also walks proc.children(recursive=True)
    and reads each child's memory_info, which on Windows means enumerating
    every process on the system once per call -- unacceptable at per-batch
    frequency, and its per-process results were discarded by that caller
    anyway. One psutil.virtual_memory() read is cheap enough to run on every
    single batch.
    """
    try:
        import psutil

        vm = psutil.virtual_memory()
        if not vm.total:
            return None
        return (vm.total - vm.available) / vm.total
    except Exception:
        return None


def _fmt_gib(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}GiB"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}%"


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}img/s"


def _vram_pressure_fraction(
    *,
    vram_used: float | None,
    vram_total: float | None,
    current_frac: float | None,
    peak_frac: float | None,
    wddm_growth: float | None,
) -> float | None:
    """Return the strongest evidence of pre-eviction VRAM pressure.

    WDDM may evict dedicated pages into non-local memory just before local
    VRAM reaches 100%, so the post-spill local fraction can fall below the
    pressure threshold. Adding confirmed non-local growth back to current
    local use reconstructs a conservative pressure-equivalent working set.
    Throughput collapse and the WDDM-growth threshold remain separate required
    conditions; this helper does not make WDDM growth actionable on its own.
    """
    candidates = [value for value in (current_frac, peak_frac) if value is not None]
    if (
        vram_used is not None
        and vram_total
        and wddm_growth is not None
        and wddm_growth > 0
    ):
        candidates.append((vram_used + wddm_growth) / vram_total)
    return max(candidates) if candidates else None


def _configure_cpu_runtime(device, dataloader_workers: int) -> None:
    """Configure PyTorch CPU parallelism before training starts."""
    if str(device).strip().lower() != "cpu":
        return

    import torch

    try:
        import psutil

        physical_cpu = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except Exception:
        physical_cpu = os.cpu_count() or 1
    logical_cpu = os.cpu_count() or 1

    requested = os.environ.get("AMADEUS_CPU_THREADS")
    intra_threads = int(requested) if requested not in (None, "") else int(physical_cpu)
    intra_threads = max(1, min(intra_threads, int(logical_cpu)))

    requested_interop = os.environ.get("AMADEUS_CPU_INTEROP_THREADS")
    interop_threads = (
        int(requested_interop)
        if requested_interop not in (None, "")
        else min(4, max(1, intra_threads // 4))
    )
    interop_threads = max(1, interop_threads)

    torch.set_num_threads(intra_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError as exc:
        print(f"[WARN] Could not set PyTorch inter-op threads: {exc}")

    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True

    print(
        "[INFO] CPU runtime configured: "
        f"logical_cpu={logical_cpu}, physical_cpu={physical_cpu}, "
        f"torch_intra_threads={torch.get_num_threads()}, "
        f"torch_interop_threads={torch.get_num_interop_threads()}, "
        f"dataloader_workers={int(dataloader_workers)}, "
        f"mkldnn={getattr(torch.backends.mkldnn, 'enabled', 'n/a')}"
    )


def _enable_cudnn_autotuner(device) -> None:
    """Opt-in: enable cuDNN's algorithm autotuner (AMADEUS_CUDNN_BENCHMARK=1).

    Ultralytics leaves torch.backends.cudnn.benchmark at PyTorch's default of
    False, so cuDNN picks convolution algorithms without measuring them. The
    autotuner pays off only when input shapes stay constant, which is exactly
    this configuration: fixed imgsz, fixed batch, rect=False, multi_scale=False
    -- every training step sees the same tensor shapes, so one round of
    benchmarking at the start is amortised over the whole run.

    It is off by default because it costs strict reproducibility. Ultralytics
    is called here with deterministic=True, which sets cudnn.deterministic and
    so restricts the autotuner to deterministic algorithms; results within one
    process therefore stay repeatable. But which of those algorithms wins is
    decided by wall-clock timing, which varies with machine load, and
    different algorithms accumulate in different orders -- so bitwise-identical
    results across separate runs are no longer guaranteed. Use it while
    exploring; leave it off for runs that must be exactly reproducible.

    Called from on_train_start, which runs after Ultralytics' _setup_train
    (and the init_seeds call inside it), so this setting is not overwritten.
    """
    if str(os.environ.get("AMADEUS_CUDNN_BENCHMARK", "")).strip().lower() not in {"1", "true", "yes"}:
        return
    if _parse_cuda_device_index(device) is None:
        return

    import torch

    torch.backends.cudnn.benchmark = True
    print(
        "[INFO] cuDNN autotuner enabled (AMADEUS_CUDNN_BENCHMARK): input shapes are fixed, "
        "but run-to-run bitwise reproducibility is no longer guaranteed.",
        flush=True,
    )


def _apply_amp_override() -> None:
    """Opt-in: force AMP on regardless of Ultralytics' check (AMADEUS_FORCE_AMP=1).

    Ultralytics' check_amp matches the GPU name against a hardcoded list of
    devices known to produce NaN losses or zero mAP under AMP -- the GTX 16
    series (1630/1650/1660), several Quadro T parts, and Tesla K40M -- and
    returns False without running any measurement. On those GPUs the
    "checks failed" warning is a blocklist hit, not a detected anomaly, and no
    amount of environment tuning will turn AMP on.

    That matters because those GPUs do have fast FP16: Turing TU116/TU117 run
    half precision at twice the FP32 rate (no Tensor Cores, but packed math),
    and halved activation memory also allows a larger batch. Forcing AMP is
    therefore worth measuring -- but the blocklist exists for a reason, so
    treat any forced run as unverified until a short trial shows losses
    staying finite and mAP above zero.

    The patch targets ultralytics.engine.trainer.check_amp rather than
    ultralytics.utils.checks.check_amp: the trainer imports the name into its
    own module namespace, so rebinding it at the source module has no effect.
    """
    if str(os.environ.get("AMADEUS_FORCE_AMP", "")).strip().lower() not in {"1", "true", "yes"}:
        return

    from ultralytics.engine import trainer as ultralytics_trainer

    ultralytics_trainer.check_amp = lambda model: True
    print(
        "[WARN] AMP forced on (AMADEUS_FORCE_AMP), bypassing Ultralytics' GPU blocklist. "
        "Confirm on a short run that box/cls/dfl losses stay finite and mAP is above zero "
        "before trusting these weights.",
        flush=True,
    )


def _infer_bottleneck(
    gpu_util: float | None,
    vram_used: float | None,
    vram_total: float | None,
    cpu_percent: float | None,
    ram_used: float | None,
    ram_total: float | None,
    child_rss: float | None,
    workers: int,
) -> str:
    vram_frac = (vram_used / vram_total) if vram_used is not None and vram_total else None
    ram_frac = (ram_used / ram_total) if ram_used is not None and ram_total else None

    if ram_frac is not None and ram_frac >= 0.90:
        return "RAM pressure"

    if vram_frac is not None and vram_frac >= TRAIN_VRAM_PRESSURE_RATIO:
        return "VRAM-limited"

    if gpu_util is not None and gpu_util >= 90:
        return "GPU compute-bound"

    if gpu_util is not None and gpu_util <= 50:
        if cpu_percent is not None and cpu_percent >= 70:
            return "CPU/DataLoader-bound"

        if ram_frac is not None and ram_frac >= 0.75 and child_rss is not None and child_rss >= 8.0:
            return "DataLoader memory pressure"

        if workers <= 1:
            return "DataLoader underfeeding possible"

        return "I/O/DataLoader wait possible"

    return "balanced or unclear"


class _DeviceSampler:
    """Single background owner of every subprocess-backed measurement.

    Both device measurements available on Windows -- nvidia-smi for VRAM and
    GPU utilization, and the WDDM "Non Local Usage" performance counter via
    PowerShell (see _query_wddm_non_local_usage) -- cost an entire process
    launch per sample. Calling either one from on_train_batch_end blocks the
    training loop synchronously: the GPU drains its queued work and then sits
    idle until Python returns to submit the next batch. At a few hundred
    milliseconds to a second or more per PowerShell launch, against batches
    of a similar order, that alone can dominate step time -- so the measuring
    apparatus ends up depressing the very throughput it exists to protect,
    and, worse, the starting epoch's own measured throughput (which becomes
    _PerformanceMonitor's baseline for the whole run) and every batch-size
    verdict derived from it are all taken under those distorted conditions.

    This class removes that coupling entirely. One daemon thread samples on
    its own cadence and publishes results under a lock; the training path
    only ever calls snapshot(), which is a lock-protected tuple read and
    never spawns anything. Callers must treat any None in a snapshot as
    "unknown" -- never as zero -- exactly as with the underlying queries.
    """

    def __init__(
        self,
        device,
        interval: float = TRAIN_SAMPLER_INTERVAL_SECONDS,
        wddm_interval: float = TRAIN_SAMPLER_WDDM_INTERVAL_SECONDS,
    ) -> None:
        self.device = device
        self.runtime = runtime_for(device)
        self.interval = max(1.0, float(interval))
        self.wddm_interval = max(self.interval, float(wddm_interval))
        self._lock = threading.Lock()
        self._gpu_util: float | None = None
        self._vram_used: float | None = None
        self._vram_total: float | None = None
        self._vram_frac: float | None = None
        self._wddm_non_local: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Take one synchronous sample, then hand sampling to the thread.

        The first sample is taken here, before training starts, so that a
        pre-training WDDM baseline exists from batch 1 onward (see
        _InitialLoadCheck) rather than only from the thread's first tick.
        This one blocking call happens outside the training loop, where its
        cost is harmless.
        """
        gpu_util, used, total = self._sample_memory()
        non_local = (_query_wddm_non_local_usage(device=self.device)
                     if self.runtime.wddm else None)
        self._publish(gpu_util, used, total, non_local)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _sample_memory(self) -> tuple[float | None, float | None, float | None]:
        """Return (gpu_util_pct, used_GiB, total_GiB) for any accelerator.
        NVIDIA may supply utilization; HIP/MPS use their allocator primitives
        when no external provider is available."""
        return sample_accelerator(self.device)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 10.0)
            self._thread = None

    def _publish(
        self,
        gpu_util: float | None,
        used: float | None,
        total: float | None,
        non_local: float | None,
    ) -> None:
        with self._lock:
            self._gpu_util = gpu_util
            self._vram_used = used
            self._vram_total = total
            self._vram_frac = (used / total) if (used is not None and total) else None
            # A failed WDDM query means "unknown", so the previous known
            # value is kept rather than being overwritten with None.
            if non_local is not None:
                self._wddm_non_local = non_local

    def _loop(self) -> None:
        next_wddm = time.monotonic() + self.wddm_interval
        while not self._stop.wait(self.interval):
            gpu_util, used, total = self._sample_memory()
            non_local = None
            if time.monotonic() >= next_wddm:
                non_local = (_query_wddm_non_local_usage(device=self.device)
                             if self.runtime.wddm else None)
                next_wddm = time.monotonic() + self.wddm_interval
            self._publish(gpu_util, used, total, non_local)

    def snapshot(self) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        """Return (gpu_util_pct, vram_used_GiB, vram_total_GiB, vram_fraction,
        wddm_non_local_GiB) from the most recent samples. Never blocks on a
        subprocess -- safe to call from a per-batch callback."""
        with self._lock:
            return (
                self._gpu_util,
                self._vram_used,
                self._vram_total,
                self._vram_frac,
                self._wddm_non_local,
            )


class _PerformanceMonitor:
    """Periodic [PERF] logger for the whole run, plus a throughput watchdog
    for training and validation. Completed-batch rolling throughput is driven
    by phase-specific callbacks; an in-flight watchdog also handles a batch
    that has stopped producing callbacks long enough to prove a collapse.

    _InitialLoadCheck owns the entire starting epoch (index 0 of a fresh
    run, or whatever epoch a resume begins at) -- RAM pressure, WDDM
    observation, and VRAM headroom, none of them window-based (see its own
    docstring) -- and hands this class a baseline only once, when that
    epoch finishes: the epoch's own full training-only throughput (see
    main()). This class compares a rolling window of real measured throughput
    against that baseline for every epoch after it. Whenever the average is
    below TRAIN_COLLAPSE_RATIO of baseline over at least
    TRAIN_COLLAPSE_WINDOW_SECONDS, the drop is actionable only when the same
    cached device snapshot also confirms both WDDM non-local growth of at
    least TRAIN_WDDM_GROWTH_MIN_GIB above this process's pre-training baseline
    and VRAM pressure of at least TRAIN_VRAM_PRESSURE_RATIO. Pressure uses the
    strongest of current local usage, the attempt's observed local peak, and
    current local plus confirmed WDDM growth; the last reconstructs the
    pre-eviction working set after WDDM has lowered local VRAM. Missing or
    insufficient WDDM data, and low or unavailable VRAM, produce a warning
    and training continues. An in-flight batch is actionable only after its
    elapsed time exceeds both TRAIN_COLLAPSE_WINDOW_SECONDS and the time at
    which even one completed batch would remain below the collapse threshold.
    That lets a sustained sysmem-backed CUDA stall be stopped without waiting
    for its batch-end callback, while ordinary batch-time variation cannot
    trip the watchdog.

    record_batch is only ever called from on_train_batch_end -- i.e. only
    for real training batches, never during validation or a checkpoint
    save. That makes the accumulated window's elapsed time strictly real
    time *between recorded batches*, so validation/checkpoint-save time can
    never be counted in it -- unlike measuring on a wall-clock timer in a
    background thread (the previous design here), which keeps advancing
    regardless of what training is doing, and can only discover -- after
    the fact, on its *next* scheduled poll -- that an epoch boundary was
    crossed; a poll landing during that dead time, before the next epoch's
    first batch has been recorded, could already have computed a window
    whose elapsed time silently included it. record_batch still resets the
    accumulator on every epoch-boundary batch (the first batch of a new
    epoch), which is still necessary -- without it, elapsed time would
    still be measured from a batch before the validation gap to a batch
    after it -- but doing that reset synchronously, in the same call that
    detects the boundary, makes it exact instead of racing a separate
    thread's polling cadence.

    Validation uses a separate accumulator and never changes the official
    training baseline. It compares its own completed batches against the
    available training reference, a conservative check because validation is
    normally no slower than training. Validation time therefore cannot make a
    later training window look slow, while a validation-only WDDM collapse is
    still recoverable.

    The child background thread performs the in-flight check once per second,
    while the CLI supervisor independently reads an mmap heartbeat and samples
    the child PID. The supervisor remains responsive even if a CUDA call holds
    the child's GIL, and can terminate only that child before restarting with
    a lower batch. Once automatic recovery has reached batch=1, a confirmed
    WDDM throughput collapse is warned about but no longer aborts the run:
    there is no smaller usable batch, so slow training is allowed to continue.
    Disabling PERF output via AMADEUS_DISABLE_PERF_LOG suppresses only log
    lines; all collapse detection remains active.

    baseline_throughput starts at None until main() sets it once
    _InitialLoadCheck's take_official_throughput() has something to give it.
    During the starting epoch, completed batches may supply a conservative
    provisional training-only rate solely to protect against an in-flight
    batch that demonstrably collapses while spilling to WDDM. The provisional
    value never becomes the later-epoch baseline and WDDM growth without that
    measured stall remains non-actionable.
    """

    def __init__(
        self,
        *,
        device,
        sampler: _DeviceSampler,
        batch_size: int,
        image_size: int,
        workers: int,
        baseline_throughput: float | None = None,
        emergency_abort: Callable[[str], None] | None = None,
    ) -> None:
        self.device = device
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.workers = int(workers)
        self.effective_workers = effective_dataloader_workers(device, workers)
        self.baseline_throughput = baseline_throughput
        self.provisional_throughput: float | None = None
        self.emergency_abort = emergency_abort
        self.interval = max(3.0, float(os.environ.get("AMADEUS_PERF_LOG_INTERVAL", "15")))
        self._perf_log_enabled = str(
            os.environ.get("AMADEUS_DISABLE_PERF_LOG", "")
        ).strip().lower() not in {"1", "true", "yes"}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._batch_time_lock = threading.Lock()
        self._batch_inflight_since: float | None = None
        self._batch_inflight_size = self.batch_size
        self._epoch: int | None = None
        self._accum_batches = 0
        self._accum_start: float | None = None
        self._collapsed = threading.Event()
        self._collapsed_reason = ""
        self._pressure_action: str | None = None
        self._pressure_reason = ""
        self._last_rolling_throughput: float | None = None
        self._healthy_spill_warned = False
        self._batch_floor_warned = False
        self._last_stall_warning = ""
        self._observed_vram_peak: float | None = None
        self._val_batch_size = 0
        self._val_accum_batches = 0
        self._val_accum_start: float | None = None
        # The sampler is synchronously primed before this monitor is built,
        # so this is this process's own pre-training WDDM baseline. None is
        # deliberately retained as unknown: it must never be treated as zero.
        _u, _used, _total, _frac, self._wddm_non_local_baseline = sampler.snapshot()

    def _remember_vram_peak(self, vram_frac: float | None) -> float | None:
        if vram_frac is not None:
            self._observed_vram_peak = (
                vram_frac
                if self._observed_vram_peak is None
                else max(self._observed_vram_peak, vram_frac)
            )
        return self._observed_vram_peak

    def _keep_batch_floor(self, collapse_summary: str) -> bool:
        """Keep a slow but runnable batch=1 attempt instead of aborting it."""
        if self.batch_size > 1:
            return False
        if not self._batch_floor_warned:
            print(
                f"[WARN] {collapse_summary} with confirmed WDDM spill and VRAM "
                "pressure, but automatic recovery is already at batch=1; "
                "continuing training with batch=1.",
                flush=True,
            )
            self._batch_floor_warned = True
        return True

    def record_batch_start(self, epoch: int) -> float:
        """Mark a real training batch as in flight for the background watchdog."""
        now = time.monotonic()
        with self._batch_time_lock:
            self._batch_inflight_since = now
            self._batch_inflight_size = self.batch_size
        return now

    def _check_memory_pressure(self, phase: str) -> None:
        """Convert whole-run RAM/MPS pressure into an actionable recovery."""
        if self._pressure_action is not None:
            return
        if _accelerator_type(self.device) != "mps":
            return
        ram_frac = _query_ram_fraction()
        if ram_frac is not None and ram_frac >= TRAIN_INITIAL_RAM_TARGET:
            self._pressure_action = "lower_batch"
            self._pressure_reason = (
                f"RAM usage {ram_frac:.0%} reached {TRAIN_INITIAL_RAM_TARGET:.0%} "
                f"during {phase}; configured_workers={self.workers}, "
                f"effective_workers={self.effective_workers}"
            )
            return

        if str(self.device).strip().lower() == "mps":
            _used, _total, _frac, _wddm = self.sampler.snapshot()[1:]
            if _frac is not None and _frac >= TRAIN_VRAM_PRESSURE_RATIO:
                self._pressure_action = "lower_batch"
                self._pressure_reason = (
                    f"MPS driver allocation reached {_frac:.1%} of its recommended "
                    f"working set during {phase} "
                    f"(threshold {TRAIN_VRAM_PRESSURE_RATIO:.1%})"
                )

    def pressure_verdict(self) -> tuple[str, str] | None:
        if self._pressure_action is None:
            return None
        return self._pressure_action, self._pressure_reason

    def record_batch(self, epoch: int) -> None:
        self._check_memory_pressure(f"training epoch {epoch + 1}")
        now = time.monotonic()
        with self._batch_time_lock:
            self._batch_inflight_since = None
        if epoch != self._epoch:
            # Epoch boundary (or the very first call ever, since _epoch
            # starts at None and no real epoch equals that) -- start fresh.
            # See the class docstring for why this reset, done synchronously
            # here, is both necessary and now exact.
            self._epoch = epoch
            self._accum_batches = 0
            self._accum_start = now
            return

        if not self.baseline_throughput or self._accum_start is None:
            # No baseline yet (still inside the starting epoch -- see
            # _InitialLoadCheck) -- nothing to compare against.
            self._accum_batches = 0
            self._accum_start = now
            return

        self._accum_batches += 1
        accum_elapsed = now - self._accum_start
        if accum_elapsed < TRAIN_COLLAPSE_WINDOW_SECONDS:
            return

        rolling_throughput = self._accum_batches * self.batch_size / accum_elapsed
        self._last_rolling_throughput = rolling_throughput
        self._accum_batches = 0
        self._accum_start = now

        # Read from the shared background sampler rather than spawning
        # nvidia-smi or PowerShell here: this runs inside on_train_batch_end,
        # and cached samples are ample confirmation for a condition averaged
        # over TRAIN_COLLAPSE_WINDOW_SECONDS.
        _gpu_util, vram_used, vram_total, vram_frac, non_local = self.sampler.snapshot()
        vram_peak = self._remember_vram_peak(vram_frac)
        wddm_growth = (
            non_local - self._wddm_non_local_baseline
            if non_local is not None and self._wddm_non_local_baseline is not None
            else None
        )
        throughput_collapsed = (
            rolling_throughput < self.baseline_throughput * TRAIN_COLLAPSE_RATIO
        )

        if not throughput_collapsed:
            if (
                not self._healthy_spill_warned
                and wddm_growth is not None
                and wddm_growth >= TRAIN_WDDM_GROWTH_MIN_GIB
                and vram_frac is not None
                and vram_frac >= TRAIN_VRAM_PRESSURE_RATIO
            ):
                print(
                    f"[WARN] WDDM non-local memory increased by {wddm_growth:.2f}GiB "
                    f"while VRAM usage is {vram_frac:.1%} "
                    f"({_fmt_gib(vram_used)}/{_fmt_gib(vram_total)}), but throughput "
                    f"remains healthy at {rolling_throughput:.1f} img/s "
                    f"(baseline {self.baseline_throughput:.1f} img/s); "
                    f"keeping batch={self.batch_size}.",
                    flush=True,
                )
                self._healthy_spill_warned = True
            return

        collapse_summary = (
            f"throughput collapsed to {rolling_throughput:.1f} img/s "
            f"(baseline {self.baseline_throughput:.1f} img/s)"
        )
        if wddm_growth is None:
            print(
                f"[WARN] {collapse_summary}, but WDDM non-local memory is unavailable, "
                f"so no WDDM spill is confirmed; keeping batch={self.batch_size} "
                "and continuing training.",
                flush=True,
            )
            return
        if wddm_growth < TRAIN_WDDM_GROWTH_MIN_GIB:
            print(
                f"[WARN] {collapse_summary}, but WDDM non-local growth is only "
                f"{wddm_growth:.2f}GiB (below {TRAIN_WDDM_GROWTH_MIN_GIB:.2f}GiB), "
                f"so no WDDM spill is confirmed; keeping batch={self.batch_size} "
                "and continuing training.",
                flush=True,
            )
            return
        pressure_vram_frac = _vram_pressure_fraction(
            vram_used=vram_used,
            vram_total=vram_total,
            current_frac=vram_frac,
            peak_frac=vram_peak,
            wddm_growth=wddm_growth,
        )
        if pressure_vram_frac is None:
            print(
                f"[WARN] {collapse_summary} with confirmed WDDM non-local growth of "
                f"{wddm_growth:.2f}GiB, but current VRAM usage is unavailable; "
                f"keeping batch={self.batch_size} and continuing training.",
                flush=True,
            )
            return
        if pressure_vram_frac < TRAIN_VRAM_PRESSURE_RATIO:
            current_vram_text = "unavailable" if vram_frac is None else f"{vram_frac:.1%}"
            print(
                f"[WARN] {collapse_summary} with confirmed WDDM non-local growth of "
                f"{wddm_growth:.2f}GiB, but VRAM usage is only {current_vram_text} "
                f"(pressure-equivalent {pressure_vram_frac:.1%}); "
                f"keeping batch={self.batch_size} and continuing training.",
                flush=True,
            )
            return

        if self._keep_batch_floor(collapse_summary):
            return

        current_vram_text = "unavailable" if vram_frac is None else f"{vram_frac:.1%}"
        self._collapsed_reason = (
            "throughput collapse with confirmed WDDM spill and VRAM pressure: "
            f"{collapse_summary}, averaged over the last {accum_elapsed:.0f}s; "
            f"WDDM non-local grew by {wddm_growth:.2f}GiB "
            f"(from {self._wddm_non_local_baseline:.2f}GiB to {non_local:.2f}GiB); "
            f"VRAM usage is {current_vram_text} (pressure-equivalent {pressure_vram_frac:.1%}) "
            f"({_fmt_gib(vram_used)}/{_fmt_gib(vram_total)})"
        )
        self._collapsed.set()

    def start_validation(self, batch_size: int) -> None:
        self._val_batch_size = max(1, int(batch_size))
        self._val_accum_batches = 0
        self._val_accum_start = None

    def record_validation_batch_start(self, batch_size: int) -> float:
        now = time.monotonic()
        self._val_batch_size = max(1, int(batch_size))
        if self._val_accum_start is None:
            self._val_accum_start = now
        with self._batch_time_lock:
            self._batch_inflight_since = now
            self._batch_inflight_size = self._val_batch_size
        return now

    def record_validation_batch_end(self) -> None:
        self._check_memory_pressure("validation")
        now = time.monotonic()
        with self._batch_time_lock:
            self._batch_inflight_since = None
        reference_throughput = self.baseline_throughput or self.provisional_throughput
        if (
            self._collapsed.is_set()
            or not reference_throughput
            or self._val_accum_start is None
        ):
            return

        self._val_accum_batches += 1
        elapsed = now - self._val_accum_start
        if elapsed < TRAIN_COLLAPSE_WINDOW_SECONDS:
            return
        throughput = self._val_accum_batches * self._val_batch_size / elapsed
        self._val_accum_batches = 0
        self._val_accum_start = now
        if throughput >= reference_throughput * TRAIN_COLLAPSE_RATIO:
            return

        _gpu_util, vram_used, vram_total, vram_frac, non_local = self.sampler.snapshot()
        vram_peak = self._remember_vram_peak(vram_frac)
        wddm_growth = (
            non_local - self._wddm_non_local_baseline
            if non_local is not None and self._wddm_non_local_baseline is not None
            else None
        )
        collapse_summary = (
            f"validation throughput collapsed to {throughput:.1f} img/s "
            f"(training reference {reference_throughput:.1f} img/s)"
        )
        if wddm_growth is None:
            print(
                f"[WARN] {collapse_summary}, but WDDM non-local memory is unavailable, "
                f"so no WDDM spill is confirmed; keeping batch={self.batch_size} "
                "and continuing validation.",
                flush=True,
            )
            return
        if wddm_growth < TRAIN_WDDM_GROWTH_MIN_GIB:
            print(
                f"[WARN] {collapse_summary}, but WDDM non-local growth is only "
                f"{wddm_growth:.2f}GiB, so no WDDM spill is confirmed; "
                f"keeping batch={self.batch_size} and continuing validation.",
                flush=True,
            )
            return
        pressure_vram_frac = _vram_pressure_fraction(
            vram_used=vram_used,
            vram_total=vram_total,
            current_frac=vram_frac,
            peak_frac=vram_peak,
            wddm_growth=wddm_growth,
        )
        if pressure_vram_frac is None or pressure_vram_frac < TRAIN_VRAM_PRESSURE_RATIO:
            vram_text = "unavailable" if vram_frac is None else f"only {vram_frac:.1%}"
            print(
                f"[WARN] {collapse_summary} with confirmed WDDM non-local growth of "
                f"{wddm_growth:.2f}GiB, but VRAM usage is {vram_text}; "
                f"keeping batch={self.batch_size} and continuing validation.",
                flush=True,
            )
            return

        if self._keep_batch_floor(collapse_summary):
            return

        current_vram_text = "unavailable" if vram_frac is None else f"{vram_frac:.1%}"
        self._collapsed_reason = (
            "validation throughput collapse with confirmed WDDM spill and VRAM pressure: "
            f"{collapse_summary}, averaged over the last {elapsed:.0f}s; WDDM non-local "
            f"grew by {wddm_growth:.2f}GiB (from {self._wddm_non_local_baseline:.2f}GiB "
            f"to {non_local:.2f}GiB); VRAM usage is {current_vram_text} "
            f"(pressure-equivalent {pressure_vram_frac:.1%}) "
            f"({_fmt_gib(vram_used)}/{_fmt_gib(vram_total)})"
        )
        self._collapsed.set()

    def collapsed(self) -> bool:
        return self._collapsed.is_set()

    def collapsed_reason(self) -> str:
        return self._collapsed_reason

    def start(self) -> None:
        if self._perf_log_enabled:
            try:
                import psutil  # noqa: F401
                # Prime psutil.cpu_percent() so later samples are meaningful.
                psutil.cpu_percent(interval=None)
            except Exception:
                pass
            print(
                f"[PERF] monitor started: interval={self.interval:.0f}s, "
                f"batch={self.batch_size}, imgsz={self.image_size}, "
                f"workers={self.effective_workers} (configured={self.workers}), "
                f"device={self.device}",
                flush=True,
            )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _warn_stalled_batch_once(self, key: str, message: str) -> None:
        if key == self._last_stall_warning:
            return
        self._last_stall_warning = key
        print(message, flush=True)

    def _check_inflight_collapse(self, now: float) -> None:
        """Detect a collapsed batch while CUDA is still executing it.

        No callback can interrupt a CUDA call that has not returned. This
        method therefore signals emergency_abort after the same three pieces
        of evidence as the completed-window path: a conservative throughput
        upper bound below the configured ratio, confirmed WDDM growth, and
        VRAM pressure. The supervised CLI uses that signal to terminate only
        the CUDA child process and restart from the last checkpoint at a lower
        batch size.
        """
        reference_throughput = self.baseline_throughput or self.provisional_throughput
        if self._collapsed.is_set() or not reference_throughput:
            return
        with self._batch_time_lock:
            batch_inflight_since = self._batch_inflight_since
            batch_inflight_size = self._batch_inflight_size
        if batch_inflight_since is None:
            return

        stalled_for = now - batch_inflight_since
        collapse_rate = reference_throughput * TRAIN_COLLAPSE_RATIO
        if collapse_rate <= 0:
            return
        # Even if the in-flight batch completed immediately, one batch over
        # this elapsed interval would still fall below collapse_rate.
        minimum_evidence_seconds = max(
            TRAIN_COLLAPSE_WINDOW_SECONDS,
            batch_inflight_size / collapse_rate,
        )
        if stalled_for < minimum_evidence_seconds:
            return

        _gpu_util, vram_used, vram_total, vram_frac, non_local = self.sampler.snapshot()
        vram_peak = self._remember_vram_peak(vram_frac)
        wddm_growth = (
            non_local - self._wddm_non_local_baseline
            if non_local is not None and self._wddm_non_local_baseline is not None
            else None
        )
        upper_bound_throughput = batch_inflight_size / stalled_for
        reference_name = (
            "baseline" if self.baseline_throughput else "provisional starting-epoch rate"
        )
        stall_summary = (
            f"no training batch completed for {stalled_for:.0f}s; throughput is at most "
            f"{upper_bound_throughput:.2f} img/s "
            f"({reference_name} {reference_throughput:.1f} img/s)"
        )

        if wddm_growth is None:
            self._warn_stalled_batch_once(
                "wddm-unavailable",
                f"[WARN] {stall_summary}, but WDDM non-local memory is unavailable, "
                f"so no WDDM spill is confirmed; keeping batch={self.batch_size} "
                "and continuing training.",
            )
            return
        if wddm_growth < TRAIN_WDDM_GROWTH_MIN_GIB:
            self._warn_stalled_batch_once(
                "wddm-insufficient",
                f"[WARN] {stall_summary}, but WDDM non-local growth is only "
                f"{wddm_growth:.2f}GiB (below {TRAIN_WDDM_GROWTH_MIN_GIB:.2f}GiB), "
                f"so no WDDM spill is confirmed; keeping batch={self.batch_size} "
                "and continuing training.",
            )
            return
        pressure_vram_frac = _vram_pressure_fraction(
            vram_used=vram_used,
            vram_total=vram_total,
            current_frac=vram_frac,
            peak_frac=vram_peak,
            wddm_growth=wddm_growth,
        )
        if pressure_vram_frac is None:
            self._warn_stalled_batch_once(
                "vram-unavailable",
                f"[WARN] {stall_summary} with confirmed WDDM non-local growth of "
                f"{wddm_growth:.2f}GiB, but current VRAM usage is unavailable; "
                f"keeping batch={self.batch_size} and continuing training.",
            )
            return
        if pressure_vram_frac < TRAIN_VRAM_PRESSURE_RATIO:
            current_vram_text = "unavailable" if vram_frac is None else f"{vram_frac:.1%}"
            self._warn_stalled_batch_once(
                "vram-insufficient",
                f"[WARN] {stall_summary} with confirmed WDDM non-local growth of "
                f"{wddm_growth:.2f}GiB, but VRAM usage is only {current_vram_text} "
                f"(pressure-equivalent {pressure_vram_frac:.1%}); "
                f"keeping batch={self.batch_size} and continuing training.",
            )
            return

        # A batch may have completed while the sampler snapshot was read.
        # Recheck the timestamp before committing to an emergency termination.
        with self._batch_time_lock:
            if self._batch_inflight_since != batch_inflight_since:
                return
        if self._keep_batch_floor(stall_summary):
            return
        current_vram_text = "unavailable" if vram_frac is None else f"{vram_frac:.1%}"
        self._collapsed_reason = (
            "throughput collapse with confirmed WDDM spill and VRAM pressure: "
            f"{stall_summary}; WDDM non-local grew by {wddm_growth:.2f}GiB "
            f"(from {self._wddm_non_local_baseline:.2f}GiB to {non_local:.2f}GiB); "
            f"VRAM usage is {current_vram_text} (pressure-equivalent {pressure_vram_frac:.1%}) "
            f"({_fmt_gib(vram_used)}/{_fmt_gib(vram_total)})"
        )
        self._collapsed.set()
        if self.emergency_abort is not None:
            self.emergency_abort(self._collapsed_reason)

    def _loop(self) -> None:
        next_perf_log = time.monotonic() + self.interval
        while not self._stop.wait(1.0):
            now = time.monotonic()
            self._check_memory_pressure("training")
            self._remember_vram_peak(self.sampler.snapshot()[3])
            self._check_inflight_collapse(now)
            if not self._perf_log_enabled or now < next_perf_log:
                continue
            next_perf_log = now + self.interval
            gpu_util, vram_used, vram_total, _vram_frac, wddm_non_local = self.sampler.snapshot()
            torch_alloc, torch_reserved = _query_torch_vram(self.device)
            cpu_percent, ram_used, ram_total, proc_rss, child_rss = _query_cpu_memory()
            bottleneck = _infer_bottleneck(
                gpu_util,
                vram_used,
                vram_total,
                cpu_percent,
                ram_used,
                ram_total,
                child_rss,
                self.workers,
            )
            ratio = (
                self._last_rolling_throughput / self.baseline_throughput
                if self._last_rolling_throughput is not None and self.baseline_throughput
                else None
            )
            print(
                "[PERF] "
                f"gpu={_fmt_pct(gpu_util)} "
                f"vram={_fmt_gib(vram_used)}/{_fmt_gib(vram_total)} "
                f"torch_alloc={_fmt_gib(torch_alloc)} torch_reserved={_fmt_gib(torch_reserved)} "
                f"wddm_non_local={_fmt_gib(wddm_non_local)} "
                f"cpu={_fmt_pct(cpu_percent)} "
                f"ram={_fmt_gib(ram_used)}/{_fmt_gib(ram_total)} "
                f"proc_ram={_fmt_gib(proc_rss)} child_ram={_fmt_gib(child_rss)} "
                f"throughput={_fmt_rate(self._last_rolling_throughput)} "
                f"baseline={_fmt_rate(self.baseline_throughput)} "
                f"ratio={_fmt_pct(ratio * 100 if ratio is not None else None)} "
                f"batch={self.batch_size} workers={self.workers} imgsz={self.image_size} "
                f"bottleneck={bottleneck}",
                flush=True,
            )


class _BatchTooLargeError(RuntimeError):
    """Raised internally when a training attempt needs automatic recovery.

    Causes include OOM, GPU->CPU fallback, a DataLoader failure, RAM pressure,
    VRAM headroom, or a measured throughput collapse with confirmed WDDM spill
    and VRAM pressure (a completed training/validation window, or a supervised
    in-flight stall once a reference rate exists). WDDM-related collapse stops
    generating this signal once automatic recovery is already at batch=1;
    slow but runnable training then continues (see
    _InitialLoadCheck / _PerformanceMonitor).

    .action tells main()'s retry loop what to adjust: "lower_batch",
    "lower_workers", or "raise_batch". Batch reductions use the existing
    ~2/3 step from next_batch_candidate_soft_down. An explicit BATCH_SIZE or
    NUM_WORKERS is never silently changed; the relevant path raises a plain
    RuntimeError instead."""

    def __init__(self, message: str, action: str = "lower_batch") -> None:
        super().__init__(message)
        self.action = action


class _GpuFallbackWatcher(logging.Handler):
    """Watches Ultralytics' own LOGGER for TaskAlignedAssigner's silent
    CUDA-OOM-to-CPU fallback (ultralytics.utils.tal.TaskAlignedAssigner.forward
    catches torch.cuda.OutOfMemoryError internally and recomputes on CPU for
    that one batch, without raising). Left undetected, training would just
    keep running -- correctly, but drastically slower -- with no exception to
    catch. Active for the whole run (not just the starting epoch): it's a
    passive log hook, not a poll, so there's no cost to leaving it on. It
    requests a lower batch whenever one exists; at batch=1 the fallback is
    warned about and allowed to continue because Ultralytics did complete the
    operation on CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.triggered = threading.Event()
        self.message = ""

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "TaskAlignedAssigner" in msg and "using CPU" in msg:
            self.message = msg
            self.triggered.set()


def _take_gpu_fallback_action(
    watcher: _GpuFallbackWatcher, batch_size: int
) -> tuple[str, str] | None:
    """Consume one fallback signal and decide whether recovery is possible."""
    if not watcher.triggered.is_set():
        return None
    message = watcher.message
    watcher.triggered.clear()
    action = "lower_batch" if int(batch_size) > 1 else "continue"
    return action, message


class _TrainingAborted(Exception):
    """Internal sentinel raised from training, validation, or epoch callbacks
    when the GPU-fallback watcher (above batch=1), _InitialLoadCheck (RAM
    pressure or VRAM
    headroom), or _PerformanceMonitor (a measured throughput collapse with
    confirmed WDDM spill and VRAM pressure) finds a clear, actionable
    condition. A WDDM-related collapse is not actionable at batch=1 and only
    produces a warning so training can continue. Other callback-detected cases
    are caught in _do_train and converted to _BatchTooLargeError with the same
    .action. Actual CUDA OOM exceptions
    are converted separately; silent Ultralytics CUDA-OOM-to-CPU fallback is
    reported through the watcher. After adjusting batch/workers, main()
    resumes any valid checkpoint when ACCEPT_RESUME is enabled and otherwise
    starts fresh."""

    def __init__(self, action: str, reason: str) -> None:
        super().__init__(reason)
        self.action = action


def _checkpoint_train_args(path: str) -> dict:
    ckpt = _torch_load_checkpoint(path)
    if not isinstance(ckpt, dict):
        return {}
    return dict(ckpt.get("train_args") or {})


def _reset_run_directory(project_dir: str, experiment_name: str) -> None:
    """Discard a partial run only when recovery must start fresh."""
    run_dir = os.path.join(project_dir, experiment_name)
    if os.path.isdir(run_dir):
        print(f"[WARN] Discarding partial run directory: {run_dir}")
        shutil.rmtree(run_dir, ignore_errors=True)


def _load_emergency_recovery_state(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        batch = int(state["batch"])
        workers = int(state["workers"])
        if batch < 1 or workers < 0:
            raise ValueError("negative recovery value")
        return {"batch": batch, "workers": workers}
    except Exception as exc:
        raise RuntimeError(f"Invalid emergency recovery state: {path}: {exc}") from exc


def _write_emergency_recovery_state(path: str, *, batch: int, workers: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump({"batch": int(batch), "workers": int(workers)}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.isfile(temp_path):
            os.remove(temp_path)


def _clear_emergency_recovery_state(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


class _WatchdogSharedState:
    """Tiny mmap heartbeat shared between the CUDA child and CLI supervisor."""

    def __init__(self, path: str, *, owner: bool = False) -> None:
        self.path = path
        self.owner = owner
        self._file = open(path, "r+b")
        self._map = mmap.mmap(self._file.fileno(), _WATCHDOG_STATE_STRUCT.size)

    @classmethod
    def create(cls) -> "_WatchdogSharedState":
        fd, path = tempfile.mkstemp(prefix="amadeus-train-watchdog-", suffix=".bin")
        os.close(fd)
        with open(path, "wb") as f:
            f.truncate(_WATCHDOG_STATE_STRUCT.size)
        return cls(path, owner=True)

    def publish(
        self,
        *,
        inflight_since: float,
        reference_throughput: float,
        batch: int,
        recovery_batch: int,
        workers: int,
        device_index: int,
        active: bool,
        can_lower_batch: bool,
    ) -> None:
        current_seq = _WATCHDOG_STATE_STRUCT.unpack_from(self._map)[0]
        odd_seq = current_seq + 1 if current_seq % 2 == 0 else current_seq + 2
        even_seq = odd_seq + 1
        _WATCHDOG_STATE_STRUCT.pack_into(
            self._map,
            0,
            odd_seq,
            float(inflight_since),
            float(reference_throughput),
            int(batch),
            int(recovery_batch),
            int(workers),
            int(device_index),
            int(bool(active)),
            int(bool(can_lower_batch)),
        )
        struct.pack_into("<Q", self._map, 0, even_seq)

    def read(self) -> dict | None:
        for _ in range(3):
            first = _WATCHDOG_STATE_STRUCT.unpack_from(self._map)
            if first[0] % 2:
                continue
            second = _WATCHDOG_STATE_STRUCT.unpack_from(self._map)
            if first[0] == second[0]:
                return {
                    "inflight_since": second[1],
                    "reference_throughput": second[2],
                    "batch": second[3],
                    "recovery_batch": second[4],
                    "workers": second[5],
                    "device_index": second[6],
                    "active": bool(second[7]),
                    "can_lower_batch": bool(second[8]),
                }
        return None

    def close(self) -> None:
        self._map.close()
        self._file.close()
        if self.owner:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass


class _InitialLoadCheck:
    """One-shot real-load sanity check covering this training process's
    starting epoch, in full -- not a rolling watchdog for the rest of
    training, and not a separate throwaway probe: it reuses real training's
    own batches, so it costs nothing beyond what training already does.
    There is no short measurement window anymore (no warmup skip, no
    fixed-duration sample-then-decide step) -- every check here either
    fires immediately when its own condition is met, or is only decided
    once, at the moment the starting epoch itself ends.

    "Starting epoch" is whatever epoch on_train_batch_end first reports it
    as -- epoch index 0 for a fresh run, or the resume epoch for a resumed
    one (see record_batch) -- so the same one-shot check runs after every
    restart, resumed or not. Three independent things happen during it:

    - RAM pressure: checked on every batch via _query_ram_fraction (one
      psutil.virtual_memory() read, cheap enough to not need throttling);
      the instant system RAM reaches TRAIN_INITIAL_RAM_TARGET, verdict
      becomes "lower_workers".
    - WDDM spill observation (Windows only -- see
      _query_wddm_non_local_usage):
      this process's own WDDM "Non Local Usage" baseline is captured in
      __init__ from _DeviceSampler's pre-training sample, so a spill can be
      observed from batch 1 onward. Growth over that baseline together with
      VRAM pressure is logged once, but never aborts the starting epoch and
      never lowers batch by itself. The epoch is allowed to finish so its
      training-only throughput can establish whether the spill has any real
      performance impact. Actual OOM and the separate GPU-fallback watcher
      remain immediately actionable throughout this epoch.
    - VRAM headroom: peak VRAM fraction is taken from the same snapshots,
      but only *decided* once, the moment the starting epoch ends without
      RAM pressure -- if the peak never reached
      TRAIN_INITIAL_VRAM_HEADROOM_TRIGGER for the *entire* epoch, verdict
      becomes "raise_batch".

    Every device figure used here comes from _DeviceSampler.snapshot(), a
    lock-protected read of values a background thread has already collected.
    Nothing on this path spawns a subprocess, so the starting epoch runs at
    the same speed as every epoch after it -- which is what makes its
    measured throughput a valid baseline for the rest of the run, and the
    batch-size verdicts above decisions about the hardware rather than about
    the cost of measuring it.

    Completed batches in the starting epoch build a provisional training-only
    rate for one narrow purpose: the background watchdog may use it to stop a
    later in-flight batch only when its elapsed-time throughput upper bound
    has collapsed and the same snapshot confirms WDDM spill plus VRAM
    pressure. This preserves the rule that WDDM growth alone never aborts the
    starting epoch while still handling a batch that would otherwise never
    reach its callback. If the whole starting epoch completes without RAM
    pressure, its own full
    training-only throughput (validation/checkpoint-save time is naturally
    excluded, since no on_train_batch_end fires during either) becomes
    _PerformanceMonitor's baseline for every epoch after it (see
    take_official_throughput and main()) -- only then does ordinary
    completed-window throughput-collapse monitoring begin. Batch is lowered
    only when a measured throughput collapse, confirmed WDDM growth, and VRAM
    pressure are all present in _PerformanceMonitor.

    allow_headroom_raise gates the "raise_batch" verdict: VRAM-headroom
    scale-up is a one-time opportunity for the whole run, not something to
    re-check after every restart, so main() passes False once it has
    already been applied once.
    """

    def __init__(
        self,
        *,
        device,
        sampler: _DeviceSampler,
        batch_size: int,
        workers: int,
        allow_headroom_raise: bool,
    ) -> None:
        self.device = device
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.workers = int(workers)
        self.effective_workers = effective_dataloader_workers(device, workers)
        self.allow_headroom_raise = allow_headroom_raise
        self._start_epoch: int | None = None
        self._epoch_done = False
        self._verdict: tuple[str, str] | None = None
        # Full-starting-epoch training-only throughput.
        self._epoch_batch_count = 0
        self._epoch_measure_start: float | None = None
        self._epoch_last_batch_time: float | None = None
        self._official_throughput: float | None = None
        self._official_throughput_consumed = False
        # Peak VRAM fraction over the whole starting epoch, for the
        # once-at-epoch-end raise_batch decision.
        self._max_vram_frac: float | None = None
        # WDDM spill observation -- baseline taken from the sampler's own
        # pre-training sample (see _DeviceSampler.start), so growth is
        # detectable from batch 1 onward without this constructor blocking
        # on a PowerShell launch of its own.
        _u, _used, _total, _frac, self._wddm_non_local_baseline = sampler.snapshot()
        self._wddm_spill_warned = False

    def record_batch(self, epoch: int) -> None:
        if self._start_epoch is None:
            self._start_epoch = epoch
        if self._epoch_done:
            return

        if epoch != self._start_epoch:
            # The starting epoch just ended cleanly -- nothing else here
            # would still be reaching this branch otherwise, since RAM
            # pressure sets _epoch_done immediately when it fires (see
            # below). Its own full training-only
            # throughput becomes the baseline handed to main() via
            # take_official_throughput(); VRAM headroom is decided now too,
            # using the whole epoch's own peak rather than any short window.
            self._finish_starting_epoch()
            return

        now = time.monotonic()
        if self._epoch_measure_start is None:
            # Clock starts at this first batch's completion, so it isn't
            # itself counted -- only batches after it are, matching the
            # elapsed interval this starts measuring from.
            self._epoch_measure_start = now
        else:
            self._epoch_batch_count += 1
        self._epoch_last_batch_time = now

        ram_frac = _query_ram_fraction()
        if ram_frac is not None and ram_frac >= TRAIN_INITIAL_RAM_TARGET:
            action = "lower_batch" if self.effective_workers == 0 else "lower_workers"
            self._verdict = (
                action,
                f"RAM usage {ram_frac:.0%} reached {TRAIN_INITIAL_RAM_TARGET:.0%} "
                f"during epoch {self._start_epoch + 1}; configured_workers={self.workers}, "
                f"effective_workers={self.effective_workers}",
            )
            self._epoch_done = True
            return

        self._evaluate_device_state()

    def finish_epoch(self, epoch: int) -> None:
        """Finalize the starting epoch from on_train_epoch_end.

        Doing this before validation makes the baseline available before the
        next epoch's very first batch starts, which is required for the
        in-flight watchdog to catch a collapse in that batch. The epoch-change
        branch in record_batch remains as a defensive fallback for callers
        that do not provide an epoch-end callback.
        """
        if self._epoch_done or self._start_epoch is None or epoch != self._start_epoch:
            return
        self._finish_starting_epoch()

    def _finish_starting_epoch(self) -> None:
        self._finalize_epoch_throughput()
        if (
            self.allow_headroom_raise
            and self._max_vram_frac is not None
            and self._max_vram_frac < TRAIN_INITIAL_VRAM_HEADROOM_TRIGGER
        ):
            self._verdict = (
                "raise_batch",
                f"VRAM usage stayed at {self._max_vram_frac:.0%} throughout epoch "
                f"{self._start_epoch + 1} (below {TRAIN_INITIAL_VRAM_HEADROOM_TRIGGER:.0%})",
            )
        self._epoch_done = True

    def _evaluate_device_state(self) -> None:
        """Evaluate the sampler's latest device snapshot. Updates
        self._max_vram_frac and warns once when WDDM growth and VRAM pressure
        coexist, but never sets a lower_batch verdict: spill alone has no
        demonstrated performance impact during the baseline epoch. Reads
        cached values only: no subprocess, no throttle needed, safe on every
        batch."""
        _gpu_util, vram_used, vram_total, vram_frac, non_local = self.sampler.snapshot()

        if vram_frac is not None:
            self._max_vram_frac = (
                vram_frac if self._max_vram_frac is None else max(self._max_vram_frac, vram_frac)
            )

            if (
                str(self.device).strip().lower() == "mps"
                and vram_frac >= TRAIN_VRAM_PRESSURE_RATIO
            ):
                self._verdict = (
                    "lower_batch",
                    f"MPS driver allocation reached {vram_frac:.1%} of its recommended "
                    f"working set during epoch {self._start_epoch + 1} "
                    f"(threshold {TRAIN_VRAM_PRESSURE_RATIO:.1%})",
                )
                self._epoch_done = True
                return

        if self._wddm_non_local_baseline is None or non_local is None or vram_frac is None:
            return
        growth = non_local - self._wddm_non_local_baseline
        if growth < TRAIN_WDDM_GROWTH_MIN_GIB or vram_frac < TRAIN_VRAM_PRESSURE_RATIO:
            return
        if not self._wddm_spill_warned:
            print(
                f"[WARN] WDDM non-local memory increased by {growth:.2f}GiB during "
                f"starting epoch {self._start_epoch + 1} "
                f"(from {self._wddm_non_local_baseline:.2f}GiB to {non_local:.2f}GiB) "
                f"while VRAM usage is {vram_frac:.1%} "
                f"({_fmt_gib(vram_used)}/{_fmt_gib(vram_total)}); throughput baseline "
                f"is still being measured, so keeping batch={self.batch_size} and "
                "continuing training.",
                flush=True,
            )
            self._wddm_spill_warned = True

    def _finalize_epoch_throughput(self) -> None:
        if (
            self._epoch_batch_count > 0
            and self._epoch_measure_start is not None
            and self._epoch_last_batch_time is not None
        ):
            elapsed = self._epoch_last_batch_time - self._epoch_measure_start
            if elapsed > 0:
                self._official_throughput = self._epoch_batch_count * self.batch_size / elapsed

    def provisional_throughput(self) -> float | None:
        """Return the current starting-epoch training-only rate when stable enough.

        At least three measured inter-batch intervals are required. This value
        is never an official later-epoch baseline; it only lets the in-flight
        watchdog prove that a subsequent batch has collapsed instead of
        treating WDDM growth itself as actionable.
        """
        if (
            self._epoch_batch_count < 3
            or self._epoch_measure_start is None
            or self._epoch_last_batch_time is None
        ):
            return None
        elapsed = self._epoch_last_batch_time - self._epoch_measure_start
        if elapsed <= 0:
            return None
        return self._epoch_batch_count * self.batch_size / elapsed

    def verdict(self) -> tuple[str, str] | None:
        """Returns (action, reason) the moment one is decided (action is
        "lower_workers" or "raise_batch" -- see the class docstring); None as
        long as the starting epoch is still running cleanly. WDDM growth does
        not produce a verdict. There is no "ok" verdict: reaching the end of the starting
        epoch without one of these firing is signaled separately, by
        take_official_throughput() returning a value."""
        return self._verdict

    def take_official_throughput(self) -> float | None:
        """Returns the finalized whole-starting-epoch training-only
        throughput exactly once, the moment it becomes available (when the
        starting epoch ends without RAM pressure, which aborts via verdict()
        before the epoch could finish), and None on every
        call before or after that -- so main() can use a plain non-None
        check to know when to set it as _PerformanceMonitor's baseline,
        without tracking its own separate has-this-been-applied flag."""
        if self._official_throughput is None or self._official_throughput_consumed:
            return None
        self._official_throughput_consumed = True
        return self._official_throughput


def main(
    PRETRAINED_MODEL: str,
    EPOCHS: int = 50,
    LR0: float = DEFAULT_LR0,
    LRF: float = DEFAULT_LRF,
    TRAIN_IMG_SIZE: int = 640,
    BATCH_SIZE=16,
    DEVICE: str = "auto",
    DATASET_DIR: str = "yolo_dataset",
    NUM_CLASSES: int = 8,
    CLASS_NAMES: List[str] | None = None,
    PROJECT_DIR: str = "runs/training",
    EXPERIMENT_NAME: str = "detection",
    NUM_WORKERS="auto",
    SAVE_PERIOD=5,
    RANDOM_SEED: int = 0,
    ACCEPT_RESUME: bool = True,
) -> str:
    if CLASS_NAMES is None:
        CLASS_NAMES = list(DIRECTION_CLASS_NAMES)

    _apply_amp_override()

    EPOCHS = resolve_epoch_count(EPOCHS)
    lr0 = require_single_lr0(LR0)
    lrf = require_single_lrf(LRF)
    random_seed = normalize_seed(RANDOM_SEED)
    device = resolve_device(DEVICE, purpose="training")

    verify_dataset_dirs(DATASET_DIR)
    data_yaml = get_dataset_yaml_path(DATASET_DIR)
    verify_data_yaml(data_yaml, NUM_CLASSES, CLASS_NAMES)

    train_labels_dir = os.path.join(DATASET_DIR, "train", "labels")

    weights_dir = os.path.join(PROJECT_DIR, EXPERIMENT_NAME, "weights")
    last_path = os.path.join(weights_dir, "last.pt")
    results_csv = os.path.join(PROJECT_DIR, EXPERIMENT_NAME, "results.csv")
    emergency_recovery_path = os.path.join(
        PROJECT_DIR, EXPERIMENT_NAME, _EMERGENCY_RECOVERY_FILENAME
    )

    trainer_cls = _get_epochs_obb_trainer()

    explicit_batch = not (BATCH_SIZE is None or str(BATCH_SIZE).strip().lower() in {"", "auto"})
    explicit_workers = not (NUM_WORKERS is None or str(NUM_WORKERS).strip().lower() in {"", "auto"})
    # There is no preflight calibration: auto batch comes from the existing
    # closed-form heuristic, workers start at TRAIN_DEFAULT_WORKERS, and real
    # training begins immediately. Retries adjust only an automatic knob and
    # are triggered by clear evidence: OOM/GPU fallback, RAM or DataLoader
    # pressure, VRAM headroom, or a sustained throughput collapse with
    # confirmed WDDM spill and VRAM pressure. Explicit BATCH_SIZE and
    # NUM_WORKERS values are never changed automatically.
    #
    # Every successful automatic adjustment follows one recovery rule: when
    # ACCEPT_RESUME is enabled and a valid checkpoint exists after the failed
    # attempt, keep the run directory and resume it with the new batch/workers.
    # Only an explicitly fresh recovery may discard the partial directory.
    emergency_state = _load_emergency_recovery_state(emergency_recovery_path)
    if emergency_state and explicit_batch:
        _clear_emergency_recovery_state(emergency_recovery_path)
        emergency_state = {}
    supervised_batch = os.environ.get(_EMERGENCY_BATCH_ENV)
    supervised_workers = os.environ.get(_EMERGENCY_WORKERS_ENV)
    override_batch: int | None = (
        int(supervised_batch)
        if supervised_batch is not None
        else int(emergency_state["batch"])
        if emergency_state
        else None
    )
    override_workers: int | None = (
        int(supervised_workers)
        if supervised_workers is not None and not explicit_workers
        else int(emergency_state["workers"])
        if emergency_state and not explicit_workers
        else None
    )
    # VRAM-headroom batch increases are a one-time opportunity for the whole
    # run, and permanently disabled the moment batch has ever been lowered
    # (OOM/GPU-fallback/throughput collapse) -- otherwise a lower/raise pair
    # could oscillate indefinitely around the same unsafe batch. Once
    # locked, _InitialLoadCheck stops offering "raise_batch" on any later
    # restart attempt (see allow_headroom_raise).
    vram_headroom_locked = bool(emergency_state or supervised_batch)

    if emergency_state or supervised_batch:
        print(
            "[RECOVERY] loading emergency lower-batch state: "
            f"batch={override_batch}, workers={override_workers}"
        )

    print(f"[LR] direction_training lr0={format_lr0_for_name(lr0)} lrf={format_lrf_for_name(lrf)}")
    print(f"[SEED] direction_training seed={random_seed}")

    for attempt in itertools.count(1):
        resume_checkpoint, resume_epoch, already_complete = _find_resume_checkpoint(weights_dir, int(EPOCHS))
        if already_complete:
            completed_epoch = (resume_epoch + 1) if resume_epoch is not None else "unknown"
            print(
                f"Found existing checkpoint {completed_epoch} "
                f"(epoch{completed_epoch}.pt; YOLO metadata epoch={resume_epoch}); "
                f"target EPOCHS={int(EPOCHS)} is already complete."
            )
            if not os.path.isfile(last_path):
                raise FileNotFoundError(
                    f"Completed training checkpoint exists, but last.pt was not found: {last_path}"
                )
            _clear_emergency_recovery_state(emergency_recovery_path)
            return last_path

        if resume_checkpoint:
            if resume_epoch is None:
                raise RuntimeError(f"Selected resume checkpoint has unknown epoch: {resume_checkpoint}")
            _reconcile_results_csv_for_resume(results_csv, resume_checkpoint, resume_epoch)

            # Recovery overrides take priority over checkpoint train_args.
            # EpochsOBBTrainer.check_resume restores workers after upstream
            # check_resume loads the checkpoint arguments, before Ultralytics
            # constructs either DataLoader. Upstream already honors batch and
            # recomputes its batch-dependent optimizer settings.
            ckpt_train_args = _checkpoint_train_args(resume_checkpoint)
            ckpt_batch = ckpt_train_args.get("batch")
            ckpt_workers = ckpt_train_args.get("workers")
            if override_batch is not None:
                batch_size = override_batch
            else:
                batch_size = int(ckpt_batch) if ckpt_batch else resolve_batch_size(
                    BATCH_SIZE, int(TRAIN_IMG_SIZE), device, mode="train",
                    labels_dir=train_labels_dir, task="obb",
                )
                if ckpt_batch and explicit_batch and int(BATCH_SIZE) != batch_size:
                    print(
                        f"[WARN] Configured BATCH_SIZE={int(BATCH_SIZE)} differs from this run's "
                        f"original batch={batch_size}; resuming with the original batch to avoid "
                        "mixing batch sizes within one run."
                    )
            if override_workers is not None:
                dl_workers = override_workers
            else:
                dl_workers = int(ckpt_workers) if ckpt_workers is not None else _resolve_num_workers(
                    NUM_WORKERS, "dataloader", image_size=int(TRAIN_IMG_SIZE), device=device,
                )
            recovery_override = override_batch is not None or override_workers is not None
            source = "recovery overrides" if recovery_override else "checkpoint train_args"
            print(
                f"[INFO] Resuming with batch={batch_size} workers={dl_workers} "
                f"(effective={effective_dataloader_workers(device, dl_workers)}, {source})."
            )
            _configure_cpu_runtime(device, dl_workers)

            model = YOLO(resume_checkpoint)
            completed_epoch = resume_epoch + 1
            first_resumed_epoch = resume_epoch + 2
            expected_lr_factor = _linear_lr_factor(resume_epoch + 1, int(EPOCHS), float(lrf))
            expected_lr = float(lr0) * expected_lr_factor
            print(f"[INFO] Selected resume checkpoint: {resume_checkpoint}")
            print(f"[INFO] Resuming after completed epoch {completed_epoch}")
            print(f"[INFO] First resumed epoch: {first_resumed_epoch}")
            print(f"[INFO] Target completed epoch: {int(EPOCHS)}")
            print(f"[INFO] Resume lr0: {_format_resume_float(lr0)}")
            print(f"[INFO] Resume lrf: {_format_resume_float(lrf)}")
            print(f"[INFO] Expected first resumed LR factor: {_format_resume_float(expected_lr_factor)}")
            print(f"[INFO] Expected first resumed LR: {_format_resume_float(expected_lr)}")
            print(
                f"[INFO] Resuming after checkpoint {completed_epoch} "
                f"(epoch{completed_epoch}.pt; YOLO metadata epoch={resume_epoch}) "
                f"to epoch {int(EPOCHS)} "
                f"(batch={batch_size}, workers={dl_workers})..."
            )
        else:
            batch_size = (
                override_batch if override_batch is not None
                else max(1, int(BATCH_SIZE)) if explicit_batch
                else auto_batch_size(
                    int(TRAIN_IMG_SIZE), device, mode="train",
                    labels_dir=train_labels_dir, task="obb",
                )
            )
            dl_workers = (
                override_workers if override_workers is not None
                else max(0, int(NUM_WORKERS)) if explicit_workers
                else TRAIN_DEFAULT_WORKERS
            )
            _configure_cpu_runtime(device, dl_workers)
            model = YOLO(resolve_model_name(PRETRAINED_MODEL))
            print(
                f"[INFO] Starting training (batch={batch_size}, workers={dl_workers}, "
                f"effective_workers={effective_dataloader_workers(device, dl_workers)}, "
                f"device={device})..."
            )

        base_kwargs = dict(
            data=data_yaml,                       # default: None
            epochs=int(EPOCHS),                  # default: 50
            imgsz=int(TRAIN_IMG_SIZE),           # default: 640
            project=PROJECT_DIR,                  # default: None
            name=EXPERIMENT_NAME,                 # default: None
            device=device,                        # default: None
            workers=dl_workers,                   # default: 8
            exist_ok=True,                        # default: False
            verbose=False,                        # default: True
            # AMADEUS retains periodic 1-based epochN.pt files for interruption
            # recovery. Their embedded Ultralytics epoch metadata remains N-1;
            # model selection also supports best/last explicitly.
            save_period=int(SAVE_PERIOD),         # default: 5
            seed=int(random_seed),
            deterministic=True,
            amp=runtime_for(device).capabilities.amp,
            cache=False,                           # explicit: avoid RAM-cached datasets
            flipud=0.0,                           # default: 0.0
            fliplr=0.0,                           # default: 0.5
            # Both final and refine detectors call this function and therefore use
            # the same Mosaic/SGD training policy.
            mosaic=1.0,                           # default: 1.0
            close_mosaic=0,                       # never disable Mosaic
            cos_lr=False,                         # use linear LR schedule
            lr0=float(lr0),
            lrf=float(lrf),
            optimizer="SGD",
        )

        def _do_train():
            train_kwargs = {**base_kwargs, "batch": batch_size}
            watchdog_state_path = os.environ.get(_WATCHDOG_STATE_ENV)
            watchdog_state = (
                _WatchdogSharedState(watchdog_state_path)
                if watchdog_state_path
                else None
            )
            device_index = _parse_cuda_device_index(device)
            heartbeat_device_index = -1 if device_index is None else device_index

            # Started before either monitor is built: _InitialLoadCheck takes
            # its pre-training WDDM baseline from the sampler's first sample.
            sampler = _DeviceSampler(device)
            sampler.start()

            def _emergency_abort(reason: str) -> None:
                """Called by the watchdog thread while a CUDA batch is stuck."""
                if os.environ.get(_TRAIN_WORKER_ENV) != "1":
                    print(
                        "[WARN] The in-flight batch watchdog detected an actionable "
                        "collapse, but this imported main() call is not running under "
                        "the CLI supervisor; recovery will occur when CUDA returns.",
                        flush=True,
                    )
                    return
                if explicit_batch or batch_size <= 1:
                    print(f"[RECOVERY] cause={reason}", flush=True)
                    print("[RECOVERY] action=lower_batch", flush=True)
                    limit = "fixed BATCH_SIZE" if explicit_batch else "batch=1 recovery floor"
                    print(f"[ERROR] Emergency recovery cannot lower {limit}.", flush=True)
                    os._exit(_EMERGENCY_FATAL_EXIT_CODE)

                next_batch = next_batch_candidate_soft_down(batch_size)
                _write_emergency_recovery_state(
                    emergency_recovery_path,
                    batch=next_batch,
                    workers=dl_workers,
                )
                print(f"[RECOVERY] cause={reason}", flush=True)
                print("[RECOVERY] action=lower_batch", flush=True)
                print(f"[RECOVERY] batch={batch_size} -> {next_batch}", flush=True)
                print(f"[RECOVERY] workers={dl_workers} -> {dl_workers}", flush=True)
                print(
                    "[RECOVERY] emergency=terminating stalled CUDA child; the supervisor "
                    "will resume from the latest valid checkpoint",
                    flush=True,
                )
                os._exit(_EMERGENCY_RECOVERY_EXIT_CODE)

            monitor = _PerformanceMonitor(
                device=device,
                sampler=sampler,
                batch_size=batch_size,
                image_size=int(TRAIN_IMG_SIZE),
                workers=dl_workers,
                emergency_abort=_emergency_abort,
            )
            initial_check = _InitialLoadCheck(
                device=device,
                sampler=sampler,
                batch_size=batch_size,
                workers=dl_workers,
                allow_headroom_raise=not explicit_batch and not vram_headroom_locked,
            )
            gpu_fallback_watcher = _GpuFallbackWatcher()
            gpu_fallback_floor_warned = False
            ultralytics_logger = logging.getLogger("ultralytics")

            def _apply_initial_check_result() -> None:
                verdict = initial_check.verdict()
                if verdict is not None:
                    action, reason = verdict
                    raise _TrainingAborted(action, reason)
                official_throughput = initial_check.take_official_throughput()
                if official_throughput is not None:
                    print(
                        "[INFO] Baseline throughput set from the starting epoch's own full "
                        f"measured rate: {official_throughput:.1f} img/s"
                    )
                    monitor.baseline_throughput = official_throughput
                    monitor.provisional_throughput = None

            def _on_batch_start(trainer):
                inflight_since = monitor.record_batch_start(epoch=trainer.epoch)
                if watchdog_state is not None:
                    watchdog_state.publish(
                        inflight_since=inflight_since,
                        reference_throughput=(
                            monitor.baseline_throughput
                            or monitor.provisional_throughput
                            or 0.0
                        ),
                        batch=batch_size,
                        recovery_batch=batch_size,
                        workers=dl_workers,
                        device_index=heartbeat_device_index,
                        active=True,
                        can_lower_batch=not explicit_batch and batch_size > 1,
                    )

            def _on_batch_end(trainer):
                nonlocal gpu_fallback_floor_warned
                if watchdog_state is not None:
                    watchdog_state.publish(
                        inflight_since=0.0,
                        reference_throughput=0.0,
                        batch=batch_size,
                        recovery_batch=batch_size,
                        workers=dl_workers,
                        device_index=heartbeat_device_index,
                        active=False,
                        can_lower_batch=not explicit_batch and batch_size > 1,
                    )
                # Always-on, whole run: a silent CUDA-OOM-to-CPU fallback
                # never raises, so it's checked on every batch regardless of
                # epoch (see _GpuFallbackWatcher).
                fallback = _take_gpu_fallback_action(gpu_fallback_watcher, batch_size)
                if fallback is not None:
                    fallback_action, fallback_message = fallback
                    if fallback_action == "lower_batch":
                        raise _TrainingAborted(
                            "lower_batch", f"GPU->CPU fallback: {fallback_message}"
                        )
                    if not gpu_fallback_floor_warned:
                        print(
                            "[WARN] Ultralytics used its silent CUDA-OOM -> CPU fallback, "
                            "but automatic recovery is already at batch=1; continuing "
                            "training with batch=1.",
                            flush=True,
                        )
                        gpu_fallback_floor_warned = True
                # This run's starting epoch only (index 0 for a fresh run,
                # the resume epoch for a resumed one -- see
                # _InitialLoadCheck): RAM pressure can abort at any point;
                # WDDM growth is only observed and warned about, and VRAM
                # headroom is decided once at the epoch's own end. Completed
                # batches also provide a provisional rate only for detecting
                # a later in-flight batch that stalls with confirmed spill.
                initial_check.record_batch(epoch=trainer.epoch)
                if monitor.baseline_throughput is None:
                    provisional_throughput = initial_check.provisional_throughput()
                    if provisional_throughput is not None:
                        monitor.provisional_throughput = provisional_throughput
                _apply_initial_check_result()
                # Inert until baseline_throughput is set above (see
                # _PerformanceMonitor.record_batch), i.e. for the whole
                # starting epoch. Every later throughput-collapse candidate
                # is actionable only when the cached snapshot also confirms
                # WDDM growth and VRAM pressure.
                monitor.record_batch(epoch=trainer.epoch)
                pressure = monitor.pressure_verdict()
                if pressure is not None:
                    raise _TrainingAborted(*pressure)
                if monitor.collapsed():
                    raise _TrainingAborted("lower_batch", monitor.collapsed_reason())

            def _on_fit_epoch_end(trainer):
                # Runs after validation and checkpoint save, but before the
                # next epoch. Finalize the baseline here so the in-flight
                # watchdog can protect the next epoch's very first batch.
                if trainer.epoch + 1 >= trainer.epochs:
                    return
                initial_check.finish_epoch(epoch=trainer.epoch)
                _apply_initial_check_result()

            def _validation_batch_size(validator) -> int:
                dataloader = getattr(validator, "dataloader", None)
                return max(1, int(getattr(dataloader, "batch_size", batch_size) or batch_size))

            def _on_val_start(validator):
                monitor.start_validation(_validation_batch_size(validator))

            def _on_val_batch_start(validator):
                val_batch_size = _validation_batch_size(validator)
                inflight_since = monitor.record_validation_batch_start(val_batch_size)
                if watchdog_state is not None:
                    watchdog_state.publish(
                        inflight_since=inflight_since,
                        reference_throughput=(
                            monitor.baseline_throughput
                            or monitor.provisional_throughput
                            or 0.0
                        ),
                        batch=val_batch_size,
                        recovery_batch=batch_size,
                        workers=dl_workers,
                        device_index=heartbeat_device_index,
                        active=True,
                        can_lower_batch=not explicit_batch and batch_size > 1,
                    )

            def _on_val_batch_end(validator):
                if watchdog_state is not None:
                    watchdog_state.publish(
                        inflight_since=0.0,
                        reference_throughput=0.0,
                        batch=_validation_batch_size(validator),
                        recovery_batch=batch_size,
                        workers=dl_workers,
                        device_index=heartbeat_device_index,
                        active=False,
                        can_lower_batch=not explicit_batch and batch_size > 1,
                    )
                monitor.record_validation_batch_end()
                pressure = monitor.pressure_verdict()
                if pressure is not None:
                    raise _TrainingAborted(*pressure)
                if monitor.collapsed():
                    raise _TrainingAborted("lower_batch", monitor.collapsed_reason())

            def _on_train_start(trainer):
                _enable_cudnn_autotuner(device)

            monitor.start()
            ultralytics_logger.addHandler(gpu_fallback_watcher)
            try:
                if resume_checkpoint:
                    train_kwargs["resume"] = resume_checkpoint
                model.add_callback("on_train_start", _on_train_start)
                model.add_callback("on_train_batch_start", _on_batch_start)
                model.add_callback("on_train_batch_end", _on_batch_end)
                model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
                model.add_callback("on_val_start", _on_val_start)
                model.add_callback("on_val_batch_start", _on_val_batch_start)
                model.add_callback("on_val_batch_end", _on_val_batch_end)
                model.train(trainer=trainer_cls, **train_kwargs)
            except _TrainingAborted as exc:
                raise _BatchTooLargeError(str(exc), action=exc.action) from exc
            except Exception as exc:
                is_oom, backend = _is_oom_error(exc)
                if is_oom:
                    raise _BatchTooLargeError(
                        f"{backend} memory exhaustion mid-training at batch={batch_size}.",
                        action="lower_batch",
                    ) from exc
                if _is_dataloader_failure(exc):
                    raise _BatchTooLargeError(
                        f"DataLoader failure mid-training: {exc}", action="lower_workers",
                    ) from exc
                raise
            finally:
                monitor.stop()
                sampler.stop()
                ultralytics_logger.removeHandler(gpu_fallback_watcher)
                if watchdog_state is not None:
                    watchdog_state.publish(
                        inflight_since=0.0,
                        reference_throughput=0.0,
                        batch=batch_size,
                        recovery_batch=batch_size,
                        workers=dl_workers,
                        device_index=heartbeat_device_index,
                        active=False,
                        can_lower_batch=not explicit_batch and batch_size > 1,
                    )
                    watchdog_state.close()

        try:
            _do_train()
        except _BatchTooLargeError as exc:
            action = exc.action
            if action == "lower_workers":
                if explicit_workers:
                    raise RuntimeError(f"{exc} Training cannot proceed with a fixed NUM_WORKERS.") from exc
                if dl_workers <= 0:
                    raise RuntimeError(
                        f"{exc} Cannot reduce dataloader workers further: workers={dl_workers} is "
                        "already at the automatic recovery floor."
                    ) from exc
                override_workers = next_worker_candidate(dl_workers)
                override_batch = batch_size
                print(
                    f"[WARN] {exc} Reducing configured workers={dl_workers} -> {override_workers}; "
                    f"effective workers will be {effective_dataloader_workers(device, override_workers)}; "
                    f"keeping batch={batch_size} (attempt {attempt})."
                )
            elif action == "raise_batch":
                if explicit_batch:
                    raise RuntimeError(f"{exc} Training cannot change a fixed BATCH_SIZE.") from exc
                # A single conservative ~1.25x step, not a closed-form
                # estimate toward some target utilization -- VRAM use isn't
                # reliably linear in batch size, so extrapolating from one
                # data point risks overshooting straight into the next OOM.
                vram_headroom_locked = True
                override_batch = next_batch_candidate(batch_size, "up")
                override_workers = dl_workers
                print(
                    f"[INFO] {exc} Increasing batch={batch_size} -> {override_batch} "
                    f"to use the available VRAM headroom (attempt {attempt})."
                )
            elif action == "lower_batch":
                if explicit_batch or batch_size <= 1:
                    verb = "Training cannot proceed with a fixed batch size" if explicit_batch else (
                        f"Giving up after {attempt} recovery attempt(s) at progressively smaller batch sizes "
                        "(batch=1 itself failed)"
                    )
                    raise RuntimeError(f"{exc} {verb}.") from exc
                # Once a batch has had to come down, a lower/raise pair
                # oscillating around the same unsafe batch is exactly what
                # this is meant to prevent -- permanently forbid raising it
                # again via VRAM headroom for the rest of this run.
                vram_headroom_locked = True
                override_batch = next_batch_candidate_soft_down(batch_size)
                override_workers = dl_workers
                print(
                    f"[WARN] {exc} Reducing batch={batch_size} -> {override_batch} "
                    f"(attempt {attempt})."
                )
            else:
                raise RuntimeError(f"Unknown automatic recovery action: {action!r}") from exc

            # Check again after the failed attempt because it may have written
            # a newer periodic checkpoint than the one selected at attempt
            # start. All adjustment causes share this same resume decision.
            next_checkpoint: str | None = None
            resume_recovery = False
            recovery_reason: str | None = None
            if ACCEPT_RESUME:
                next_checkpoint, _next_epoch, _complete = _find_resume_checkpoint(
                    weights_dir, int(EPOCHS)
                )
                resume_recovery = next_checkpoint is not None
                if not resume_recovery:
                    recovery_reason = "no checkpoint available"
            else:
                recovery_reason = "ACCEPT_RESUME is disabled"

            checkpoint_log = next_checkpoint
            if checkpoint_log is None and not ACCEPT_RESUME:
                checkpoint_log = resume_checkpoint or (
                    os.path.abspath(last_path) if os.path.isfile(last_path) else None
                )
            cause_log = " ".join(str(exc).split())
            print(f"[RECOVERY] cause={cause_log}")
            print(f"[RECOVERY] action={action}")
            print(f"[RECOVERY] batch={batch_size} -> {override_batch}")
            print(f"[RECOVERY] workers={dl_workers} -> {override_workers}")
            print(f"[RECOVERY] accept_resume={bool(ACCEPT_RESUME)}")
            print(f"[RECOVERY] checkpoint={checkpoint_log or 'none'}")
            print(f"[RECOVERY] resume={resume_recovery}")
            if recovery_reason is not None:
                print(f"[RECOVERY] reason={recovery_reason}")

            if resume_recovery:
                print(
                    "[INFO] Keeping checkpoint, results.csv, and training history; "
                    f"the next attempt will resume with adjusted settings (attempt {attempt})."
                )
            else:
                _reset_run_directory(PROJECT_DIR, EXPERIMENT_NAME)
            del model
            gc.collect()
            empty_accelerator_cache(device)
            continue

        break

    if not os.path.isfile(last_path):
        raise FileNotFoundError(f"Training finished but last.pt was not found: {last_path}")

    _clear_emergency_recovery_state(emergency_recovery_path)
    print("Training complete!")
    return last_path


def _main_from_config(cfg_path: str) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = resolve_config_paths(cfg)

    if cfg.get("skip_training"):
        print("skip_training is True; exiting.")
        return

    session_path = cfg["SESSION_PATH"]
    training = cfg.get("training", {}) or {}
    pretrained_model = training["PRETRAINED_MODEL"]
    epochs = resolve_epoch_count(training.get("EPOCHS", 50))
    lr0 = require_single_lr0(training.get("LR0", DEFAULT_LR0))
    lrf = require_single_lrf(training.get("LRF", DEFAULT_LRF))
    training_image_size = int(cfg["TRAIN_IMG_SIZE"])
    batch_size = training["BATCH_SIZE"]
    device = training.get("DEVICE", "auto")
    num_workers = cfg.get("NUM_WORKERS", "auto")
    class_names = resolve_class_names(cfg)
    num_classes = len(class_names)
    dataset_dir = resolve_yolo_dataset_dir(cfg, session_path)
    model_name_noext = resolve_model_name(pretrained_model).split(".")[0]
    project_dir = training_project_dir(session_path, model_name_noext)
    experiment_name = resolve_existing_experiment_dir_name(
        session_path,
        model_name_noext,
        experiment_dir_name_from_cfg(cfg),
        stages=("training",),
        warn_fn=print,
    )
    save_period = resolve_save_period(training.get("SAVE_PERIOD", 5))
    random_seed = normalize_seed(cfg.get("RANDOM_SEED", 0))
    accept_resume = bool(training.get("ACCEPT_RESUME", True))
    os.makedirs(project_dir, exist_ok=True)

    main(
        PRETRAINED_MODEL=pretrained_model,
        EPOCHS=epochs,
        LR0=lr0,
        LRF=lrf,
        TRAIN_IMG_SIZE=training_image_size,
        BATCH_SIZE=batch_size,
        DEVICE=device,
        DATASET_DIR=dataset_dir,
        NUM_CLASSES=num_classes,
        CLASS_NAMES=class_names,
        PROJECT_DIR=project_dir,
        EXPERIMENT_NAME=experiment_name,
        NUM_WORKERS=num_workers,
        SAVE_PERIOD=save_period,
        RANDOM_SEED=random_seed,
        ACCEPT_RESUME=accept_resume,
    )


def _run_supervised_cli(args: list[str]) -> int:
    """Run CUDA training in a replaceable child process.

    A Python thread cannot safely interrupt a host thread blocked inside a
    sysmem-backed CUDA kernel. Keeping this small supervisor free of CUDA
    state lets the watchdog terminate only the stuck child with os._exit();
    the child persists its next batch first, so rerunning the same command
    resumes from the latest checkpoint with that lower batch. At batch=1 the
    supervisor only warns about WDDM-related collapse and leaves the child
    running because no lower viable batch exists.
    """
    with open(args[0], encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    monitor_device = (config.get("training") or {}).get("DEVICE", "auto")
    monitor_runtime = runtime_for(monitor_device)
    use_wddm = monitor_runtime.wddm
    child_env = os.environ.copy()
    child_env[_TRAIN_WORKER_ENV] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONFAULTHANDLER"] = "1"
    command = [sys.executable, "-u", os.path.abspath(__file__), *args]
    shared_state = _WatchdogSharedState.create()
    child_env[_WATCHDOG_STATE_ENV] = shared_state.path
    try:
        while True:
            shared_state.publish(
                inflight_since=0.0,
                reference_throughput=0.0,
                batch=0,
                recovery_batch=0,
                workers=0,
                device_index=-1,
                active=False,
                can_lower_batch=False,
            )
            child = subprocess.Popen(command, env=child_env)
            baseline_non_local = (_query_wddm_non_local_usage(child.pid, device=monitor_device)
                                  if use_wddm else None)
            peak_vram_frac: float | None = None
            last_supervisor_warning = ""
            next_device_sample = time.monotonic()
            emergency_restart = False

            while True:
                return_code = child.poll()
                if return_code is not None:
                    break

                now = time.monotonic()
                state = shared_state.read()
                if state is None or now < next_device_sample:
                    time.sleep(1.0)
                    continue
                next_device_sample = now + TRAIN_SAMPLER_WDDM_INTERVAL_SECONDS

                current_non_local = (_query_wddm_non_local_usage(child.pid, device=monitor_device)
                                  if use_wddm else None)
                if baseline_non_local is None and current_non_local is not None:
                    baseline_non_local = current_non_local
                vram_used = vram_total = vram_frac = None
                if state["device_index"] >= 0 and monitor_runtime.capabilities.telemetry == "nvidia":
                    _gpu_util, vram_used, vram_total = _query_nvidia_smi(
                        str(state["device_index"])
                    )
                    vram_frac = (
                        vram_used / vram_total
                        if vram_used is not None and vram_total
                        else None
                    )
                    if vram_frac is not None:
                        peak_vram_frac = (
                            vram_frac
                            if peak_vram_frac is None
                            else max(peak_vram_frac, vram_frac)
                        )
                if (
                    not state["active"]
                    or state["reference_throughput"] <= 0
                    or state["inflight_since"] <= 0
                ):
                    time.sleep(1.0)
                    continue

                stalled_for = now - state["inflight_since"]
                collapse_rate = state["reference_throughput"] * TRAIN_COLLAPSE_RATIO
                if collapse_rate <= 0:
                    time.sleep(1.0)
                    continue
                evidence_seconds = max(
                    TRAIN_COLLAPSE_WINDOW_SECONDS,
                    state["batch"] / collapse_rate,
                )
                if stalled_for < evidence_seconds:
                    time.sleep(1.0)
                    continue

                wddm_growth = (
                    current_non_local - baseline_non_local
                    if current_non_local is not None and baseline_non_local is not None
                    else None
                )
                pressure_vram_frac = _vram_pressure_fraction(
                    vram_used=vram_used,
                    vram_total=vram_total,
                    current_frac=vram_frac,
                    peak_frac=peak_vram_frac,
                    wddm_growth=wddm_growth,
                )
                warning_key = ""
                warning_message = ""
                if wddm_growth is None:
                    warning_key = "wddm-unavailable"
                    warning_message = (
                        f"[WARN] monitored batch throughput collapsed for {stalled_for:.0f}s, "
                        "but WDDM non-local memory is unavailable, so no WDDM spill is "
                        f"confirmed; keeping batch={state['recovery_batch']}."
                    )
                elif wddm_growth < TRAIN_WDDM_GROWTH_MIN_GIB:
                    warning_key = "wddm-insufficient"
                    warning_message = (
                        f"[WARN] monitored batch throughput collapsed for {stalled_for:.0f}s, "
                        f"but WDDM non-local growth is only {wddm_growth:.2f}GiB, so no "
                        f"WDDM spill is confirmed; keeping batch={state['recovery_batch']}."
                    )
                elif (
                    pressure_vram_frac is None
                    or pressure_vram_frac < TRAIN_VRAM_PRESSURE_RATIO
                ):
                    warning_key = "vram-insufficient"
                    pressure_text = (
                        "unavailable"
                        if pressure_vram_frac is None
                        else f"{pressure_vram_frac:.1%}"
                    )
                    warning_message = (
                        f"[WARN] monitored batch throughput collapsed for {stalled_for:.0f}s "
                        f"with WDDM growth {wddm_growth:.2f}GiB, but VRAM pressure is "
                        f"{pressure_text}; keeping batch={state['recovery_batch']}."
                    )
                if warning_key:
                    if warning_key != last_supervisor_warning:
                        print(warning_message, flush=True)
                        last_supervisor_warning = warning_key
                    time.sleep(1.0)
                    continue

                if state["recovery_batch"] <= 1:
                    warning_key = "batch-floor"
                    if warning_key != last_supervisor_warning:
                        print(
                            "[WARN] monitored batch throughput collapsed with confirmed "
                            "WDDM spill and VRAM pressure, but automatic recovery is "
                            "already at batch=1; continuing training with batch=1.",
                            flush=True,
                        )
                        last_supervisor_warning = warning_key
                    time.sleep(1.0)
                    continue

                upper_bound = state["batch"] / stalled_for
                reason = (
                    "throughput collapse with confirmed WDDM spill and VRAM pressure: "
                    f"no monitored batch completed for {stalled_for:.0f}s; throughput is "
                    f"at most {upper_bound:.2f} img/s (reference "
                    f"{state['reference_throughput']:.1f} img/s); WDDM non-local grew by "
                    f"{wddm_growth:.2f}GiB (from {baseline_non_local:.2f}GiB to "
                    f"{current_non_local:.2f}GiB); VRAM usage is "
                    f"{_fmt_pct(vram_frac * 100 if vram_frac is not None else None)} "
                    f"(pressure-equivalent {pressure_vram_frac:.1%}) "
                    f"({_fmt_gib(vram_used)}/{_fmt_gib(vram_total)})"
                )
                print(f"[RECOVERY] cause={reason}", flush=True)
                print("[RECOVERY] action=lower_batch", flush=True)

                child.terminate()
                try:
                    child.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()

                if not state["can_lower_batch"]:
                    print(
                        "[ERROR] Emergency recovery cannot lower the fixed batch; "
                        "training was stopped.",
                        flush=True,
                    )
                    return _EMERGENCY_FATAL_EXIT_CODE

                next_batch = next_batch_candidate_soft_down(state["recovery_batch"])
                child_env[_EMERGENCY_BATCH_ENV] = str(next_batch)
                child_env[_EMERGENCY_WORKERS_ENV] = str(state["workers"])
                print(
                    f"[RECOVERY] batch={state['recovery_batch']} -> {next_batch}",
                    flush=True,
                )
                print(
                    f"[RECOVERY] workers={state['workers']} -> {state['workers']}",
                    flush=True,
                )
                print(
                    "[RECOVERY] supervisor=terminated stalled CUDA child; restarting "
                    "from the latest valid checkpoint with the lower batch",
                    flush=True,
                )
                emergency_restart = True
                break

            if emergency_restart:
                continue
            print(f"[INFO] {describe_child_return_code(return_code)}", flush=True)
            if return_code == _EMERGENCY_RECOVERY_EXIT_CODE:
                print(
                    "[RECOVERY] supervisor=restarting training with persisted lower batch",
                    flush=True,
                )
                continue
            return return_code
    finally:
        shared_state.close()


if __name__ == "__main__":
    from without_direction_estimation import prepare_config
    if len(sys.argv) > 1:
        sys.argv[1] = prepare_config(sys.argv[1])
    if len(sys.argv) < 2:
        raise SystemExit("Usage: obb_detector_training.py CONFIG_PATH")
    if os.environ.get(_TRAIN_WORKER_ENV) == "1":
        _main_from_config(sys.argv[1])
    else:
        raise SystemExit(_run_supervised_cli(sys.argv[1:]))

