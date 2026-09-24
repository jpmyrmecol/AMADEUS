# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Run the configured tracking stages in order and record their progress."""

import sys
from pathlib import Path
_MAIN_DIR = Path(__file__).resolve().parent.parent
for _path in (str(_MAIN_DIR), str(_MAIN_DIR.parent)):
    if _path not in sys.path:
        sys.path.append(_path)


import os
import sys
import csv
import codecs
import copy
import subprocess
import tempfile
import traceback
import platform
import importlib.metadata
import yaml
from datetime import datetime
from pathlib import Path

from experiment_utils import (
    DEFAULT_LR0,
    DEFAULT_LRF,
    experiment_dir_name_from_cfg,
    format_lr0_for_name,
    format_lrf_for_name,
    parse_lr0_values,
    parse_lrf_values,
)
from path_utils import resolve_config_paths
from random_utils import normalize_seed
from main.compact_log import CompactChildOutput, GuiProgressPassthrough

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gui.project_paths import MAIN_DIR as MAIN_PATH, PROJECT_ROOT, with_pythonpath

CURRENT_DIR = str(Path(__file__).resolve().parent)

INITIAL_TRACKING_SCRIPT = os.path.join(CURRENT_DIR, "initial_tracking.py")
DIRECTION_CLASS_ASSIGNMENT_SCRIPT = os.path.join(CURRENT_DIR, "direction_class_assignment.py")
DIRECTION_CLASS_FILTERING_SCRIPT = os.path.join(CURRENT_DIR, "direction_class_filtering.py")
INTERACTION_IMAGE_SYNTHESIS_SCRIPT = os.path.join(CURRENT_DIR, "interaction_image_synthesis.py")
INTERACTION_IMAGE_SYNTHESIS_CLUSTERED_SCRIPT = os.path.join(CURRENT_DIR, "interaction_image_synthesis_clustered.py")
CROP_IMAGES_SCRIPT = os.path.join(CURRENT_DIR, "crop_images.py")
CREATE_DATASET_SCRIPT = os.path.join(CURRENT_DIR, "create_dataset.py")
OBB_DETECTOR_TRAINING_SCRIPT = os.path.join(CURRENT_DIR, "obb_detector_training.py")
OBB_DETECTION_SCRIPT = os.path.join(CURRENT_DIR, "obb_detection.py")
MULTI_STAGED_ASSOCIATION_SCRIPT = os.path.join(CURRENT_DIR, "multi_staged_association.py")
REFINEMENT_SCRIPT = os.path.join(CURRENT_DIR, "refinement.py")
CREATE_TRACKING_VIDEO_SCRIPT = os.path.join(CURRENT_DIR, "create_video.py")
APPLY_CLASS_LABEL_FILTERING_SCRIPT = os.path.join(CURRENT_DIR, "apply_class_label_filtering.py")

_TIMING_CSV_PATH = ""
_MASTER_SEED = 0
_ARCHIVE_LOG = None
_ARCHIVE_LOG_PATH = ""
_CONSOLE_STDOUT = None
_CONSOLE_STDERR = None


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping.")
    return resolve_config_paths(cfg)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"
        self.errors = getattr(streams[0], "errors", "replace") if streams else "replace"

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                safe = data.encode(encoding, errors="replace").decode(encoding, errors="replace")
                stream.write(safe)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"
    except Exception as exc:
        return f"unavailable ({exc})"


def _write_archive(text: str) -> None:
    if _ARCHIVE_LOG is None or not text:
        return
    _ARCHIVE_LOG.write(text)
    _ARCHIVE_LOG.flush()


def _begin_logging(config_path: str) -> str:
    global _ARCHIVE_LOG, _ARCHIVE_LOG_PATH, _CONSOLE_STDOUT, _CONSOLE_STDERR
    cfg_abs = os.path.abspath(config_path)
    cfg_dir = os.path.dirname(cfg_abs) or os.getcwd()
    log_path = os.path.join(cfg_dir, "log.txt")
    if _ARCHIVE_LOG is not None and _ARCHIVE_LOG_PATH == log_path:
        return log_path
    if _ARCHIVE_LOG is not None:
        _finish_logging("replaced")
    if _CONSOLE_STDOUT is None:
        _CONSOLE_STDOUT = sys.stdout
        _CONSOLE_STDERR = sys.stderr
    _ARCHIVE_LOG_PATH = log_path
    _ARCHIVE_LOG = open(log_path, "a", encoding="utf-8", buffering=1)
    separator = f"=== AMADEUS RUN START {_iso_timestamp()} PID={os.getpid()} ===\n"
    _write_archive(separator)
    header = [
        f"python_executable={sys.executable}",
        f"python_version={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"cwd={os.getcwd()}",
        f"config_path={cfg_abs}",
        f"argv={sys.argv!r}",
        f"torch_version={_package_version('torch')}",
        f"torchvision_version={_package_version('torchvision')}",
        f"ultralytics_version={_package_version('ultralytics')}",
    ]
    _write_archive("[RUN HEADER] " + "\n[RUN HEADER] ".join(header) + "\n")
    sys.stdout = TeeStream(_CONSOLE_STDOUT, _ARCHIVE_LOG)
    sys.stderr = TeeStream(_CONSOLE_STDERR, _ARCHIVE_LOG)
    print(f"Saving CLI log to: {log_path}", flush=True)
    return log_path


def _finish_logging(status: str) -> None:
    global _ARCHIVE_LOG, _ARCHIVE_LOG_PATH
    if _ARCHIVE_LOG is None:
        return
    _write_archive(f"=== AMADEUS RUN END {_iso_timestamp()} PID={os.getpid()} status={status} ===\n")
    try:
        _ARCHIVE_LOG.flush()
        os.fsync(_ARCHIVE_LOG.fileno())
    except (OSError, ValueError):
        pass
    _ARCHIVE_LOG.close()
    if _CONSOLE_STDOUT is not None:
        sys.stdout = _CONSOLE_STDOUT
    if _CONSOLE_STDERR is not None:
        sys.stderr = _CONSOLE_STDERR
    _ARCHIVE_LOG = None
    _ARCHIVE_LOG_PATH = ""


def _timestamp(dt: datetime | None = None) -> str:
    return (dt or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")


def _iso_timestamp(dt: datetime | None = None) -> str:
    return (dt or datetime.now()).isoformat(timespec="seconds")


def _init_timing_csv(path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "script", "status", "start_time", "end_time", "elapsed_seconds", "elapsed"])


def _append_timing_row(
    *,
    stage: str,
    script: str,
    status: str,
    start: datetime,
    end: datetime,
) -> None:
    if not _TIMING_CSV_PATH:
        return
    elapsed = end - start
    with open(_TIMING_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            stage,
            script,
            status,
            _iso_timestamp(start),
            _iso_timestamp(end),
            f"{elapsed.total_seconds():.3f}",
            _fmt_elapsed(elapsed),
        ])


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    env.setdefault("TQDM_ASCII", " 123456789#")
    env.setdefault("COLUMNS", "120")
    env["PYTHONHASHSEED"] = str(_MASTER_SEED % (2**32))
    with_pythonpath(env, PROJECT_ROOT, MAIN_PATH)

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # Keep progress output responsive.  The child scripts are already invoked
        # with -u, and this prevents the intermediate pipe from adding extra
        # buffering in batch.py.
        bufsize=0,
    )
    assert proc.stdout is not None
    console = _CONSOLE_STDOUT or sys.stdout
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    output_filter = CompactChildOutput(
        training=any(os.path.basename(str(arg)) == "obb_detector_training.py" for arg in cmd)
    )
    gui_progress_protocol = os.environ.get("AMADEUS_GUI_PROGRESS_PROTOCOL", "").strip().lower() in {
        "1", "true", "yes"
    }
    gui_output = GuiProgressPassthrough() if gui_progress_protocol else None
    while True:
        chunk = proc.stdout.read(512)
        if not chunk:
            break
        text = decoder.decode(chunk, final=False)
        if not text:
            continue
        compact = output_filter.feed(text)
        if compact:
            _write_archive(compact)
        visible = gui_output.feed(text) if gui_output is not None else compact
        if visible:
            console.write(visible)
            console.flush()
    tail = decoder.decode(b"", final=True)
    compact_tail = output_filter.feed(tail) + output_filter.flush()
    if compact_tail:
        _write_archive(compact_tail)
    visible_tail = (
        gui_output.feed(tail) + gui_output.flush()
        if gui_output is not None
        else compact_tail
    )
    if visible_tail:
        console.write(visible_tail)
        console.flush()
    summary = output_filter.summary_line()
    if summary:
        _write_archive(summary)
        console.write(summary)
        console.flush()
    rc = proc.wait()
    exit_line = f"[INFO] child exit: {rc}\n"
    _write_archive(exit_line)
    console.write(exit_line)
    console.flush()

    if rc != 0:
        if rc < 0:
            try:
                import signal
                signal_name = signal.Signals(-rc).name
            except (ValueError, ImportError):
                signal_name = "unknown signal"
            signal_line = f"[ERROR] child terminated by signal {-rc} ({signal_name})\n"
            _write_archive(signal_line)
            console.write(signal_line)
            console.flush()
        raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")

def _fmt_elapsed(td) -> str:
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def run_script(script: str, *args: str) -> None:
    name = os.path.splitext(os.path.basename(script))[0].removesuffix("_variable")
    start = datetime.now()
    status = "success"
    print(f"\n[{_timestamp(start)}] START: {name}", flush=True)
    cmd = [sys.executable, "-u", script] + list(args)
    try:
        run(cmd)
    except Exception:
        status = "failed"
        raise
    finally:
        end = datetime.now()
        label = "END" if status == "success" else "FAIL"
        print(f"[{_timestamp(end)}] {label}:   {name}  (elapsed: {_fmt_elapsed(end - start)})", flush=True)
        _append_timing_row(stage=name, script=script, status=status, start=start, end=end)


def skip_stage(stage: str, reason: str = "") -> None:
    now = datetime.now()
    suffix = f" ({reason})" if reason else ""
    print(f"[{_timestamp(now)}] SKIP:  {stage}{suffix}  (elapsed: 00:00:00)", flush=True)
    _append_timing_row(stage=stage, script="", status="skipped", start=now, end=now)


def run_lrf_independent_stages(cfg: dict, cfg_path: str) -> None:
    """Run preprocessing and dataset creation exactly once per batch."""
    if cfg.get("skip_initial_tracking", False):
        skip_stage("initial_tracking", "skip_initial_tracking is True")
    else:
        print("Running initial tracking...")
        run_script(INITIAL_TRACKING_SCRIPT, cfg_path)
        cfg.clear()
        cfg.update(load_config(cfg_path))

    if cfg.get("skip_trajectory_direction_filtering", False):
        skip_stage("direction_class_assignment", "skip_trajectory_direction_filtering is True")
    else:
        print("Creating single animal images (trajectory / direction filtering)...")
        run_script(DIRECTION_CLASS_ASSIGNMENT_SCRIPT, cfg_path)

    if cfg.get("skip_refine_blobs_through_tracking", False):
        skip_stage("direction_class_filtering", "skip_refine_blobs_through_tracking is True")
    else:
        print("Running iterative refine blobs through training...")
        run_script(DIRECTION_CLASS_FILTERING_SCRIPT, cfg_path)

    if cfg.get("skip_paste_blobs_clustered", False):
        skip_stage("interaction_image_synthesis_clustered", "skip_paste_blobs_clustered is True")
    else:
        print("Creating clustered pasted blobs...")
        run_script(INTERACTION_IMAGE_SYNTHESIS_CLUSTERED_SCRIPT, cfg_path)

    if cfg.get("skip_paste_blobs_with_crossing", False):
        skip_stage("interaction_image_synthesis", "skip_paste_blobs_with_crossing is True")
    else:
        print("Creating mixed pasted blobs...")
        run_script(INTERACTION_IMAGE_SYNTHESIS_SCRIPT, cfg_path)

    if cfg.get("skip_creating_direction_dataset", False):
        skip_stage("crop_images", "skip_creating_direction_dataset is True")
    elif cfg.get("skip_cropping", False):
        skip_stage("crop_images", "skip_cropping is True")
    else:
        print("Starting cropping process...")
        run_script(CROP_IMAGES_SCRIPT, cfg_path)

    if cfg.get("skip_creating_direction_dataset", False):
        skip_stage("create_dataset", "skip_creating_direction_dataset is True")
    else:
        print("Creating direction-estimation dataset...")
        run_script(CREATE_DATASET_SCRIPT, cfg_path)


def run_lrf_dependent_stages(cfg: dict, cfg_path: str) -> None:
    """Run stages whose inputs/outputs belong to one LRF experiment."""
    if cfg.get("skip_training", False):
        skip_stage("obb_detector_training", "skip_training is True")
    else:
        print("Starting direction-estimation training...")
        run_script(OBB_DETECTOR_TRAINING_SCRIPT, cfg_path)

    if cfg.get("skip_detection", False):
        skip_stage("obb_detection", "skip_detection is True")
    else:
        print("Starting object detection process...")
        run_script(OBB_DETECTION_SCRIPT, cfg_path)

    if cfg.get("skip_id_tracking", False):
        skip_stage("multi_staged_association", "skip_id_tracking is True")
        skip_stage("refinement", "skip_id_tracking is True")
    else:
        print("Starting ID tracking process...")
        run_script(os.path.join(CURRENT_DIR, "multi_staged_association_variable.py") if cfg.get("VARIABLE_NUM_OBJECTS", False) else MULTI_STAGED_ASSOCIATION_SCRIPT, cfg_path)
        if cfg.get("skip_refinement", False):
            skip_stage("refinement", "skip_refinement is True")
        else:
            print("Starting refinement...")
            run_script(os.path.join(CURRENT_DIR, "refinement_variable.py") if cfg.get("VARIABLE_NUM_OBJECTS", False) else REFINEMENT_SCRIPT, cfg_path)

    if cfg.get("skip_creating_video", False):
        skip_stage("create_video", "skip_creating_video is True")
    else:
        print("Creating tracking video...")
        run_script(CREATE_TRACKING_VIDEO_SCRIPT, cfg_path)


def _main_impl() -> None:
    global _MASTER_SEED, _TIMING_CSV_PATH

    session_path = ""

    cfg_path = sys.argv[1]
    cfg_path_abs = os.path.abspath(cfg_path)
    cfg_dir = os.path.dirname(cfg_path_abs) or os.getcwd()
    _TIMING_CSV_PATH = os.path.join(cfg_dir, "time.csv")
    _init_timing_csv(_TIMING_CSV_PATH)
    log_path = _begin_logging(cfg_path)

    cfg = load_config(cfg_path)

    training_cfg = cfg.get("training", {}) or {}
    if not isinstance(training_cfg, dict):
        raise ValueError("training must be a mapping.")
    lr0_values = parse_lr0_values(training_cfg.get("LR0", DEFAULT_LR0))
    lrf_values = parse_lrf_values(training_cfg.get("LRF", DEFAULT_LRF))

    _MASTER_SEED = normalize_seed(cfg.get("RANDOM_SEED", 0))
    print(f"[SEED] RANDOM_SEED={_MASTER_SEED}")

    print(f"Using configuration: {cfg_path}")
    print(f"Saving timing CSV to: {_TIMING_CSV_PATH}")

    _batch_start = datetime.now()

    run_lrf_independent_stages(cfg, cfg_path)

    lr_pairs = [
        (current_lr0, current_lrf)
        for current_lr0 in lr0_values
        for current_lrf in lrf_values
    ]
    with tempfile.TemporaryDirectory(prefix="amadeus_lr_") as temp_cfg_dir:
        total_lr = len(lr_pairs)
        for index, (current_lr0, current_lrf) in enumerate(lr_pairs, start=1):
            run_cfg = copy.deepcopy(cfg)
            run_training = run_cfg.setdefault("training", {})
            run_training["LR0"] = current_lr0
            run_training["LRF"] = current_lrf
            experiment_name = experiment_dir_name_from_cfg(run_cfg)
            lr0_text = format_lr0_for_name(current_lr0)
            lrf_text = format_lrf_for_name(current_lrf)
            print(
                f"\n[LR BATCH] {index}/{total_lr} "
                f"lr0={lr0_text} lrf={lrf_text} experiment={experiment_name}",
                flush=True,
            )

            temp_cfg_path = os.path.join(temp_cfg_dir, f"config_lr_{index}.yaml")
            with open(temp_cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(run_cfg, f, sort_keys=False, allow_unicode=True)

            try:
                run_lrf_dependent_stages(run_cfg, temp_cfg_path)
            except Exception as exc:
                raise RuntimeError(
                    f"LR batch failed at {index}/{total_lr}: "
                    f"lr0={lr0_text}, lrf={lrf_text}, experiment={experiment_name}"
                ) from exc

    _batch_end = datetime.now()
    print(f"\n[{_batch_end.strftime('%Y-%m-%d %H:%M:%S')}] All steps done. Total elapsed: {_fmt_elapsed(_batch_end - _batch_start)}", flush=True)
    _append_timing_row(
        stage="batch_total",
        script="batch.py",
        status="success",
        start=_batch_start,
        end=_batch_end,
    )
    _write_archive("[INFO] batch completed successfully\n")


def main() -> None:
    cfg_path = sys.argv[1]
    _begin_logging(cfg_path)
    status = "success"
    try:
        _main_impl()
    except BaseException:
        status = "failed"
        traceback.print_exc()
        raise
    finally:
        _finish_logging(status)


if __name__ == "__main__":
    from without_direction_estimation import prepare_config
    if len(sys.argv) < 2:
        raise SystemExit("Usage: batch.py CONFIG_PATH")
    _begin_logging(sys.argv[1])
    try:
        sys.argv[1] = prepare_config(sys.argv[1])
        _finish_logging("dispatch")
        main()
    except BaseException:
        traceback.print_exc()
        _finish_logging("failed")
        raise
