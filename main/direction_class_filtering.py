# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Run identify_incorrect_class_labels.py and apply_class_label_filtering.py repeatedly.

The number of refinement passes is read from REFINE_ITERS in the YAML config.
Each pass owns an independent workspace under single_animal_images:

  pass 0: single_animal_images/refine
  pass 1: single_animal_images/refine_add1
  pass 2: single_animal_images/refine_add2
  ...

For each pass, this script writes a temporary config that sets REFINE_DIR_NAME
so the existing refine/apply scripts can operate on that pass-specific workspace.
"""

import os
import sys
import subprocess
from typing import Any

import yaml

from path_utils import resolve_config_paths

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IDENTIFY_INCORRECT_CLASS_LABELS_SCRIPT = os.path.join(CURRENT_DIR, "identify_incorrect_class_labels.py")
APPLY_CLASS_LABEL_FILTERING_SCRIPT = os.path.join(CURRENT_DIR, "apply_class_label_filtering.py")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return resolve_config_paths(cfg)


def dump_config(path: str, cfg: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    rc = subprocess.run(cmd, env=env, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def run_script(script: str, cfg_path: str) -> None:
    run([sys.executable, "-u", script, cfg_path])


def parse_num_refinement(value: Any, default: int = 1) -> int:
    if value is None or str(value).strip() == "":
        n = int(default)
    else:
        n = int(value)
    if n < 0:
        raise ValueError(f"REFINE_ITERS must be >= 0, got {value!r}")
    return n


def refine_dir_name(iteration: int) -> str:
    if iteration <= 0:
        return "refine"
    return f"refine_add{iteration}"


def refine_root(session_path: str, iteration: int) -> str:
    return os.path.join(session_path, "single_animal_images", refine_dir_name(iteration))


def original_ready(root: str) -> bool:
    original = os.path.join(root, "original")
    return (
        os.path.isdir(original)
        and os.path.isdir(os.path.join(original, "images"))
        and os.path.isdir(os.path.join(original, "labels"))
        and os.path.exists(os.path.join(original, "object_pool", "manifest.csv"))
    )


def _dir_has_files(path: str, suffix: str | tuple[str, ...] | None = None) -> bool:
    if not os.path.isdir(path):
        return False
    return any(
        os.path.isfile(os.path.join(path, name))
        and (suffix is None or name.endswith(suffix))
        for name in os.listdir(path)
    )


def dataset_ready(root: str) -> bool:
    dataset = os.path.join(root, "dataset")
    if not os.path.exists(os.path.join(dataset, "data.yaml")):
        return False

    train_img = os.path.join(dataset, "train", "images")
    train_lbl = os.path.join(dataset, "train", "labels")
    test_img = os.path.join(dataset, "test", "images")
    test_lbl = os.path.join(dataset, "test", "labels")
    if (
        _dir_has_files(train_img, (".png", ".jpg", ".jpeg"))
        and _dir_has_files(train_lbl, ".txt")
        and _dir_has_files(test_img, (".png", ".jpg", ".jpeg"))
        and _dir_has_files(test_lbl, ".txt")
    ):
        return True

    return False


def last_model_ready(root: str) -> bool:
    return os.path.isfile(os.path.join(root, "training", "detection", "weights", "last.pt"))


def refine_pass_started(root: str) -> bool:
    # Any of these means the next pass has already consumed the previous pass's
    # apply output. Re-running the previous apply would be redundant and can be
    # expensive after user interruption.
    return (
        original_ready(root)
        or dataset_ready(root)
        or last_model_ready(root)
        or delete_csv_ready(root)
    )


def delete_csv_ready(root: str) -> bool:
    return os.path.exists(os.path.join(root, "delete_blobs.csv"))


def applied_marker_path(root: str, apply_mode: str) -> str:
    safe_mode = str(apply_mode).strip().lower().replace("/", "_").replace("\\", "_")
    return os.path.join(root, f".apply_refine_deletions.{safe_mode}.done")


def apply_done(root: str, apply_mode: str) -> bool:
    return os.path.exists(applied_marker_path(root, apply_mode))


def mark_apply_done(root: str, apply_mode: str) -> None:
    os.makedirs(root, exist_ok=True)
    with open(applied_marker_path(root, apply_mode), "w", encoding="utf-8") as f:
        f.write(f"{apply_mode}\n")


def apply_mode_for_iteration(iteration: int, num_refinement: int) -> str:
    return "all_candidates" if int(iteration) < int(num_refinement) - 1 else "run_policy"


def filtered_without_crossing_ready(session_path: str) -> bool:
    root = os.path.join(session_path, "single_animal_images")
    return (
        os.path.isdir(os.path.join(root, "images"))
        and os.path.isdir(os.path.join(root, "labels"))
        and os.path.isdir(os.path.join(root, "masks"))
        and os.path.isdir(os.path.join(root, "object_pool"))
        and os.path.exists(os.path.join(root, "object_pool", "manifest.csv"))
    )


def should_skip_apply(root: str, iteration: int, num_refinement: int, apply_mode: str, session_path: str) -> tuple[bool, str]:
    # For intermediate passes, the apply output is only an input generator for
    # the next refinement pass.  If the next pass has already created original,
    # dataset, training/last.pt, or delete_blobs.csv, do not redo this apply,
    # even when the previous apply marker was not written because the user
    # interrupted a later step.
    if int(iteration) < int(num_refinement) - 1:
        next_root = refine_root(session_path, int(iteration) + 1)
        if refine_pass_started(next_root):
            return True, f"next pass already started: {next_root}"
        if apply_done(root, apply_mode):
            return False, "apply marker exists but next pass has not started; marker is stale"
        return False, "intermediate apply not completed"

    if int(iteration) >= int(num_refinement) - 1:
        if apply_done(root, apply_mode):
            if filtered_without_crossing_ready(session_path):
                return True, "apply marker exists and filtered single_animal_images output exists"
            return False, "apply marker exists but filtered single_animal_images output is missing; marker is stale"
        return False, "final apply marker is missing"


def make_iteration_config(base_cfg: dict, base_cfg_path: str, iteration: int, apply_mode: str) -> str:
    cfg = dict(base_cfg)
    cfg["REFINE_DIR_NAME"] = refine_dir_name(iteration)
    cfg["REFINE_APPLY_MODE"] = str(apply_mode)
    tmp_dir = os.path.join(str(cfg["SESSION_PATH"]), "single_animal_images", "_iterative_refine_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, f"config_{refine_dir_name(iteration)}.yaml")
    dump_config(out_path, cfg)
    return out_path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python direction_class_filtering.py config.yaml")

    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)
    session_path = str(cfg["SESSION_PATH"])
    num_refinement = parse_num_refinement(cfg.get("REFINE_ITERS", 1), default=1)

    if num_refinement == 0:
        print("REFINE_ITERS=0; skipping iterative refinement.")
        return

    print(f"Using configuration: {cfg_path}")
    print(f"REFINE_ITERS={num_refinement}")

    for iteration in range(num_refinement):
        name = refine_dir_name(iteration)
        root = refine_root(session_path, iteration)
        apply_mode = apply_mode_for_iteration(iteration, num_refinement)
        iter_cfg_path = make_iteration_config(cfg, cfg_path, iteration, apply_mode)

        print(f"\n=== Refinement pass {iteration + 1}/{num_refinement}: {name} ===")
        print(f"Workspace: {root}")
        print(f"Apply mode: {apply_mode}")

        if original_ready(root) and delete_csv_ready(root):
            print(f"Found existing required files in {name}; skipping identify_incorrect_class_labels.py.")
        else:
            print(f"Running identify_incorrect_class_labels.py for {name}...")
            run_script(IDENTIFY_INCORRECT_CLASS_LABELS_SCRIPT, iter_cfg_path)

        if not original_ready(root):
            raise RuntimeError(f"Required original workspace is missing after refine step: {os.path.join(root, 'original')}")
        if not delete_csv_ready(root):
            raise RuntimeError(f"Required delete CSV is missing after refine step: {os.path.join(root, 'delete_blobs.csv')}")

        skip_apply, skip_reason = should_skip_apply(root, iteration, num_refinement, apply_mode, session_path)
        if skip_apply:
            print(f"Skipping apply_class_label_filtering.py for {name} / {apply_mode}: {skip_reason}")
        else:
            if apply_done(root, apply_mode):
                print(f"Found stale apply marker for {name} / {apply_mode}: {skip_reason}")
            print(f"Applying refine deletions for {name} with mode={apply_mode}...")
            run_script(APPLY_CLASS_LABEL_FILTERING_SCRIPT, iter_cfg_path)
            mark_apply_done(root, apply_mode)

    print("\nIterative refinement completed.")


if __name__ == "__main__":
    main()
