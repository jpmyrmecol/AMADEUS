# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
from pathlib import Path
_MAIN_DIR = Path(__file__).resolve().parent.parent
for _path in (str(_MAIN_DIR), str(_MAIN_DIR.parent)):
    if _path not in sys.path:
        sys.path.append(_path)


import os
from pathlib import Path
from typing import Iterator

from experiment_utils import (
    resolve_existing_experiment_dir_name,
)


OUTPUT_ROOT_DIR = "main"
TRAINING_STAGE_DIR = "training"
TRAINING_STAGE_DIRS = (TRAINING_STAGE_DIR,)


def training_project_dir(session_path: str, model_name: str) -> str:
    model_name = str(model_name).replace('\\', '/').split('/')[-1]
    """Canonical output root for new direction-model training runs."""
    return os.path.join(str(session_path), OUTPUT_ROOT_DIR, str(model_name), TRAINING_STAGE_DIR)


def iter_training_weights_dirs(session_path: str, model_name: str, dataset_name: str) -> Iterator[str]:
    resolved_dataset_name = resolve_existing_experiment_dir_name(
        session_path,
        model_name,
        str(dataset_name),
        stages=TRAINING_STAGE_DIRS,
        warn_fn=print,
    )
    weights_dir = os.path.join(
        training_project_dir(session_path, model_name),
        resolved_dataset_name,
        "weights",
    )
    if os.path.isdir(weights_dir):
        yield weights_dir


def training_weight_path(session_path: str, model_name: str, dataset_name: str, weight: str) -> str:
    filename = f"{weight}.pt"
    for weights_dir in iter_training_weights_dirs(session_path, model_name, dataset_name):
        path = os.path.join(weights_dir, filename)
        if os.path.isfile(path):
            return path
    return os.path.join(
        training_project_dir(session_path, model_name),
        str(dataset_name),
        "weights",
        filename,
    )


def training_results_csv_path(session_path: str, model_name: str, dataset_name: str) -> str:
    resolved_dataset_name = resolve_existing_experiment_dir_name(
        session_path,
        model_name,
        str(dataset_name),
        stages=TRAINING_STAGE_DIRS,
        warn_fn=print,
    )
    return os.path.join(
        training_project_dir(session_path, model_name),
        resolved_dataset_name,
        "results.csv",
    )
