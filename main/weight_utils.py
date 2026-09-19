# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from checkpoint_utils import weight_to_checkpoint_number
from training_paths import training_results_csv_path


@dataclass(frozen=True)
class TrainingEpochSummary:
    best_epoch: int
    last_epoch: int
    best_fitness: float


def _metric_key(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    return re.sub(r"\s+", "", text)


def _find_column(fieldnames: list[str], required: str, excluded: str | None = None) -> str | None:
    required_key = _metric_key(required)
    excluded_key = _metric_key(excluded) if excluded is not None else None
    for name in fieldnames:
        key = _metric_key(name)
        if required_key in key and (excluded_key is None or excluded_key not in key):
            return name
    return None


def read_training_epoch_summary(results_csv: str) -> TrainingEpochSummary | None:
    """Resolve best/last completed epochs using the Ultralytics fitness formula."""
    if not os.path.isfile(results_csv):
        return None

    with open(results_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [str(name) for name in (reader.fieldnames or [])]
        epoch_col = _find_column(fieldnames, "epoch")
        map5095_col = _find_column(fieldnames, "map50-95")
        map50_col = _find_column(fieldnames, "map50", "map50-95")
        if epoch_col is None or map50_col is None or map5095_col is None:
            return None

        rows: list[tuple[int, float]] = []
        for row in reader:
            try:
                raw_epoch_float = float(row[epoch_col])
                map50 = float(row[map50_col])
                map5095 = float(row[map5095_col])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (raw_epoch_float, map50, map5095)):
                continue
            if not raw_epoch_float.is_integer():
                continue
            rows.append((int(raw_epoch_float), 0.1 * map50 + 0.9 * map5095))

    if not rows:
        return None

    # Ultralytics versions have emitted both 0-based and 1-based epoch columns.
    epoch_offset = 1 if min(raw_epoch for raw_epoch, _ in rows) == 0 else 0
    completed_rows = [
        (raw_epoch + epoch_offset, fitness)
        for raw_epoch, fitness in rows
    ]
    last_epoch = max(epoch for epoch, _ in completed_rows)
    best_fitness = max(fitness for _, fitness in completed_rows)
    # Ultralytics updates best.pt on an equal best fitness, so prefer the latest
    # matching epoch rather than the first maximum.
    best_epoch = max(
        epoch
        for epoch, fitness in completed_rows
        if math.isclose(fitness, best_fitness, rel_tol=1e-12, abs_tol=1e-15)
    )
    return TrainingEpochSummary(best_epoch, last_epoch, best_fitness)


def _weight_epoch(weight: str, summary: TrainingEpochSummary) -> int | None:
    normalized = str(weight).strip().lower()
    if normalized == "best":
        return summary.best_epoch
    if normalized == "last":
        return summary.last_epoch
    match = re.fullmatch(r"epoch(\d+)", normalized)
    return weight_to_checkpoint_number(normalized) if match else None


def deduplicate_best_epoch_weights(
    session_path: str,
    model_name: str,
    dataset_name: str,
    weights: Iterable[str],
    *,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Drop numeric/last weights that resolve to the requested best.pt epoch."""
    unique_weights = list(dict.fromkeys(str(weight) for weight in weights))
    if "best" not in {weight.strip().lower() for weight in unique_weights}:
        return unique_weights

    results_csv = training_results_csv_path(session_path, model_name, dataset_name)
    try:
        summary = read_training_epoch_summary(results_csv)
    except (OSError, csv.Error) as exc:
        log(f"[WEIGHT DEDUP] Could not read {results_csv}: {exc}")
        return unique_weights
    if summary is None:
        log(
            "[WEIGHT DEDUP] Could not determine best epoch from "
            f"{results_csv}; keeping all requested weights."
        )
        return unique_weights

    kept: list[str] = []
    for weight in unique_weights:
        normalized = weight.strip().lower()
        epoch = _weight_epoch(weight, summary)
        if normalized != "best" and epoch == summary.best_epoch:
            log(
                f"[WEIGHT DEDUP] skip weight={weight}: same epoch as best "
                f"(epoch={summary.best_epoch}, fitness={summary.best_fitness:.10g})"
            )
            continue
        kept.append(weight)

    log(
        f"[WEIGHT DEDUP] best epoch={summary.best_epoch}, "
        f"last epoch={summary.last_epoch}, weights={','.join(kept)}"
    )
    return kept
