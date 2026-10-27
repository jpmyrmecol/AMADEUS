# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Group refinement candidate rows for display without removing source rows."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateRun:
    first_index: int
    start_frame: int
    end_frame: int
    tid: int
    assign_code: float
    member_indices: tuple[int, ...]


def build_candidate_runs(candidates: Sequence[Mapping[str, Any]]) -> list[CandidateRun]:
    """Return one representative for each consecutive ID/classification run.

    The displayed classification includes provenance tags in ``label_short``.
    Explanatory long labels do not split a run. Same-frame duplicates belong to
    the same run, and all indices still refer to the original candidate list.
    """
    ordered = sorted(
        range(len(candidates)),
        key=lambda index: (
            int(candidates[index]["frame"]),
            int(candidates[index]["tid"]),
            (int(candidates[index]["source_tid"])
             if int(candidates[index]["tid"]) < 0 else int(candidates[index]["tid"])),
            float(candidates[index]["assign_code"]),
            str(candidates[index]["label_short"]),
            index,
        ),
    )
    runs: list[dict[str, Any]] = []
    latest: dict[tuple[int, int, float, str], dict[str, Any]] = {}
    for index in ordered:
        item = candidates[index]
        frame = int(item["frame"])
        tid = int(item["tid"])
        code = float(item["assign_code"])
        source_tid = int(item["source_tid"]) if tid < 0 else tid
        key = (tid, source_tid, code, str(item["label_short"]))
        run = latest.get(key)
        if run is None or frame > run["end_frame"] + 1:
            run = {
                "first_index": index,
                "start_frame": frame,
                "end_frame": frame,
                "tid": tid,
                "assign_code": code,
                "member_indices": [index],
            }
            runs.append(run)
            latest[key] = run
        else:
            run["end_frame"] = frame
            run["member_indices"].append(index)
    return [
        CandidateRun(**{**run, "member_indices": tuple(run["member_indices"])})
        for run in runs
    ]


def candidate_run_edit_state(
    run: CandidateRun,
    edited_keys: Collection[tuple[int, int]],
) -> str:
    """Report whether no, some, or all distinct frames in a run were edited."""
    count = sum(
        (frame, run.tid) in edited_keys
        for frame in range(run.start_frame, run.end_frame + 1)
    )
    if count == 0:
        return ""
    return "yes" if count == run.end_frame - run.start_frame + 1 else "partial"


def visible_candidate_runs(
    runs: Sequence[CandidateRun],
    enabled_assign_codes: Collection[float],
    edited_keys: Collection[tuple[int, int]],
    *,
    show_only_unedited: bool = False,
) -> list[CandidateRun]:
    """Filter complete runs so a hidden first row cannot expose a later row."""
    return [
        run for run in runs
        if run.assign_code in enabled_assign_codes
        and not (show_only_unedited and candidate_run_edit_state(run, edited_keys) == "yes")
    ]
