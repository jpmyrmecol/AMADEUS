# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Pure workflow helpers shared by Segmentation GUI transitions."""


def stage_after_background_change(
    previous_stage: str,
    *,
    needs_background: bool,
    preview_seen: bool,
) -> str:
    """Re-run analysis after background changes without re-gating its preview.

    Showing Background is a one-time video review, not a review of each
    median/max/min image. The confirmation stays valid across method changes,
    but the per-method analysis remains invalid until re-analyzed.
    """
    if needs_background and not preview_seen:
        return "background"
    if previous_stage in {"analysis", "outlier", "processing"}:
        return "analysis"
    return "threshold"
