# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Variable-population embedding score with shared training and cache code."""
from __future__ import annotations

from functools import partial
import identity_correction as fixed
from identity_correction import *
from identity_correction import (
    _ASSIGN_MAX_ROSTER,
    _EmbedNet,
    _embed_crop_rows,
)


def _compute_silhouette(
    model: _EmbedNet,
    crops_np: np.ndarray,
    fragments: list[Fragment],
    ss_sample_total: int,
    device: torch.device,
    crops_t: torch.Tensor | None = None,
    seed: int = 0,
) -> float:
    """Evaluate silhouette score on co-observed isolated fragments (no fixed K).

    The sampled-row RNG is re-seeded from (seed, "silhouette_rows") on every
    call rather than reused from a persistent generator, so repeated calls
    within the same training run (constant total_crops/sample_count) draw the
    same crop subset -- silhouette movement then reflects model updates, not
    subset churn.
    """
    # Evaluate only a simultaneously observed set of isolated fragments.
    # Re-entering animals may have different IDs, so global ID count is not K.
    by_frame = {}
    for frag in fragments:
        if len(frag.crop_indices) >= 2:
            for frame in frag.frames:
                by_frame.setdefault(int(frame), []).append(frag)
    if not by_frame:
        return 0.0
    cohort = max(by_frame.values(), key=lambda fs: (len(fs), sum(len(f.crop_indices) for f in fs)))
    if len(cohort) < 2:
        return 0.0
    rng = make_python_rng(seed, 'silhouette_rows')
    per_fragment = max(2, min(1000, int(ss_sample_total)//len(cohort)))
    rows, labels = [], []
    for label, frag in enumerate(cohort):
        chosen = rng.sample(frag.crop_indices, min(per_fragment, len(frag.crop_indices)))
        rows.extend(chosen)
        labels.extend([label]*len(chosen))
    embeds = _embed_crop_rows(model, crops_np, rows, device, crops_t=crops_t)
    return float(silhouette_score(embeds, np.asarray(labels), sample_size=None))

train_embedding = partial(fixed.train_embedding, silhouette_fn=_compute_silhouette)
load_cached_embedding_artifacts = partial(
    fixed.load_cached_embedding_artifacts, silhouette_fn=_compute_silhouette)
load_or_train_embedding = partial(
    fixed.load_or_train_embedding, silhouette_fn=_compute_silhouette)
