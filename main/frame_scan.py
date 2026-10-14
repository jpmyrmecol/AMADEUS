# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""Ordered, bounded-memory frame scanning shared by Search and Processing.

Frames are decoded on one thread (sequential decoding is far cheaper than
seeking) and processed by a worker pool batch by batch. Results are handed
to the caller in ascending frame order, so streaming detectors that need
consecutive frames can consume a parallel scan. Images of at most two batches
(the one being processed and, when prefetching, the next decoded one) are
held at a time.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
from typing import Any, Callable, Iterator, Sequence

import numpy as np


def frame_blob_digest(blobs) -> int:
    """64-bit fingerprint of one frame's blob order and exact contours.

    Search results address blobs by (frame, blob_index). Processing segments
    the frames again, so it compares this digest before reusing an index:
    a mismatch means the index could name a different blob.
    """
    digest = hashlib.blake2b(digest_size=8)
    digest.update(len(blobs).to_bytes(4, "little"))
    for blob in blobs:
        contour = np.ascontiguousarray(np.asarray(blob.contour, dtype=np.int32).reshape(-1, 2))
        digest.update(int(blob.blob_index).to_bytes(4, "little", signed=True))
        digest.update(int(contour.shape[0]).to_bytes(4, "little"))
        digest.update(contour.tobytes())
    return int.from_bytes(digest.digest(), "little")


def iter_frames_in_order(
    frame_ids: Sequence[int],
    decode_batch: Callable[[list[int]], list[tuple[int, Any]]],
    process: Callable[[int, Any], Any],
    *,
    workers: int,
    batch_size: int,
    prefetch: bool,
    cancelled: Callable[[], bool] = lambda: False,
) -> Iterator[tuple[int, Any]]:
    """Yield ``(frame, process(frame, image))`` in ascending frame order.

    ``decode_batch`` runs on a single decoder thread and must return one
    image for every requested frame. ``process`` runs on ``workers`` threads.
    A batch is submitted to the pool before the previous batch's results are
    handed out, so the caller's per-frame work overlaps the workers. The
    previous batch has finished (its images are released) before the next is
    decoded or submitted, so images of one batch are being processed and at
    most one more is decoded ahead (with ``prefetch``).
    """
    frame_ids = [int(fid) for fid in frame_ids]
    batch_size = max(1, int(batch_size))
    batches = [frame_ids[start:start + batch_size]
               for start in range(0, len(frame_ids), batch_size)]

    def submitted(batch, frames):
        futures = {fid: executor.submit(process, fid, image) for fid, image in frames}
        missing = [fid for fid in batch if fid not in futures]
        if missing:
            raise RuntimeError(f"Frame {missing[0]} was not decoded.")
        return [(fid, futures[fid]) for fid in batch]

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor, \
            ThreadPoolExecutor(max_workers=1) as decoder:
        pending = None
        previous: list = []
        for index, batch in enumerate(batches):
            if cancelled():
                return
            wait([future for _fid, future in previous])
            if pending is None:
                pending = decoder.submit(decode_batch, batch)
            frames = pending.result()
            # Decode the next batch while the pool processes this one.
            pending = (decoder.submit(decode_batch, batches[index + 1])
                       if prefetch and index + 1 < len(batches) else None)
            current = submitted(batch, frames)
            del frames
            for fid, future in previous:
                yield fid, future.result()
                if cancelled():
                    return
            previous = current
        for fid, future in previous:
            yield fid, future.result()
