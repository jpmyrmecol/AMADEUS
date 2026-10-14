# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""Ordered, bounded-memory frame scanning shared by Search and Processing.

Frames are decoded on one thread (sequential decoding is far cheaper than
seeking) and processed by a worker pool one batch at a time. Results are
handed to the caller in ascending frame order, so streaming detectors that
need consecutive frames can consume a parallel scan. Only the current batch,
plus the next decoded batch when prefetching, is ever held in memory.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
    Workers complete in any order (``as_completed``); results are buffered
    only until their batch is complete and then released in frame order.
    """
    frame_ids = [int(fid) for fid in frame_ids]
    batch_size = max(1, int(batch_size))
    batches = [frame_ids[start:start + batch_size]
               for start in range(0, len(frame_ids), batch_size)]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor, \
            ThreadPoolExecutor(max_workers=1) as decoder:
        pending = None
        for index, batch in enumerate(batches):
            if cancelled():
                return
            if pending is None:
                pending = decoder.submit(decode_batch, batch)
            frames = pending.result()
            # Decode the next batch while the pool processes this one.
            pending = (decoder.submit(decode_batch, batches[index + 1])
                       if prefetch and index + 1 < len(batches) else None)
            futures = {executor.submit(process, fid, image): fid for fid, image in frames}
            del frames
            results: dict[int, Any] = {}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
            futures.clear()
            for fid in batch:
                if fid not in results:
                    raise RuntimeError(f"Frame {fid} was not decoded.")
                yield fid, results.pop(fid)
