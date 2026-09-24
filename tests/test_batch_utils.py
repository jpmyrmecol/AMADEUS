"""Deterministic tests for automatic CPU and MPS batch sizing."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import types
import unittest
from unittest.mock import patch

from main import batch_utils


GIB = 1024.0**3


class AutoBatchSizeTests(unittest.TestCase):
    def auto_batch(
        self,
        device_kind: str,
        *,
        available_gib: float | None = 19.1,
        mps_recommended_gib: float | None = 21.3,
        physical_cpu: int = 10,
        env: dict[str, str] | None = None,
    ) -> int:
        if available_gib is None:
            virtual_memory = lambda: (_ for _ in ()).throw(OSError("RAM unavailable"))
        else:
            virtual_memory = lambda: types.SimpleNamespace(available=available_gib * GIB)
        fake_psutil = types.SimpleNamespace(
            virtual_memory=virtual_memory,
            cpu_count=lambda logical=False: physical_cpu,
        )

        with (
            patch.dict(os.environ, env or {}, clear=True),
            patch.dict(sys.modules, {"psutil": fake_psutil}),
            patch.object(batch_utils, "_accelerator_type", return_value=device_kind),
            patch.object(batch_utils, "_cuda_total_gib", return_value=None),
            patch.object(
                batch_utils, "_mps_recommended_gib", return_value=mps_recommended_gib
            ),
            patch.object(
                batch_utils,
                "_estimate_yolo_label_density",
                return_value=(5.0, 4.0, 10_000, 2_000),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return batch_utils.auto_batch_size(
                image_size=576,
                device=device_kind,
                mode="train",
                labels_dir="labels",
                task="obb",
            )

    def test_cpu_default_cap_stays_at_16(self) -> None:
        self.assertEqual(self.auto_batch("cpu", mps_recommended_gib=None), 16)

    def test_mps_selects_profiled_batch_24_when_memory_allows(self) -> None:
        self.assertEqual(self.auto_batch("mps"), 24)

    def test_mps_recommended_memory_remains_a_hard_limit(self) -> None:
        self.assertEqual(self.auto_batch("mps", mps_recommended_gib=8.0), 16)

    def test_mps_unknown_ram_uses_conservative_fallback(self) -> None:
        self.assertEqual(self.auto_batch("mps", available_gib=None), 8)

    def test_mps_environment_cap_is_respected(self) -> None:
        self.assertEqual(
            self.auto_batch("mps", env={"AMADEUS_MAX_AUTO_CPU_BATCH": "8"}), 8
        )


if __name__ == "__main__":
    unittest.main()
