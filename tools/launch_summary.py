# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Print one concise accelerator status after the launcher's environment check."""

import torch

from main.compute_backend import Backend, detect_backend


def main() -> None:
    backend = detect_backend()
    if backend == Backend.NVIDIA_CUDA:
        print(
            f"[AMADEUS] GPU: {torch.cuda.get_device_name(0)} "
            f"(CUDA {torch.version.cuda}, available)",
            flush=True,
        )
    elif backend == Backend.AMD_ROCM:
        print(
            f"[AMADEUS] GPU: {torch.cuda.get_device_name(0)} "
            f"(ROCm {torch.version.hip}, available)",
            flush=True,
        )
    elif backend == Backend.APPLE_MPS:
        print("[AMADEUS] GPU: Apple Silicon (MPS, available)", flush=True)
    else:
        print("[AMADEUS] GPU: unavailable (CPU mode)", flush=True)


if __name__ == "__main__":
    main()
