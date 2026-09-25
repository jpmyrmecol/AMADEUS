# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Manual hardware smoke: python tests/smoke_compute_backend.py [auto|cpu|cuda:0|mps].

Uses synthetic inputs and an untrained Ultralytics model; no datasets/downloads.
Does not certify model quality, sustained throughput, or checkpoint recovery.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision
from ultralytics.nn.tasks import DetectionModel
from main.compute_backend import resolve_device, runtime_for
from main.compute_telemetry import sample_accelerator, query_wddm_non_local_usage


def main():
    selected = resolve_device(sys.argv[1] if len(sys.argv) > 1 else 'auto', 'smoke')
    runtime = runtime_for(selected)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = DetectionModel('yolo11n.yaml', nc=1, verbose=False).to(runtime.torch_device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    images = torch.rand(2, 3, 64, 64, device=runtime.torch_device)
    predictions = model(images)
    loss = sum(output.square().mean() for output in predictions)
    assert torch.isfinite(loss).item()
    loss.backward()
    optimizer.step()
    runtime.synchronize()
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=runtime.torch_device)
    scores = torch.tensor([0.9, 0.8], device=runtime.torch_device)
    assert torchvision.ops.nms(boxes, scores, 0.5).numel() == 1
    print('Backend:', runtime.backend.value, 'torch:', torch.__version__, 'vision:', torchvision.__version__)
    print('Memory:', runtime.memory(), 'telemetry:', sample_accelerator(selected))
    print('WDDM:', query_wddm_non_local_usage(device=selected) if runtime.wddm else 'disabled')
    del model, optimizer, images, predictions, loss
    runtime.empty_cache()
    print('PASS: synthetic Ultralytics forward/backward/optimizer, NMS, memory, cache, sync')


if __name__ == '__main__':
    main()
