# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Execute production definitions without importing GUI or accelerator libraries.

These are tests of the actual function bodies. They do not test application
imports, neural training, or device behavior.
"""
from pathlib import Path
from dataclasses import dataclass, field
import ast
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ('main', 'main/without_direction_estimation')


def load_definitions(path: Path, names: set[str], namespace: dict | None = None) -> dict:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    selected = []
    found = set()
    for node in tree.body:
        name = getattr(node, 'name', None)
        if name in names:
            selected.append(node)
            found.add(name)
        elif isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & names:
                selected.append(node)
                found.update(targets & names)
    if found != names:
        raise AssertionError(f'{path}: missing definitions {names - found}')
    module = ast.Module(body=[
        ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
        *selected,
    ], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {'np': np, 'dataclass': dataclass, 'field': field}
    if namespace is not None:
        scope.update(namespace)
    exec(compile(module, str(path), 'exec'), scope)
    return scope


def initializer(mode: str):
    return load_definitions(ROOT / mode / 'multi_staged_association.py', {
        'INITIAL_CONFIDENCE_THRESHOLD', '_select_initial_assignment',
    })['_select_initial_assignment']


def samplers(mode: str):
    import hashlib
    import random
    rng = load_definitions(ROOT / 'main/random_utils.py', {
        'derive_seed', 'make_numpy_rng', 'make_python_rng',
    }, {'hashlib': hashlib, 'random': random})
    scope = load_definitions(ROOT / mode / 'identity_correction.py', {
        'Fragment', '_SampledPair', '_PairSampler',
        '_SCORE_ALPHA', '_SCORE_DECAY', '_SCORE_BUMP',
    }, {
        'make_numpy_rng': rng['make_numpy_rng'],
        'make_python_rng': rng['make_python_rng'],
    })
    baseline = load_definitions(
        ROOT / 'tests/fixtures/pair_sampler_before.py.txt', {'_PairSampler'}, scope,
    )['_PairSampler']
    return scope['Fragment'], scope['_SampledPair'], scope['_PairSampler'], baseline


class ArrayLosses:
    """Supply a fixed loss array through the Tensor methods used by the sampler."""
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values
