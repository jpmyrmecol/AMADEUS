# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Direction-free project configuration; numerical stages live in matching scripts."""

import sys
from pathlib import Path
_MAIN_DIR = Path(__file__).resolve().parent.parent
for _path in (str(_MAIN_DIR), str(_MAIN_DIR.parent)):
    if _path not in sys.path:
        sys.path.append(_path)

from pathlib import Path
import copy
import json
import shutil
import yaml
from path_utils import resolve_config_paths

MODE_DIR = 'without_direction_estimation'
SCHEMA = 'single-animal-obb-horizontal-axial-embedding-v1'


def prepare_config(path):
    path = Path(path).resolve()
    cfg = resolve_config_paths(yaml.safe_load(path.read_text(encoding='utf-8')))
    if not cfg.get('WITHOUT_DIRECTION_ESTIMATION', False):
        raise ValueError('This entry point requires WITHOUT_DIRECTION_ESTIMATION: true.')
    if cfg.get('_DIRECTIONLESS_SCHEMA') == SCHEMA:
        return str(path)
    source = Path(cfg['SESSION_PATH']).resolve()
    target = source / MODE_DIR
    target.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg['SESSION_PATH'] = str(target)
    cfg['_DIRECTIONLESS_SCHEMA'] = SCHEMA
    cfg['_SOURCE_SESSION_PATH'] = str(source)
    cfg['INIT_CSV_PATH'] = str(target / 'initial_tracking' / Path(cfg['INIT_CSV_PATH']).name)
    cfg['YOLO_DATASET_DIR'] = str(target / 'yolo_dataset')
    cfg['training'] = {**cfg.get('training', {}), 'CLASS_NAMES': ['animal'], 'NUM_CLASSES': 1, 'USE_DIRECTION_CLASSES': True}
    # Custom direction-bearing datasets cannot silently cross the mode boundary.
    if cfg.get('CREATE_DATASET_SOURCE_DIRS'):
        raise ValueError('Clear CREATE_DATASET_SOURCE_DIRS before the first direction-free run.')
    # These two small files are consumed by the shared embedding segmenter.
    # Raw videos and segmentation pickles are referenced in place, never edited.
    seg = target / 'segmentation'
    seg.mkdir(exist_ok=True)
    for name in ('background.png', 'segmentation_gui_config.json'):
        src = source / 'segmentation' / name
        if src.is_file():
            shutil.copy2(src, seg / name)
    (target / 'mode.json').write_text(json.dumps({
        'schema': SCHEMA, 'source_session': str(source),
        'class_names': ['animal'], 'embedding_axis': 'horizontal, modulo 180 degrees',
        'final_csv_fields_per_id': ['cx', 'cy', 'w', 'h'],
    }, indent=2), encoding='utf-8')
    output = target / path.name
    output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return str(output)

