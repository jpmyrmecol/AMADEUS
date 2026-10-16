# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Content-validated caches for detection, association and refinement.

A completion record is written only after all outputs have been saved. Legacy
outputs without a record are not accepted as a cache. Only generated artifacts
in the current tracking run are invalidated. Final CSVs/videos are archived.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

_MAIN_DIR = Path(__file__).resolve().parent
_SCHEMA = 1
_FILE_HASHES: dict[tuple, str] = {}
_STAGE_ORDER = ('detection', 'tracking', 'embedding', 'refinement')
_SKIP_KEYS = {'detection': 'skip_detection', 'tracking': 'skip_id_tracking',
              'refinement': 'skip_refinement'}


def file_signature(path: str | Path) -> dict:
    """Hash the complete file, memoizing only within this process/stat version."""
    path = Path(path).resolve()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {'path': str(path), 'missing': True}
    key = (str(path), stat.st_dev, stat.st_ino, stat.st_size,
           stat.st_mtime_ns, stat.st_ctime_ns)
    digest = _FILE_HASHES.get(key)
    if digest is None:
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                hasher.update(chunk)
        after = path.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError(f'File changed while checking cache: {path}')
        digest = hasher.hexdigest()
        _FILE_HASHES[key] = digest
    return {'path': str(path), 'size': stat.st_size, 'sha256': digest}


def _versions(names: tuple[str, ...]) -> dict:
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _json_value(value):
    """Use the same JSON representation in memory and on disk."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _safe_component(value, default):
    value = re.sub(r'[<>:"/\\|?*\s]+', '_', str(value or '').strip())
    return re.sub(r'_+', '_', value).strip(' ._') or default


class TrackingCache:
    """One video/checkpoint/mode, with separate completion records per stage."""

    def __init__(self, cfg: dict, video_path: str, model: str, dataset: str,
                 run: str, out_dir: str):
        self.cfg = cfg
        self.video = str(video_path)
        self.model, self.dataset, self.run = str(model), str(dataset), str(run)
        self.session = Path(cfg['SESSION_PATH'])
        self.out = Path(out_dir)
        self.variable = bool(cfg.get('VARIABLE_NUM_OBJECTS', False))
        self.directionless = bool(cfg.get('WITHOUT_DIRECTION_ESTIMATION', False))
        self.source = _MAIN_DIR / 'without_direction_estimation' if self.directionless else _MAIN_DIR
        self.pending: dict[str, dict] = {}

    def record_path(self, stage: str) -> Path:
        if stage not in _STAGE_ORDER:
            raise ValueError(f'Unknown cache stage: {stage}')
        return self.out / 'log' / f'cache_{stage}.json'

    def _read(self, stage: str) -> dict | None:
        try:
            record = json.loads(self.record_path(stage).read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return record if isinstance(record, dict) else None

    def embedding_enabled(self) -> bool:
        enabled = (self.cfg.get('EMBEDDING', {}) or {}).get('ENABLE', False)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {'1', 'true', 'yes', 'on'}
        return bool(enabled) and not (
            float(self.cfg.get('MIN_OVERLAP', 0.01)) == 0.0
            and float(self.cfg.get('MAX_OVERLAP', 0.5)) == 0.0)

    def _embedding_inputs(self) -> dict[str, Path]:
        # This is the same resolver used by identity_correction._resolve_seg_config.
        from segmentation_metadata import segmentation_paths_for_session
        _, background = segmentation_paths_for_session(str(self.session), self.video)
        inputs = {'video': Path(self.video),
                  'tracking_buffers': self.out / 'buffer' / 'buffers.pkl'}
        if background:
            inputs['segmentation_background'] = Path(background)
            inputs['segmentation_settings'] = Path(background).parent / 'segmentation_gui_config.json'
        configured_background = str(self.cfg.get('BACKGROUND_PATH') or
            (self.cfg.get('EMBEDDING', {}) or {}).get('BACKGROUND_PATH', '')).strip()
        if configured_background:
            inputs['configured_background'] = Path(configured_background)
        return inputs

    def request(self, stage: str) -> dict:
        analysis = self.cfg.get('analysis', {}) or {}
        settings = {'directionless': self.directionless, 'variable_population': self.variable}
        files: dict[str, Path] = {}
        code = {_MAIN_DIR / 'cache_integrity.py'}
        packages = ('numpy', 'pandas', 'opencv-python')
        if stage == 'detection':
            training_paths = importlib.import_module(
                'without_direction_estimation.training_paths' if self.directionless else 'training_paths')
            files = {'video': Path(self.video), 'weights': Path(training_paths.training_weight_path(
                str(self.session), self.model, self.dataset, self.run))}
            num_objects = int(self.cfg.get('NUM_OBJECTS', 0))
            settings.update(
                conf=float(analysis.get('CONF', 0.1)),
                nms_iou=float(analysis.get('NMS_IOU', 0.80)),
                max_det=int(analysis.get('MAX_DET', 10000)) if self.variable else max(300, num_objects * 2),
                first_frame=int(analysis.get('FIRST_FRAME', 0)),
                last_frame=int(analysis.get('LAST_FRAME', -1)),
                batch_size=analysis.get('BATCH_SIZE', 'auto'),
                device=analysis.get('DEVICE', 'auto'))
            code.update(self.source / name for name in ('obb_detection.py', 'training_paths.py'))
            code.update(_MAIN_DIR / name for name in (
                'batch_utils.py', 'compute_backend.py', 'checkpoint_utils.py',
                'weight_utils.py', 'video_frame_count.py', 'path_utils.py'))
            packages += ('torch', 'torchvision', 'ultralytics')
        elif stage == 'tracking':
            files = {'detections': self.out / 'buffer' / 'all_blobs.pkl',
                     'detection_record': self.record_path('detection')}
            settings.update(
                num_objects=0 if self.variable else int(self.cfg['NUM_OBJECTS']),
                conf=float(analysis.get('CONF', 0.1)),
                nms_iou=float(analysis.get('NMS_IOU', 0.80)),
                first_frame=int(analysis.get('FIRST_FRAME', 0)),
                last_frame=int(analysis.get('LAST_FRAME', -1)),
                strong_iou=float(analysis.get('MATCH_IOU', 0.50)),
                strong_dir_deg=float(analysis.get('MATCH_ANGLE', 90.0)),
                max_age=int(analysis.get('MAX_AGE', 10)),
                tracking_cost=dict(analysis.get('TRACKING_COST', {}) or {}))
            settings['sharp_turn_accept_run'] = int(settings['tracking_cost'].get(
                'SHARP_TURN_ACCEPT_RUN', analysis.get('SHARP_TURN_ACCEPT_RUN', 3)))
            code.update(self.source / name for name in ('multi_staged_association.py', 'obb_detection.py'))
            code.update(_MAIN_DIR / name for name in ('assign_types.py', 'tracking_artifacts.py'))
            if self.variable:
                settings['population'] = {name: analysis.get(name, default) for name, default in (
                    ('VARIABLE_POPULATION_MODE', 'edge'), ('VARIABLE_EDGE_CONFIRM_FRAMES', 2),
                    ('VARIABLE_INTERIOR_CONFIRM_FRAMES', 10), ('VARIABLE_INTERIOR_MAX_AGE', None))}
                files['initial_tracking_stats'] = self.session / 'initial_tracking' / 'tracking_stats.csv'
                code.update((self.source / 'multi_staged_association_variable.py', _MAIN_DIR / 'variable_population.py'))
            packages += ('scipy', 'filterpy')
        elif stage in {'embedding', 'refinement'}:
            files = {'tracking_buffers': self.out / 'buffer' / 'buffers.pkl',
                     'tracking_record': self.record_path('tracking')}
            if stage == 'embedding' or self.embedding_enabled():
                files.update(self._embedding_inputs())
                embedding = dict(self.cfg.get('EMBEDDING', {}) or {})
                settings['embedding'] = {k: v for k, v in embedding.items()
                    if k not in {'PREVIEW_COUNT', 'ENABLE', 'EPISODE_MAX_LEN', 'INTERACT_IOU'}}
                settings['random_seed'] = self.cfg.get('RANDOM_SEED', 0)
                code.add(self.source / 'identity_correction.py')
                code.update(_MAIN_DIR / name for name in (
                    'segmentation_metadata.py', 'segmentation_core.py', 'roi.py',
                    'random_utils.py', 'compute_backend.py', 'batch_utils.py'))
                if self.variable:
                    code.add(self.source / 'identity_correction_variable.py')
                packages += ('torch', 'torchvision', 'h5py', 'scikit-learn')
            code.update(self.source / name for name in ('multi_staged_association.py', 'obb_detection.py'))
            packages += ('scipy', 'filterpy')
            if stage == 'refinement':
                settings.update(
                    strong_iou=float(analysis.get('MATCH_IOU', 0.50)),
                    strong_dir_deg=float(analysis.get('MATCH_ANGLE', 90.0)),
                    max_age=int(analysis.get('MAX_AGE', 10)),
                    max_axis_error_deg=float(analysis.get('MAX_AXIS_ERR', 45.0)),
                    flip_sec=float(analysis['FLIP_SEC']),
                    embedding_enabled=self.embedding_enabled())
                if self.embedding_enabled():
                    settings['episode_max_len'] = int((self.cfg.get('EMBEDDING', {}) or {}).get('EPISODE_MAX_LEN', 0))
                    settings['interact_iou'] = float((self.cfg.get('EMBEDDING', {}) or {}).get('INTERACT_IOU', 0.1))
                else:
                    # Overlap filling uses this threshold even without embedding.
                    settings['interact_iou'] = float((self.cfg.get('EMBEDDING', {}) or {}).get('INTERACT_IOU', 0.1))
                files['video'] = Path(self.video)  # FLIP_SEC is converted using this video's FPS.
                code.add(self.source / 'refinement.py')
                code.update(_MAIN_DIR / name for name in ('assign_types.py', 'tracking_constants.py', 'tracking_artifacts.py'))
                if self.variable:
                    code.update((self.source / 'refinement_variable.py', self.source / 'multi_staged_association_variable.py',
                                 _MAIN_DIR / 'variable_population.py'))
        else:
            raise ValueError(f'Unknown cache stage: {stage}')
        return _json_value({'schema': _SCHEMA, 'stage': stage, 'settings': settings,
            'inputs': {key: file_signature(path) for key, path in sorted(files.items())},
            'implementation': {str(path.relative_to(_MAIN_DIR)): file_signature(path)['sha256']
                               for path in sorted(code)},
            'packages': _versions(packages)})

    def outputs(self, stage: str, family: str = '') -> list[Path]:
        if stage == 'detection':
            return [self.out / 'buffer' / 'all_blobs.pkl']
        if stage == 'embedding':
            return [self.out / name for name in ('embedding.h5', 'embedding.pt', 'embedding_training_metrics.csv')]
        if stage == 'tracking':
            suffix = ''
        elif stage == 'refinement' and family in {'filled', 'id_resolved'}:
            suffix = '_' + family
        else:
            raise ValueError(f'Invalid cache output family: {stage}/{family}')
        paths = [self.out / 'buffer' / (name + suffix + ext) for name, ext in (
            ('buffers', '.pkl'), ('assign_type', '.pkl'), ('assign_type_history', '.pkl'), ('provenance', '.csv'))]
        if stage == 'refinement':
            csv_suffix = '_id_resolved' if family == 'id_resolved' else ''
            paths += [self.out / (name + csv_suffix + '.csv') for name in ('obbs', 'directions')]
            paths += [self.out / 'log' / ('corrections_log' + suffix + '.json')]
        if stage == 'tracking' and self.variable:
            paths.append(self.out / 'population_policy.json')
        return paths

    def current(self, stage: str, *, check_upstream: bool = True) -> bool:
        if check_upstream and stage != 'detection':
            upstream = 'detection' if stage == 'tracking' else 'tracking'
            if not self.current(upstream):
                return False
        record = self._read(stage)
        if record is None or record.get('request') != self.request(stage):
            return False
        family = record.get('family', '')
        if stage == 'refinement' and family not in {'filled', 'id_resolved'}:
            return False
        expected = self.outputs(stage, family)
        signatures = [file_signature(path) for path in expected]
        if stage == 'refinement' and self.embedding_enabled():
            dependencies = self.outputs('embedding') + [self.record_path('embedding')]
            if record.get('embedding_dependencies') != [file_signature(p) for p in dependencies]:
                return False
        return (all(item.get('size', 0) > 0 for item in signatures)
                and signatures == record.get('outputs'))

    def require(self, stage: str) -> None:
        if stage != 'detection':
            self.require('detection' if stage == 'tracking' else 'tracking')
        if not self.current(stage, check_upstream=False):
            raise RuntimeError(
                f'{stage} cache is missing, incomplete or no longer matches the input/configuration '
                f'[{self.run}:{Path(self.video).stem}]. Rerun {stage}; clear '
                f'{_SKIP_KEYS.get(stage, "the corresponding skip option")} when running the batch.')

    def family(self) -> str:
        self.require('refinement')
        return self._read('refinement')['family']

    def _archive_results(self) -> None:
        run = self.run + '_variable' if self.variable else self.run
        stem = '_'.join(_safe_component(value, default) for value, default in (
            (self.model, 'model'), (self.dataset, 'dataset'), (run, 'run'),
            (Path(self.video).stem, 'video')))
        archive = self.session / 'results' / 'cache_archive' / uuid.uuid4().hex
        for suffix in ('', '_id_resolved'):
            for extension in ('.csv', '.mp4'):
                path = self.session / 'results' / (stem + suffix + extension)
                if path.is_file():
                    archive.mkdir(parents=True, exist_ok=True)
                    os.replace(path, archive / path.name)

    def invalidate(self, stage: str) -> None:
        """Remove only known generated caches and their downstream dependants."""
        stages = {'detection': ('detection', 'tracking', 'embedding', 'refinement'),
                  'tracking': ('tracking', 'embedding', 'refinement'),
                  'embedding': ('embedding', 'refinement'), 'refinement': ('refinement',)}[stage]
        for name in stages:
            self.record_path(name).unlink(missing_ok=True)
            families = ('filled', 'id_resolved') if name == 'refinement' else ('',)
            for family in families:
                for path in self.outputs(name, family):
                    path.unlink(missing_ok=True)
            if name == 'embedding':
                for path in (self.out / 'embedding_vram_calib.json',
                             self.out / 'log' / 'embedding_vram_calib.json'):
                    path.unlink(missing_ok=True)
        self._archive_results()

    def begin(self, stage: str) -> None:
        if stage != 'detection':
            self.require('detection' if stage == 'tracking' else 'tracking')
        request = self.request(stage)
        self.invalidate(stage)
        self.pending[stage] = request
        print(f'[CACHE] Recomputing {stage}: missing or changed cache [{self.run}:{Path(self.video).stem}]', flush=True)

    def complete(self, stage: str, family: str = '') -> None:
        request = self.pending[stage]
        if request != self.request(stage):
            raise RuntimeError(f'{stage} inputs changed during processing; cache was not committed.')
        signatures = [file_signature(path) for path in self.outputs(stage, family)]
        if any(item.get('size', 0) <= 0 for item in signatures):
            raise RuntimeError(f'{stage} outputs are incomplete; cache was not committed.')
        record = {'request': request, 'family': family, 'outputs': signatures}
        if stage == 'refinement' and self.embedding_enabled():
            dependencies = self.outputs('embedding') + [self.record_path('embedding')]
            record['embedding_dependencies'] = [file_signature(p) for p in dependencies]
        path = self.record_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(record, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
        del self.pending[stage]


def cache_for_spec(spec: dict, out_dir: str) -> TrackingCache:
    video = spec['info']['path'] if 'info' in spec else spec['video_path']
    return TrackingCache(spec['cache_config'], video, spec['model_name'],
                         spec['dataset_name'], spec['run_name'], out_dir)


def validate_skipped_analysis(cfg: dict) -> None:
    """A skip never authorizes consuming stale outputs in a later stage."""
    # Video export validates its actual create_video.WEIGHT jobs separately.
    consumers = [stage for stage, flag in _SKIP_KEYS.items() if not cfg.get(flag, False)]
    if not consumers or consumers == ['detection']:
        return
    needed = set()
    if any(stage in consumers for stage in ('tracking', 'refinement')):
        if cfg.get('skip_detection', False):
            needed.add('detection')
    if 'refinement' in consumers and cfg.get('skip_id_tracking', False):
        needed.add('tracking')
    if not needed:
        return
    detection = importlib.import_module(
        'without_direction_estimation.obb_detection' if cfg.get('WITHOUT_DIRECTION_ESTIMATION', False) else 'obb_detection')
    session = str(cfg['SESSION_PATH'])
    model = detection.resolve_model_name(os.path.splitext(cfg.get('training', {}).get('PRETRAINED_MODEL', ''))[0]).split('.')[0]
    dataset = detection.resolve_existing_experiment_dir_name(session, model,
        detection.experiment_dir_name_from_cfg(cfg), stages=('tracking', 'training'), warn_fn=print)
    weights = detection.deduplicate_best_epoch_weights(session, model, dataset,
        list(dict.fromkeys(detection.parse_weight_spec((cfg.get('analysis', {}) or {}).get('WEIGHT', 'last')))))
    for video in detection.configured_tracking_video_files(cfg, str(cfg['TRACKING_VIDEO_PATH'])):
        for weight in weights:
            run = detection.build_tracking_run_name(weight)
            out = detection.build_tracking_out_dir(session, model, dataset, run, Path(video).stem)
            if cfg.get('VARIABLE_NUM_OBJECTS', False):
                out = os.path.join(out, 'variable')
            cache = TrackingCache(cfg, video, model, dataset, run, out)
            for stage in _STAGE_ORDER:
                if stage in needed:
                    cache.require(stage)
