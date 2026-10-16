"""Execute production orchestration with small files and stubbed numerical work."""
import ast
import copy
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_cache_integrity import CacheFixture, ROOT, ci


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    namespace = {'__name__': 'stage_test', **namespace}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


class CacheStageIntegrationTests(CacheFixture):
    def select_mode(self, directionless, variable):
        self.cfg.update(WITHOUT_DIRECTION_ESTIMATION=directionless, VARIABLE_NUM_OBJECTS=variable)
        self.cfg['EMBEDDING']['ENABLE'] = False
        self.out = self.session / f'integration_{directionless}_{variable}'
        self.cache = ci.TrackingCache(self.cfg, str(self.video), 'model', 'dataset', 'last', str(self.out))
        return ROOT / 'main' / ('without_direction_estimation' if directionless else '')

    def test_production_detection_reuses_only_matching_inputs(self):
        for directionless in (False, True):
            with self.subTest(directionless=directionless):
                source = self.select_mode(directionless, False)
                self.cfg['analysis']['SKIP_DETECT_PREVIEW'] = True
                self.cfg['analysis'].pop('CONF', None)
                calls = []
                def inference(*args, **kwargs):
                    calls.append(kwargs)
                    return {'detections': len(calls)}
                def save(path, data):
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    Path(path).write_bytes(pickle.dumps(data))
                namespace = dict(
                    os=os, sys=sys, TrackingCache=ci.TrackingCache,
                    load_config=lambda path: self.cfg,
                    resolve_device=lambda *a, **k: 'cpu',
                    resolve_model_name=lambda name: 'model',
                    resolve_existing_experiment_dir_name=lambda *a, **k: 'dataset',
                    experiment_dir_name_from_cfg=lambda cfg: 'dataset',
                    parse_weight_spec=lambda value: ['last'],
                    deduplicate_best_epoch_weights=lambda *a: ['last'],
                    resolve_tracking_jobs=lambda weights: [{'run_name': 'last', 'source_weight': 'last'}],
                    configured_tracking_video_files=lambda *a: [str(self.video)],
                    read_video_frame_info=lambda path: SimpleNamespace(usable_frame_count=3, width=320, height=320),
                    warn_if_frame_count_adjusted=lambda *a, **k: None,
                    video_inference_image_size=lambda w, h: 320,
                    resolve_batch_size=lambda *a, **k: 1,
                    clamp_frame_range_to_usable_count=lambda *a: (0, 2),
                    build_weight_path=lambda *a: str(self.weights),
                    build_tracking_out_dir=lambda *a: str(self.out),
                    tracking_artifact_path=lambda out, name: str(Path(out) / 'buffer' / name),
                    load_blob_pickle=lambda path: pickle.loads(Path(path).read_bytes()),
                    collect_candidate_df=inference,
                    run_with_oom_retry=lambda fn, batch: fn(batch),
                    filter_candidate_df_for_tracking=lambda data, conf: data,
                    candidate_df_to_blob_df=lambda data, *a: data,
                    save_blob_pickle=save,
                )
                function = load_function(source / 'obb_detection.py', 'main', namespace)
                with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=lambda path: object())}), patch.object(sys, 'argv', ['obb_detection.py', 'test.yaml']):
                    function()
                    first_bytes = self.cache.outputs('detection')[0].read_bytes()
                    function()
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(first_bytes, self.cache.outputs('detection')[0].read_bytes())
                    self.cfg['analysis']['CONF'] = 0.25
                    function()
                    self.assertEqual(len(calls), 2)
                    self.weights.write_bytes(b'new checkpoint ' + str(directionless).encode())
                    function()
                    self.assertEqual(len(calls), 3)
                    self.assertTrue(self.cache.current('detection'))

    def tracking_function(self, source, variable, calls):
        def save(out, buffers, count, **kwargs):
            for path in self.cache.outputs('tracking'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps(buffers))
        def track(**kwargs):
            calls.append(kwargs)
            return {'obb_buf': {0: {}}, 'variable_population_stats': {}}
        namespace = dict(os=os, pickle=pickle, json=__import__('json'),
            cache_for_spec=ci.cache_for_spec, build_tracking_out_dir=lambda *a: str(self.out),
            _artifact_path=lambda out, name, suffix='': str(Path(out) / 'buffer' / name),
            load_blob_pickle=lambda path: {}, build_detection_frames=lambda *a: {},
            track_from_detection_frames=track, save_tracking_outputs=save,
            load_population_policy=lambda *a, **k: {}, FWD_ARTIFACT_SUFFIX='',
            BUFFERS_PICKLE_NAME='buffers.pkl')
        return load_function(source / ('multi_staged_association_variable.py' if variable else 'multi_staged_association.py'),
                             '_run_single_tracking_job', namespace)

    def test_production_tracking_reuse_and_stale_upstream_rejection(self):
        for directionless in (False, True):
            for variable in (False, True):
                with self.subTest(directionless=directionless, variable=variable):
                    source = self.select_mode(directionless, variable)
                    self.cfg['analysis'].pop('MATCH_IOU', None)
                    self.make('detection')
                    spec = dict(cache_config=self.cfg, info=dict(path=str(self.video), name='video',
                        first_frame=0, last_frame=2, width=320, height=320), run_name='last',
                        session_path=str(self.session), model_name='model', dataset_name='dataset',
                        num_objects=2, conf_th=0.1, nms_iou=0.8, strong_iou=0.5,
                        strong_dir_deg=90., max_age=10, tracking_cost_cfg={}, sharp_turn_accept_run=3,
                        population_config={'analysis': self.cfg['analysis']})
                    calls = []
                    run = self.tracking_function(source, variable, calls)
                    run(spec)
                    run(spec)
                    self.assertEqual(len(calls), 1)
                    self.cfg['analysis']['MATCH_IOU'] = 0.3
                    spec['strong_iou'] = 0.3
                    run(spec)
                    self.assertEqual(len(calls), 2)
                    self.video.write_bytes(self.video.read_bytes() + b'x')
                    with self.assertRaisesRegex(RuntimeError, 'Rerun detection'):
                        run(spec)
                    self.assertEqual(len(calls), 2)

    def correction_function(self, source, variable, calls):
        def save(out, buffers, count, *, artifact_suffix, corrections_log):
            for path in self.cache.outputs('refinement', artifact_suffix):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps(buffers))
        def geometry(*args, **kwargs):
            calls['geometry'] += 1
            return []
        def save_final(*args):
            calls['exports'] += 1
            return 'result.csv'
        def train(*args, **kwargs):
            calls['train'] += 1
            for path in self.cache.outputs('embedding'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'completed embedding ' + path.name.encode())
            return object(), {0: [1.0], 1: [2.0]}, 0.95
        def cached(*args, **kwargs):
            if all(path.is_file() for path in self.cache.outputs('embedding')):
                calls['embedding_reuse'] += 1
                return [], {0: [1.0], 1: [2.0]}, 0.95
            return None
        embedding_module = SimpleNamespace(
            build_fragments=lambda *a, **k: [], load_cached_crop_metadata=lambda *a, **k: None,
            load_cached_embedding_artifacts=cached, load_or_train_embedding=train,
            embedding_available=lambda score: True, EmbeddingSegmentationRequired=type('MissingSeg', (Exception,), {}))
        modules = {name: embedding_module for name in (
            'identity_correction', 'identity_correction_variable',
            'without_direction_estimation.identity_correction',
            'without_direction_estimation.identity_correction_variable')}
        p = patch.dict(sys.modules, modules)
        p.start()
        self.addCleanup(p.stop)
        namespace = dict(os=os, cache_for_spec=ci.cache_for_spec,
            normalize_seed=lambda seed: int(seed), _cfg_bool=lambda value: bool(value),
            DEFAULT_INTERACT_IOU=0.1, np=__import__('numpy'),
            _resolve_existing_correction_out_dir=lambda spec: ('dataset', str(self.out)),
            load_tracking_buffers=lambda *a: {'obb_buf': {0: {}, 1: {}}},
            save_existing_corrected_final_result=lambda *a: save_final(),
            _artifact_path=lambda out, name, suffix='': str(Path(out) / 'buffer' / name),
            FWD_ARTIFACT_SUFFIX='', BUFFERS_PICKLE_NAME='buffers.pkl',
            ID_RESOLVED_ARTIFACT_SUFFIX='id_resolved', FILLED_ARTIFACT_SUFFIX='filled',
            _auto_img_size=lambda *a: 32,
            _resolve_contact_episodes_embedding=lambda *a, **k: [],
            _kalman_gap_fill=geometry, _overlap_gap_fill=lambda *a, **k: [],
            _fix_local_spikes=lambda *a, **k: [], _mark_missing_frames=geometry,
            save_corrected_outputs=save, save_final_result_csv=save_final)
        return load_function(source / ('refinement_variable.py' if variable else 'refinement.py'),
                             '_run_single_correction_job', namespace)

    def test_production_refinement_toggle_runs_embedding_in_all_modes(self):
        for directionless in (False, True):
            for variable in (False, True):
                with self.subTest(directionless=directionless, variable=variable):
                    source = self.select_mode(directionless, variable)
                    self.make('detection')
                    self.make('tracking')
                    calls = dict(geometry=0, train=0, exports=0, embedding_reuse=0)
                    spec = dict(cache_config=self.cfg, out_dir=str(self.out), session_path=str(self.session),
                        model_name='model', dataset_name='dataset', run_name='last', video_name='video',
                        video_path=str(self.video), num_objects=2, strong_iou=0.5, strong_dir_deg=90., max_age=10,
                        flip_frame_threshold=150, flip_duration_sec=5., source_fps=30., num_workers=1,
                        embedding_seed=0, embedding=self.cfg['EMBEDDING'])
                    run = self.correction_function(source, variable, calls)
                    run(spec)
                    before = calls['geometry']
                    run(spec)
                    self.assertEqual(calls['geometry'], before)
                    self.assertEqual(calls['train'], 0)
                    self.cfg['EMBEDDING']['ENABLE'] = True
                    run(spec)
                    self.assertEqual(calls['train'], 1)
                    self.assertEqual(self.cache.family(), 'id_resolved')
                    run(spec)
                    self.assertEqual(calls['train'], 1)
                    # Only the episode resolver changes, so reuse the embedding.
                    self.cfg['EMBEDDING']['EPISODE_MAX_LEN'] = 60
                    run(spec)
                    self.assertEqual(calls['train'], 1)
                    self.assertEqual(calls['embedding_reuse'], 1)
                    self.cfg['RANDOM_SEED'] += 1
                    spec['embedding_seed'] += 1
                    run(spec)
                    self.assertEqual(calls['train'], 2)
                    self.cfg['EMBEDDING'].pop('EPISODE_MAX_LEN')

    def test_spec_builders_pass_original_configuration(self):
        for prefix in (ROOT / 'main', ROOT / 'main' / 'without_direction_estimation'):
            for name in ('multi_staged_association.py', 'multi_staged_association_variable.py',
                         'refinement.py', 'refinement_variable.py'):
                tree = ast.parse((prefix / name).read_text(encoding='utf-8'))
                main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
                self.assertIn("'cache_config': cfg", ast.unparse(main))


if __name__ == '__main__':
    unittest.main()
