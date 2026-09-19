# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""Open-population tracking: monotonic IDs, gated association, no population fill."""
from multi_staged_association import *
from multi_staged_association import _artifact_path, _worker_budget, _auto_num_workers
import multi_staged_association as fixed
import json


def build_tracking_out_dir(*args):
    return os.path.join(fixed.build_tracking_out_dir(*args), 'variable')


def save_final_result_csv(session_path, model_name, dataset_name, run_name,
                          video_name, buffers, num_objects, family):
    path = fixed.save_final_result_csv(session_path, model_name, dataset_name,
        run_name + '_variable', video_name, buffers, num_objects, family)
    if num_objects == 0:
        first, last = buffers['variable_frame_range']
        pd.DataFrame({'frame': range(first, last+1)}).to_csv(path, index=False)
    return path

def track_from_detection_frames(detection_frames, num_objects, nms_iou, max_age,
                                strong_iou, strong_dir_deg, tracking_cost=None,
                                sharp_turn_accept_run=3):
    """Associate only live tracks; an unmatched detection creates a new ID.

    NUM_OBJECTS is deliberately ignored. After MAX_AGE missing frames an ID
    retires permanently. Missing rows remain NaN, including before birth and
    after last observation. No backward pass, ungated recovery, or gap fill.
    """
    frames = sorted(detection_frames)
    if not frames:
        raise RuntimeError('No detection frames found.')
    max_age = int(max_age)
    if max_age < 1:
        raise ValueError('MAX_AGE must be at least 1.')
    tc = dict(tracking_cost or {})
    names = ('pos', 'obb', 'class', 'score', 'assign_type', 'assign_type_history',
             'assign_val', 'assign_total', 'assign_iou_cost', 'assign_direction_cost',
             'assign_distance_cost', 'assign_score_cost', 'obb_source',
             'obb_corrected', 'direction_corrected', 'switch_corrected')
    buffers = {name + '_buf': {} for name in names}
    tracks = {}
    next_id = 0
    previous_frame = frames[0] - 1
    for frame in tqdm_it(frames, desc='ID tracking (variable)', unit='frame'):
        obbs, scores, classes = detection_frames[frame]
        keep = []
        # Detection NMS is usually already done; keep confidence priority here.
        for d in np.argsort(-np.asarray(scores), kind='stable'):
            if not valid_obb(obbs[d]):
                continue
            if 0 < nms_iou < 1 and any(iou_obb(obbs[d], obbs[k]) > nms_iou for k in keep):
                continue
            keep.append(int(d))
        obbs = np.asarray([obbs[d] for d in keep], dtype=np.float32).reshape(-1, 4, 2)
        scores = np.asarray([scores[d] for d in keep])
        directions = np.asarray([unit_vec_to_angle_deg(closest_long_axis_direction(
            obbs[i], float(classes[d]))) for i, d in enumerate(keep)])
        # Expire before association, also accounting for absent frame keys.
        tracks = {tid: tr for tid, tr in tracks.items()
                  if frame - tr['last_frame'] - 1 < max_age}
        for tr in tracks.values():
            for _ in range(frame - previous_frame):
                tr['kf'].predict()
            shift = tr['kf'].x[:2].reshape(2) - obb_center(tr['obb'])
            tr['pred'] = tr['obb'] + shift
        assigned = {}
        used = set()
        active = sorted(tracks)
        for stage, strong, heading, mode in (
            (ATYPE_S1, True, True, 'heading'), (ATYPE_S2, True, False, 'axis'),
            (ATYPE_S3, False, True, 'heading'), (ATYPE_S4, False, False, 'axis')):
            ids = [tid for tid in active if tid not in assigned]
            dets = [d for d in range(len(obbs)) if d not in used]
            if not ids or not dets:
                continue
            matches, costs = assign_ids_iou_mode(
                [tracks[t]['pred'] for t in ids], [tracks[t]['obb'] for t in ids],
                [frame-tracks[t]['last_frame']-1 for t in ids], obbs[dets],
                directions[dets], [tracks[t]['direction'] for t in ids],
                strong_iou, strong_dir_deg, max_age, require_iou_gate=strong,
                require_dir_gate=heading, direction_cost_mode=mode,
                iou_weight=float(tc.get('IOU_WEIGHT', 1)),
                direction_weight=float(tc.get('DIRECTION_WEIGHT', 1)),
                miss_weight=float(tc.get('MISS_WEIGHT', 1)))
            for i, d in enumerate(matches):
                if d is not None:
                    assigned[ids[i]] = (dets[d], stage, float(costs[i][1]))
                    used.add(dets[d])
        # Distance recovery is allowed only on consecutive observations, with
        # the existing spatial/velocity gate. Missing tracks never use a
        # distance gate that expands indefinitely with their age.
        ids = [t for t in active if t not in assigned and frame-tracks[t]['last_frame']==1]
        dets = [d for d in range(len(obbs)) if d not in used]
        if ids and dets:
            matches, costs = assign_ids_gated_distance_mode(
                [tracks[t]['pred'] for t in ids], [tracks[t]['obb'] for t in ids],
                [float(np.linalg.norm(tracks[t]['kf'].x[2:4])) for t in ids],
                [0]*len(ids), obbs[dets], max_age,
                distance_weight=float(tc.get('DISTANCE_WEIGHT', 1)),
                miss_weight=float(tc.get('MISS_WEIGHT', 1)))
            for i, d in enumerate(matches):
                if d is not None:
                    assigned[ids[i]] = (dets[d], ATYPE_VEL_DIST, float(costs[i][1]))
                    used.add(dets[d])
        for d in range(len(obbs)):
            if d not in used:
                tid = next_id
                next_id += 1
                tracks[tid] = {'kf': create_position_kf(*obb_center(obbs[d]))}
                assigned[tid] = (d, ATYPE_INIT, np.nan)
        for tid, (d, stage, value) in assigned.items():
            tr = tracks[tid]
            pts = canonicalize_obb_points(obbs[d])
            center = obb_center(pts)
            if stage != ATYPE_INIT:
                tr['kf'].update(np.asarray(center, dtype=float))
            tr.update(obb=pts, direction=float(directions[d]), last_frame=frame)
            row = dict(pos=tuple(center), obb=obb_to_tuple(pts),
                **{'class': (float(directions[d]),)}, score=(float(scores[d]),),
                assign_type=(stage,), assign_type_history=[('variable', float(stage))],
                assign_val=(value,), assign_total=(value,), assign_iou_cost=(np.nan,),
                assign_direction_cost=(np.nan,), assign_distance_cost=(np.nan,),
                assign_score_cost=(np.nan,), obb_source=(OBB_SOURCE_DETECTED,),
                obb_corrected=(0.,), direction_corrected=(0.,), switch_corrected=(0.,))
            for name, val in row.items():
                buffers[name+'_buf'].setdefault(tid, {})[frame] = val
        previous_frame = frame
    # Boundary NaNs retain the complete video range without dense per-ID dicts.
    for tid in range(next_id):
        for frame in (frames[0], frames[-1]):
            for name in names:
                width = 2 if name == 'pos' else 8 if name == 'obb' else 1
                default = [] if name == 'assign_type_history' else (np.nan,)*width
                if name in ('obb_source','obb_corrected','direction_corrected','switch_corrected'):
                    default = (0.,)
                buffers[name+'_buf'].setdefault(tid, {}).setdefault(frame, default)
    buffers['variable_frame_range'] = (frames[0], frames[-1])
    return buffers


def save_tracking_outputs(out_dir, buffers, num_objects, **kwargs):
    fixed.save_tracking_outputs(out_dir, buffers, num_objects, **kwargs)
    if num_objects != 0:
        return
    artifact_suffix = kwargs.get('artifact_suffix', FWD_ARTIFACT_SUFFIX)
    if artifact_suffix == 'filled':
        tag = ''
    elif artifact_suffix == 'id_resolved':
        tag = '_id_resolved'
    else:
        return
    first, last = buffers['variable_frame_range']
    for base in ('obbs', 'directions'):
        pd.DataFrame({'position': range(first, last + 1)}).to_csv(
            os.path.join(out_dir, f'{base}{tag}.csv'), index=False,
        )


def _run_single_tracking_job(spec: dict) -> None:
    """Process one (video_info, tracking_job) pair. Designed to run in a subprocess."""
    info          = spec['info']
    run_name      = spec['run_name']
    session_path  = spec['session_path']
    model_name    = spec['model_name']
    dataset_name  = spec['dataset_name']
    num_objects   = spec['num_objects']
    conf_th       = spec['conf_th']
    nms_iou       = spec['nms_iou']
    strong_iou    = spec['strong_iou']
    strong_dir_deg = spec['strong_dir_deg']
    max_age       = spec['max_age']
    tracking_cost_cfg = spec['tracking_cost_cfg']
    sharp_turn_accept_run = spec['sharp_turn_accept_run']
    export_final_result = bool(spec.get('export_final_result', False))

    out_dir = build_tracking_out_dir(
        session_path, model_name, dataset_name, run_name, info['name']
    )
    direction_pickle = _artifact_path(out_dir, 'all_blobs.pkl')
    if not os.path.isfile(direction_pickle):
        raise FileNotFoundError(f'Direction blobs pickle not found: {direction_pickle}')

    direction_rows_df = load_blob_pickle(direction_pickle)
    detection_frames = build_detection_frames(
        direction_rows_df, info['first_frame'], info['last_frame'], conf_th)

    common_kw = dict(
        num_objects=num_objects,
        nms_iou=nms_iou,
        max_age=max_age,
        strong_iou=strong_iou,
        strong_dir_deg=strong_dir_deg,
        tracking_cost=tracking_cost_cfg,
        sharp_turn_accept_run=sharp_turn_accept_run,
    )

    print(f"Running variable-population tracking. [{run_name}:{info['name']}]")
    fwd_buffers = track_from_detection_frames(
        detection_frames=detection_frames,
        **common_kw,
    )
    num_objects = len(fwd_buffers['obb_buf'])
    os.makedirs(out_dir, exist_ok=True)
    # Remove only stale derived files in this dedicated variable-mode directory.
    import glob
    stale_patterns = {
        os.path.join(out_dir, BUFFER_DIR_NAME): (
            '*_filled.pkl', '*_id_resolved.pkl',
            'provenance_filled.csv', 'provenance_id_resolved.csv',
        ),
        os.path.join(out_dir, LOG_DIR_NAME): (
            'corrections_log*', 'embedding_vram_calib.json',
        ),
        out_dir: ('embedding.h5', 'embedding.pt', 'embedding_training_metrics.csv'),
    }
    for directory, patterns in stale_patterns.items():
        for pattern in patterns:
            for stale_path in glob.glob(os.path.join(directory, pattern)):
                os.remove(stale_path)
    save_tracking_outputs(out_dir, fwd_buffers, num_objects,
                          artifact_suffix=FWD_ARTIFACT_SUFFIX, save_csv=False)

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python multi_staged_association_variable.py config.yaml')
    cfg = load_config(sys.argv[1])
    if cfg.get('skip_id_tracking', False):
        print('skip_id_tracking is True; exiting.')
        return

    analysis = cfg.get('analysis', {}) or {}

    # -- Workers --
    workers_cfg = resolve_num_workers(cfg, 'id_tracking', default=None)
    num_workers = (
        _auto_num_workers('process')
        if workers_cfg is None
        else max(1, int(workers_cfg))
    )
    worker_mode = 'auto' if workers_cfg is None else 'configured'
    print(f'id_tracking: workers={num_workers} ({worker_mode})')

    # -- Paths & basic settings --
    session_path = str(cfg['SESSION_PATH'])
    video_path_in = str(cfg['TRACKING_VIDEO_PATH'])
    num_objects = 0

    # -- Tracking parameters --
    conf_th          = float(analysis.get('CONF', 0.1))
    orig_first_frame = int(analysis.get('FIRST_FRAME', 0))
    orig_last_frame  = int(analysis.get('LAST_FRAME', -1))

    nms_iou        = float(analysis.get('NMS_IOU', 0.80))
    strong_iou     = float(analysis.get('MATCH_IOU', 0.50))
    strong_dir_deg = float(analysis.get('MATCH_ANGLE',   90.0))
    max_age        = int(analysis.get('MAX_AGE', 10))
    tracking_cost_cfg = dict(analysis.get('TRACKING_COST', {}) or {})
    sharp_turn_accept_run = int(
        tracking_cost_cfg.get('SHARP_TURN_ACCEPT_RUN',
            analysis.get('SHARP_TURN_ACCEPT_RUN', 3))
    )

    # -- Model and checkpoint parameters --
    model_name = os.path.splitext(cfg.get('training', {}).get('PRETRAINED_MODEL', ''))[0]
    model_name = resolve_model_name(model_name).split('.')[0]
    dataset_name = resolve_existing_experiment_dir_name(
        session_path,
        model_name,
        experiment_dir_name_from_cfg(cfg),
        stages=('tracking', 'training'),
        warn_fn=print,
    )
    requested_direction_weights = list(dict.fromkeys(parse_weight_spec(analysis.get('WEIGHT', 'last'))))
    tracking_jobs = resolve_tracking_jobs_for_requested_weights(
        session_path,
        model_name,
        dataset_name,
        requested_direction_weights,
    )

    if os.path.isdir(video_path_in):
        video_files = [os.path.join(video_path_in, f) for f in os.listdir(video_path_in) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    else:
        video_files = [video_path_in]
    if not video_files:
        raise RuntimeError(f'Video not found: {video_path_in}')

    video_infos = []
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_info = read_video_frame_info(video_path)
        warn_if_frame_count_adjusted(frame_info, label=f'id_tracking video={video_name}')
        first_frame, last_frame = clamp_frame_range_to_usable_count(
            orig_first_frame, orig_last_frame, frame_info.usable_frame_count,
        )
        print(
            f'[INFO] id_tracking video={video_name}, usable_frames={frame_info.usable_frame_count}, '
            f'frame_range={first_frame}-{last_frame}'
        )
        video_infos.append({'path': video_path, 'name': video_name, 'first_frame': first_frame, 'last_frame': last_frame})

    specs = []
    for info in video_infos:
        for job in tracking_jobs:
            specs.append({
                'info': info,
                'run_name': str(job['run_name']),
                'session_path': session_path,
                'model_name': model_name,
                'dataset_name': dataset_name,
                'num_objects': num_objects,
                'conf_th': conf_th,
                'nms_iou': nms_iou,
                'strong_iou': strong_iou,
                'strong_dir_deg': strong_dir_deg,
                'max_age': max_age,
                'tracking_cost_cfg': tracking_cost_cfg,
                'sharp_turn_accept_run': sharp_turn_accept_run,
                'export_final_result': bool(cfg.get('skip_id_correction', False)),
            })

    if not specs:
        return

    n_epoch_jobs = len(specs)
    outer_epoch, _ = _worker_budget(num_workers, n_epoch_jobs)

    if outer_epoch <= 1:
        for spec in specs:
            _run_single_tracking_job(spec)
    else:
        print(
            f'id_tracking: {n_epoch_jobs} epoch job(s), '
            f'outer={outer_epoch} parallel'
        )
        with ProcessPoolExecutor(max_workers=outer_epoch) as pool:
            futures = [pool.submit(_run_single_tracking_job, spec) for spec in specs]
            for f in futures:
                f.result()

if __name__ == '__main__':
    main()
