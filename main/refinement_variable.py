# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Variable-population correction: shared helpers, no positional filling."""
from __future__ import annotations

from functools import partial
import refinement as fixed
from refinement import *
from refinement import _RESCUE_LOG_DEFAULTS
from refinement import (
    _apply_assignment_swaps,
    _artifact_path,
    _auto_correction_workers,
    _auto_img_size,
    _build_contact_episodes,
    _build_fragment_index,
    _cfg_bool,
    _correction_worker_budget,
    _eval_missing_frames_id,
    _localize_swap_seam,
    _mark_embedding_episode_frames,
    _relabel_fragments_after_assignment,
    _resolve_flip_frame_threshold,
    _resolve_pre_post_fragments,
    _run_id_eval,
    _seam_source_label,
    _select_embedding_target_episodes,
    _split_episode_by_local_non_s1_events,
    _track_end_for_ids,
)
from multi_staged_association_variable import build_tracking_out_dir, save_final_result_csv, save_tracking_outputs
from variable_population import closed_episode, safe_swap


def _resolve_contact_episodes_embedding(
    corrected: dict,
    num_objects: int,
    fragments: list,
    centroids: dict,
    episode_max_len: int,
    corrections_log: list[dict],
    min_interact_iou: float,
    progress_label: str = '',
) -> list[dict]:
    """Resolve high-risk N-body contact episodes with contrastive embedding assignments.

    Candidate episodes are limited to IoU>0 contact episodes that either reach
    INTERACT_IOU at least once or contain an interior missing/fill trigger.
    Episodes with roster <= EMBED_MAX_ROSTER use the standard path unchanged
    (INVARIANT: existing swap decisions are not affected by the oversized path).
    Episodes with roster > EMBED_MAX_ROSTER are locally decomposed into
    2..EMBED_MAX_ROSTER sub-rosters seeded by KF/overlap fill trigger events.
    """
    # Import the heavy embedding stack only on the enabled execution path.
    from identity_correction_variable import (
        assign_episode,
        _ASSIGN_MAX_ROSTER as embed_max_roster,
    )

    log: list[dict] = []
    progress_label = str(progress_label or '')

    _ = corrections_log  # retained in the public workflow signature

    fragment_index = _build_fragment_index(fragments)
    obb_cache: dict[tuple[int, int], np.ndarray | None] = {}
    valid_frames_cache: dict[int, list[int]] = {}

    print(f'  Building embedding contact episodes...{progress_label}', flush=True)
    all_episodes = [e for e in _build_contact_episodes(corrected, num_objects, episode_max_len) if closed_episode(corrected, e)]
    all_episodes.sort(key=lambda e: (int(e['start']), list(e['roster'])))
    episodes, skipped_low_risk = _select_embedding_target_episodes(
        corrected, all_episodes, obb_cache, valid_frames_cache, min_interact_iou,
    )
    print(
        f'  Built embedding contact episodes: total={len(all_episodes)}, '
        f'target={len(episodes)}, skipped_low_risk={skipped_low_risk}, '
        f'min_interact_iou={float(min_interact_iou):.3f}{progress_label}',
        flush=True,
    )
    resolved_regions: set[tuple] = set()

    _MAX_EPISODE_REBUILDS = 20
    rebuild_count = 0
    _ep_idx = 0
    while _ep_idx < len(episodes):
        episode = episodes[_ep_idx]
        _ep_idx += 1
        if _ep_idx == 1 or _ep_idx % 100 == 0 or _ep_idx == len(episodes):
            print(
                f'  Resolving embedding episodes: {_ep_idx}/{len(episodes)} '
                f'rebuild={rebuild_count}{progress_label}',
                flush=True,
            )
        roster = [int(tid) for tid in episode['roster']]
        start = int(episode['start'])
        end = int(episode['end'])

        region_key = (frozenset(roster), start, end)
        if region_key in resolved_regions:
            log.append({
                'type': 'embedding_episode',
                'ids': sorted(roster),
                'frame_start': start,
                'frame_end': end,
                'action': 'abstain',
                'reason': 'region_already_resolved',
            })
            continue

        episode_max_iou = float(episode.get('max_pairwise_iou', 0.0))
        episode_risk_reasons = list(episode.get('risk_reasons', []))
        episode_has_fill_or_missing = bool(episode.get('has_fill_or_missing_trigger', False))

        # Oversized roster: local sub-episode decomposition
        # INVARIANT: only entered when len(roster) > EMBED_MAX_ROSTER.
        # Episodes with len(roster) <= EMBED_MAX_ROSTER pass through the
        # standard path below without any modification.
        if len(roster) > embed_max_roster:
            parent_ids = sorted(roster)

            subs = _split_episode_by_local_non_s1_events(
                corrected, obb_cache, episode, embed_max_roster,
            )

            if not subs:
                log.append({
                    'type': 'embedding_episode',
                    'ids': parent_ids,
                    'frame_start': start,
                    'frame_end': end,
                    'action': 'abstain',
                    'reason': 'no_trigger_found',
                })
                continue

            viable = [s for s in subs if s['sub_roster'] and len(s['sub_roster']) >= 2]
            if not viable:
                log.append({
                    'type': 'embedding_episode',
                    'ids': parent_ids,
                    'frame_start': start,
                    'frame_end': end,
                    'action': 'abstain',
                    'reason': 'no_candidate_above_threshold',
                })
                continue

            # Deterministic processing order: (sub_start, sub_roster) ascending
            viable.sort(key=lambda s: (int(s['sub_start']), s['sub_roster']))
            sub_resolved: set[tuple] = set()

            for sub in viable:
                sub_roster = sub['sub_roster']
                sub_start = int(sub['sub_start'])
                sub_end = int(sub['sub_end'])
                trigger_id = int(sub['trigger_id'])

                sub_key = (frozenset(sub_roster), sub_start, sub_end)
                sub_entry: dict = {
                    'type': 'embedding_subepisode',
                    'parent_ids': parent_ids,
                    'parent_start': start,
                    'parent_end': end,
                    'trigger_id': trigger_id,
                    'trigger_frames': sub['trigger_frames'],
                    'trigger_atype': sub['trigger_atype'],
                    'candidate_ids': sub['candidate_ids'],
                    'sub_roster': sub_roster,
                    'sub_start': sub_start,
                    'sub_end': sub_end,
                    'pair_score': float(sub['pair_score']),
                    'candidate_overlap_scores': sub['candidate_overlap_scores'],
                    'selection_rule': sub['selection_rule'],
                    'episode_max_iou': float(episode_max_iou),
                    'min_interact_iou': float(min_interact_iou),
                    'risk_reasons': episode_risk_reasons,
                    'has_fill_or_missing_trigger': episode_has_fill_or_missing,
                    'action': 'abstain',
                    'reason': None,
                    'seam': None,
                    'assignment': None,
                    **_RESCUE_LOG_DEFAULTS,
                }

                if sub_key in sub_resolved or sub_key in resolved_regions:
                    sub_entry['reason'] = 'region_already_resolved'
                    log.append(sub_entry)
                    continue

                if len(sub_roster) > embed_max_roster:
                    sub_entry['reason'] = 'subroster_still_oversized'
                    log.append(sub_entry)
                    continue

                # Build pre_frags / post_frags for the sub-roster.
                # Pivot = sub_start (parent episode start) mirrors the standard path.
                # Both key sets must exactly match sub_roster for assign_episode.
                pre_frags, post_frags, frag_reason = _resolve_pre_post_fragments(
                    fragment_index, centroids, sub_roster, sub_start,
                )
                if frag_reason is not None:
                    sub_entry['reason'] = frag_reason
                    log.append(sub_entry)
                    continue

                decision = assign_episode(centroids, pre_frags, post_frags)
                sub_entry['best_cost'] = decision.best_cost
                sub_entry['identity_cost'] = decision.identity_cost
                sub_entry['second_cost'] = decision.second_cost
                sub_entry['max_matched_dist'] = decision.max_matched_dist


                if decision.abstained:
                    sub_entry['reason'] = decision.reason
                    log.append(sub_entry)
                    continue

                if not decision.is_swap:
                    # Embedding assignment ran to completion and positively
                    # confirmed the current identity mapping -- distinct from
                    # the abstain cases above, which never reached a reliable
                    # decision at all.
                    sub_entry['action'] = 'identity'
                    sub_entry['reason'] = decision.reason
                    log.append(sub_entry)
                    continue

                track_end = _track_end_for_ids(
                    corrected, obb_cache, valid_frames_cache, frozenset(sub_roster),
                )
                if track_end is None:
                    sub_entry['reason'] = 'missing_track_end'
                    log.append(sub_entry)
                    continue

                swapped_ids = [int(k) for k, v in decision.mapping.items() if int(k) != int(v)]
                if not swapped_ids:
                    sub_entry['reason'] = 'identity'
                    log.append(sub_entry)
                    continue

                # Seam localization uses sub_roster (not parent roster)
                seam = _localize_swap_seam(
                    corrected, obb_cache, swapped_ids, sub_roster, sub_start, sub_end,
                )
                if int(track_end) < int(seam):
                    sub_entry['reason'] = 'empty_apply_segment'
                    log.append(sub_entry)
                    continue

                seam_source = _seam_source_label(
                    corrected, obb_cache, swapped_ids, sub_roster, sub_start, sub_end, seam,
                )
                if not safe_swap(corrected, decision.mapping, int(seam), int(track_end)):
                    sub_entry['reason'] = 'variable_population_presence_change'
                    log.append(sub_entry)
                    continue
                _apply_assignment_swaps(corrected, decision.mapping, int(seam), int(track_end))
                obb_cache.clear()
                valid_frames_cache.clear()
                # Rebuild immediately so the next subepisode sees relabelled fragments.
                _relabel_fragments_after_assignment(
                    fragments, decision.mapping, int(seam), int(track_end),
                )
                fragment_index = _build_fragment_index(fragments)

                review_frames = {int(seam)}
                _mark_embedding_episode_frames(corrected, set(swapped_ids), review_frames)
                sub_resolved.add(sub_key)
                resolved_regions.add(sub_key)

                sub_entry['action'] = 'relabel'
                sub_entry['reason'] = 'local_kf_trigger_pair'
                sub_entry['seam'] = int(seam)
                sub_entry['seam_source'] = seam_source
                sub_entry['track_end'] = int(track_end)
                sub_entry['source_episode_start'] = start
                sub_entry['source_episode_end'] = end
                sub_entry['assignment'] = {
                    str(k): int(v) for k, v in sorted(decision.mapping.items())
                }
                sub_entry['swapped_ids'] = sorted(int(tid) for tid in swapped_ids)
                sub_entry['review_frames'] = sorted(int(f) for f in review_frames)
                log.append(sub_entry)
                if rebuild_count < _MAX_EPISODE_REBUILDS:
                    rebuild_count += 1
                    print(
                        f'  Rebuilding embedding contact episodes: '
                        f'{rebuild_count}/{_MAX_EPISODE_REBUILDS}{progress_label}',
                        flush=True,
                    )
                    all_episodes = [e for e in _build_contact_episodes(corrected, num_objects, episode_max_len) if closed_episode(corrected, e)]
                    all_episodes.sort(key=lambda e: (int(e['start']), list(e['roster'])))
                    episodes, skipped_low_risk = _select_embedding_target_episodes(
                        corrected, all_episodes, obb_cache, valid_frames_cache, min_interact_iou,
                    )
                    print(
                        f'  Rebuilt embedding contact episodes: total={len(all_episodes)}, '
                        f'target={len(episodes)}, skipped_low_risk={skipped_low_risk}{progress_label}',
                        flush=True,
                    )
                    log.append({
                        'type': 'episode_rebuild',
                        'rebuild_count': rebuild_count,
                        'trigger_seam': int(seam),
                        'trigger_track_end': int(track_end),
                        'prev_episode_start': start,
                        'prev_episode_end': end,
                        'new_episode_count': len(episodes),
                        'new_total_episode_count': len(all_episodes),
                        'skipped_low_risk': int(skipped_low_risk),
                        'min_interact_iou': float(min_interact_iou),
                    })
                    _ep_idx = 0
                    break  # exit sub-loop; outer while restarts from episode 0

            continue  # done with this oversized episode

        # Standard path for roster <= EMBED_MAX_ROSTER (unchanged)
        entry: dict = {
            'type': 'embedding_episode',
            'ids': sorted(roster),
            'frame_start': start,
            'frame_end': end,
            'episode_max_iou': float(episode_max_iou),
            'min_interact_iou': float(min_interact_iou),
            'risk_reasons': episode_risk_reasons,
            'has_fill_or_missing_trigger': episode_has_fill_or_missing,
            'action': 'abstain',
            'reason': None,
            **_RESCUE_LOG_DEFAULTS,
        }

        pre_frags, post_frags, frag_reason = _resolve_pre_post_fragments(
            fragment_index, centroids, roster, start,
        )
        if frag_reason is not None:
            entry['reason'] = frag_reason
            log.append(entry)
            continue

        decision = assign_episode(centroids, pre_frags, post_frags)
        entry['best_cost'] = decision.best_cost
        entry['identity_cost'] = decision.identity_cost
        entry['second_cost'] = decision.second_cost
        entry['max_matched_dist'] = decision.max_matched_dist


        if decision.abstained:
            entry['reason'] = decision.reason
            log.append(entry)
            continue

        if not decision.is_swap:
            # Embedding assignment ran to completion and positively confirmed
            # the current identity mapping -- distinct from the abstain cases
            # above, which never reached a reliable decision at all.
            entry['action'] = 'identity'
            entry['reason'] = decision.reason
            log.append(entry)
            continue

        track_end = _track_end_for_ids(
            corrected, obb_cache, valid_frames_cache, frozenset(roster),
        )
        if track_end is None:
            entry['reason'] = 'missing_track_end'
            log.append(entry)
            continue

        swapped_ids = [int(k) for k, v in decision.mapping.items() if int(k) != int(v)]
        if not swapped_ids:
            entry['reason'] = 'identity'
            log.append(entry)
            continue

        seam = _localize_swap_seam(
            corrected, obb_cache, swapped_ids, roster, start, end,
        )
        if int(track_end) < int(seam):
            entry['reason'] = 'empty_apply_segment'
            log.append(entry)
            continue

        seam_source = _seam_source_label(
            corrected, obb_cache, swapped_ids, roster, start, end, seam,
        )
        if not safe_swap(corrected, decision.mapping, int(seam), int(track_end)):
            entry['reason'] = 'variable_population_presence_change'
            log.append(entry)
            continue
        _apply_assignment_swaps(corrected, decision.mapping, int(seam), int(track_end))
        obb_cache.clear()
        valid_frames_cache.clear()
        _relabel_fragments_after_assignment(fragments, decision.mapping, int(seam), int(track_end))
        fragment_index = _build_fragment_index(fragments)

        review_frames = {int(seam)}
        _mark_embedding_episode_frames(corrected, set(swapped_ids), review_frames)
        resolved_regions.add(region_key)

        entry['action'] = 'relabel'
        entry['reason'] = 'swap'
        entry['seam'] = int(seam)
        entry['seam_source'] = seam_source
        entry['track_end'] = int(track_end)
        entry['source_episode_start'] = start
        entry['source_episode_end'] = end
        entry['assignment'] = {str(k): int(v) for k, v in sorted(decision.mapping.items())}
        entry['swapped_ids'] = sorted(int(tid) for tid in swapped_ids)
        entry['review_frames'] = sorted(int(frame) for frame in review_frames)
        log.append(entry)
        if rebuild_count < _MAX_EPISODE_REBUILDS:
            rebuild_count += 1
            print(
                f'  Rebuilding embedding contact episodes: '
                f'{rebuild_count}/{_MAX_EPISODE_REBUILDS}{progress_label}',
                flush=True,
            )
            all_episodes = [e for e in _build_contact_episodes(corrected, num_objects, episode_max_len) if closed_episode(corrected, e)]
            all_episodes.sort(key=lambda e: (int(e['start']), list(e['roster'])))
            episodes, skipped_low_risk = _select_embedding_target_episodes(
                corrected, all_episodes, obb_cache, valid_frames_cache, min_interact_iou,
            )
            print(
                f'  Rebuilt embedding contact episodes: total={len(all_episodes)}, '
                f'target={len(episodes)}, skipped_low_risk={skipped_low_risk}{progress_label}',
                flush=True,
            )
            log.append({
                'type': 'episode_rebuild',
                'rebuild_count': rebuild_count,
                'trigger_seam': int(seam),
                'trigger_track_end': int(track_end),
                'prev_episode_start': start,
                'prev_episode_end': end,
                'new_episode_count': len(episodes),
                'new_total_episode_count': len(all_episodes),
                'skipped_low_risk': int(skipped_low_risk),
                'min_interact_iou': float(min_interact_iou),
            })
            _ep_idx = 0

    # Abstained/identity-confirmed episodes do not overwrite the frame's
    # assignment type, but they are still correction decisions. Persist one
    # review-frame history entry so gui_refinement shows the same outcome in
    # Candidate and History. action=='identity' (embedding ran to completion
    # and positively confirmed no swap) and action=='abstain' (no reliable
    # decision reached at all) are recorded with different codes so they stay
    # distinguishable downstream -- only action=='relabel' (handled earlier,
    # via _mark_embedding_episode_frames) actually changes assign_type_buf.
    for entry in log:
        action = entry.get('action')
        if action == 'identity':
            marker_code = ATYPE_EMBED_IDENTITY
            history_source = 'correction_embedding_identity'
        elif action == 'abstain':
            marker_code = ATYPE_EMBED_ABSTAIN
            history_source = 'correction_embedding_abstain'
        else:
            continue
        if entry.get('type') == 'embedding_subepisode':
            episode_ids = entry['sub_roster']
            episode_start = int(entry['sub_start'])
        elif entry.get('type') == 'embedding_episode':
            episode_ids = entry['ids']
            episode_start = int(entry['frame_start'])
        else:
            continue
        review_frames = entry.get('review_frames')
        if not isinstance(review_frames, list):
            review_frames = [episode_start]
            entry['review_frames'] = review_frames
        for tid in episode_ids:
            for frame in review_frames:
                tid_i = int(tid)
                frame_i = int(frame)
                if get_obb(corrected, tid_i, frame_i) is None:
                    continue
                history = (
                    corrected.setdefault('assign_type_history_buf', {})
                    .setdefault(tid_i, {})
                    .setdefault(frame_i, [])
                )
                marker = (history_source, float(marker_code))
                if marker not in history:
                    history.append(marker)

    return log

def _mark_missing_frames(
    corrected: dict,
    num_objects: int,
    num_workers: int = 1,
    *,
    concurrent_jobs: int = 1,
) -> list[dict]:
    """Assign ATYPE_MISSING to interior gaps in each track's assign_type_buf.

    This inexpensive final scan remains serial to avoid another Windows process
    startup after the geometry pools. Writes are applied in deterministic order.
    ``num_workers`` and ``concurrent_jobs`` remain accepted for call compatibility.
    """

    missing_by_tid = _run_id_eval(
        corrected,
        num_objects,
        1,
        desc='Variable correction: mark missing frames',
        serial_fn=_eval_missing_frames_id,
        buffer_names=('obb_buf',),
    )

    log: list[dict] = []
    for tid, frames in enumerate(missing_by_tid):
        for frame in frames:
            corrected.setdefault('assign_type_buf', {}).setdefault(tid, {})[frame] = (float(ATYPE_MISSING),)
            corrected.setdefault('assign_type_history_buf', {}).setdefault(tid, {}).setdefault(frame, []).append(
                ('correction_missing', float(ATYPE_MISSING))
            )
            log.append({'type': 'missing', 'ids': [tid], 'frame': frame, 'action': 'mark_missing'})

    print(
        f'Variable correction: mark missing frames: num corrections={len(log)}',
        flush=True,
    )
    return log

def _run_single_correction_job(spec: dict) -> None:
    """Validate a variable-population result without filling missing animals."""
    out_dir          = spec['out_dir']
    session_path     = spec['session_path']
    model_name       = spec['model_name']
    dataset_name     = spec['dataset_name']
    run_name         = spec['run_name']
    video_name       = spec['video_name']
    num_objects      = spec['num_objects']
    strong_iou       = spec['strong_iou']
    strong_dir_deg   = spec['strong_dir_deg']
    max_age          = spec['max_age']
    flip_frame_threshold = int(spec['flip_frame_threshold'])
    flip_duration_sec = float(spec['flip_duration_sec'])
    flip_source_fps = float(spec['source_fps'])
    max_axis_error_deg = spec.get('max_axis_error_deg', 45.0)
    num_workers      = spec['num_workers']
    video_path       = spec.get('video_path', '')
    embedding_seed   = normalize_seed(spec.get('embedding_seed', 0))
    embedding_cfg    = spec.get('embedding', {}) or {}
    background_path  = str(spec.get('background_path') or embedding_cfg.get('BACKGROUND_PATH', '')).strip()
    emb_enable       = _cfg_bool(
        spec.get('embedding_enabled', embedding_cfg.get('ENABLE', False))
    )
    _img_size_raw    = embedding_cfg.get('IMG_SIZE', 'auto')
    embedding_device = embedding_cfg.get('DEVICE', 'auto')
    preview_count    = int(embedding_cfg.get('PREVIEW_COUNT', 100))
    episode_max_len  = int(embedding_cfg.get('EPISODE_MAX_LEN', 0))
    min_interact_iou = FIXED_INTERACT_IOU
    concurrent_jobs = max(1, int(spec.get('concurrent_correction_jobs', 1)))

    dataset_name, out_dir = _resolve_existing_correction_out_dir(spec)
    spec = {**spec, 'dataset_name': dataset_name, 'out_dir': out_dir}

    observed = load_tracking_buffers(out_dir, FWD_ARTIFACT_SUFFIX)
    num_objects = len(observed['obb_buf'])
    spec = {**spec, 'num_objects': num_objects}
    id_resolved_req = required_id_resolved_outputs(out_dir)
    filled_req = required_filled_outputs(out_dir)
    if emb_enable:
        if all(output_has_data(kind, p) for kind, p in id_resolved_req):
            print(f'ID-resolved outputs already exist. Skipping. [{run_name}:{video_name}]')
            save_existing_corrected_final_result(spec, ID_RESOLVED_ARTIFACT_SUFFIX)
            return
        if all(output_has_data(kind, p) for kind, p in filled_req):
            print(f'Filled outputs already exist. Skipping. [{run_name}:{video_name}]')
            save_existing_corrected_final_result(spec, FILLED_ARTIFACT_SUFFIX)
            return
    else:
        if all(output_has_data(kind, p) for kind, p in filled_req):
            print(f'Filled outputs already exist. Skipping. [{run_name}:{video_name}]')
            save_existing_corrected_final_result(spec, FILLED_ARTIFACT_SUFFIX)
            return

    fwd_pkl = _artifact_path(out_dir, BUFFERS_PICKLE_NAME, FWD_ARTIFACT_SUFFIX)
    if not os.path.exists(fwd_pkl):
        print(f'fwd buffers not found; skipping. [{run_name}:{video_name}]')
        return

    print(f'Correcting [{run_name}:{video_name}]...')
    print(
        f'  Flip anchor threshold: {flip_frame_threshold} frames '
        f'({flip_duration_sec:.3g}s at {flip_source_fps:.3f} fps). '
        f'[{run_name}:{video_name}]',
        flush=True,
    )
    corrected = observed
    corrections_log: list[dict] = []
    embedding_log: list[dict] = []
    embedding_resolver_ran = False

    if emb_enable and num_objects >= 2:
        from identity_correction_variable import (
            build_fragments,
            load_cached_crop_metadata,
            load_cached_embedding_artifacts,
            load_or_train_embedding,
            embedding_available,
            EmbeddingSegmentationRequired,
        )

        fragments: list = []
        centroids: dict = {}
        best_ss = 0.0
        segmentation_missing = False

        print(f'  [SEED] embedding video={video_name} run={run_name} seed={embedding_seed}')
        try:
            cached_artifacts = load_cached_embedding_artifacts(
                out_dir, background_path, num_objects, seed=embedding_seed,
                device_config=embedding_device,
                session_path=session_path, video_path=video_path,
            )
            if cached_artifacts is not None:
                fragments, centroids, best_ss = cached_artifacts
                print(f'  Reusing embedding artifacts. [{run_name}:{video_name}]')
            else:
                cached_embedding = load_cached_crop_metadata(out_dir, background_path)
                if cached_embedding is not None:
                    fragments, emb_img_size = cached_embedding
                    print(
                        f'  Reusing embedding crop cache metadata; img_size={emb_img_size}. '
                        f'[{run_name}:{video_name}]'
                    )
                else:
                    fragments = build_fragments(corrected, num_objects, num_workers=num_workers)
                    emb_img_size = (
                        _auto_img_size(corrected, fragments)
                        if str(_img_size_raw).strip().lower() == 'auto'
                        else max(32, ((int(_img_size_raw) + 31) // 32) * 32)
                    )
                _, centroids, best_ss = load_or_train_embedding(
                    fragments, corrected, video_path, out_dir, num_objects, emb_img_size,
                    background_path,
                    preview_count=preview_count,
                    num_workers=num_workers,
                    seed=embedding_seed,
                    device_config=embedding_device,
                    session_path=session_path,
                )
        except EmbeddingSegmentationRequired as exc:
            segmentation_missing = True
            centroids = {}
            print(
                f'  {exc} Aborting embedding-based ID resolution; '
                f'continuing with geometry-only post-processing. [{run_name}:{video_name}]',
                flush=True,
            )

        if not segmentation_missing:
            if embedding_available(best_ss) and best_ss >= 0.91:
                print(
                    f'  Embedding SS={best_ss:.3f} >= 0.91; '
                    f'embedding corrections active. [{run_name}:{video_name}]'
                )
            elif embedding_available(best_ss):
                print(
                    f'  Embedding SS={best_ss:.3f} < 0.91; '
                    f'using best checkpoint for embedding corrections. [{run_name}:{video_name}]',
                    flush=True,
                )
            else:
                centroids = {}
                print(
                    f'  Embedding SS={best_ss:.3f} is not usable; '
                    f'continuing geometry correction without embedding relabels. '
                    f'[{run_name}:{video_name}]',
                    flush=True,
                )

        if centroids:
            embedding_resolver_ran = True
            embedding_log = _resolve_contact_episodes_embedding(
                corrected, num_objects, fragments, centroids, episode_max_len,
                corrections_log, min_interact_iou,
                progress_label=f' [{run_name}:{video_name}]',
            )
            embedding_corrections = sum(
                1 for entry in embedding_log if entry.get('action') == 'relabel'
            )
            print(
                f'  Finished embedding contact episode resolution; '
                f'num corrections={embedding_corrections}; '
                f'decisions={len(embedding_log)}. [{run_name}:{video_name}]',
                flush=True,
            )
            corrections_log += embedding_log

    # Geometry correction is intentionally applied only after embedding when
    # embedding is enabled.  When embedding is disabled, this is the ordinary
    # single correction pass from fwd.
    geometry_phase = 'post_embedding' if emb_enable else 'no_embedding'

    # Open population: never fill positions or enforce a conserved roster.
    kf_log = []
    overlap_log = []

    missing_log = _mark_missing_frames(
        corrected,
        num_objects,
        num_workers,
        concurrent_jobs=concurrent_jobs,
    )
    corrections_log += missing_log
    for item in missing_log:
        for tid in item['ids']:
            for name in ('obb_source_buf', 'obb_corrected_buf', 'direction_corrected_buf', 'switch_corrected_buf'):
                corrected[name].setdefault(tid, {})[int(item['frame'])] = (0.0,)

    kf_filled_gaps   = sum(1 for e in kf_log if e.get('action') == 'fill')
    kf_filled_frames = sum(e.get('gap_len', 0) for e in kf_log if e.get('action') == 'fill')
    kf_rejected_gaps = sum(1 for e in kf_log if e.get('action', '').startswith('reject'))
    ov_filled_gaps   = sum(1 for e in overlap_log if e.get('action') == 'fill')
    ov_filled_frames = sum(len(e.get('filled_frames', [])) for e in overlap_log if e.get('action') == 'fill')
    emb_relabels     = sum(1 for e in embedding_log if e.get('action') == 'relabel')
    emb_identity     = sum(1 for e in embedding_log if e.get('action') == 'identity')
    emb_abstains     = sum(1 for e in embedding_log if e.get('action') == 'abstain')
    missing_frames_n = len(missing_log)

    fill_summary = (
        f'KF fill: {kf_filled_gaps} gaps, {kf_filled_frames} frames, '
        f'{kf_rejected_gaps} rejected. Overlap fill: {ov_filled_gaps} gaps, '
        f'{ov_filled_frames} frames.'
    )
    embedding_summary = (
        f'Embedding episodes: {emb_relabels} relabel, {emb_identity} identity, {emb_abstains} abstain.'
        if emb_enable
        else 'Embedding: disabled.'
    )
    print(
        f'{embedding_summary} {fill_summary} '
        f'ATYPE_MISSING: {missing_frames_n}. [{run_name}:{video_name}]'
    )

    output_suffix = (
        ID_RESOLVED_ARTIFACT_SUFFIX
        if embedding_resolver_ran
        else FILLED_ARTIFACT_SUFFIX
    )
    print(
        f'Saving corrected outputs as {output_suffix}... '
        f'[{run_name}:{video_name}]',
        flush=True,
    )
    save_corrected_outputs(
        out_dir,
        corrected,
        num_objects,
        corrections_log=corrections_log,
        artifact_suffix=output_suffix,
    )
    save_final_result_csv(
        session_path,
        model_name,
        dataset_name,
        run_name,
        video_name,
        corrected,
        num_objects,
        output_suffix,
    )
    # 'identity' (embedding positively confirmed no swap was needed) is a
    # completed decision, not an applied change -- excluded here the same
    # way 'abstain' is.
    applied = sum(1 for e in corrections_log if e.get('action') not in ('abstain', 'identity'))
    print(f'Correction done. {applied} applied ({len(corrections_log)} decisions). [{run_name}:{video_name}]')

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python refinement_variable.py config.yaml')
    cfg = load_config(sys.argv[1])
    if cfg.get('skip_refinement', False):
        print('skip_refinement is True; exiting.')
        return

    analysis       = cfg.get('analysis', {}) or {}
    strong_iou     = float(analysis.get('MATCH_IOU', 0.50))
    strong_dir_deg = float(analysis.get('MATCH_ANGLE',   90.0))
    max_age        = int(analysis.get('MAX_AGE', 10))
    max_axis_error_deg = float(analysis.get('MAX_AXIS_ERR', 45.0))
    workers_cfg = resolve_num_workers(cfg, 'correction', default=None)
    if workers_cfg is None:
        workers_cfg = resolve_num_workers(cfg, 'id_correction', default=None)
    num_workers = (
        _auto_correction_workers()
        if workers_cfg is None
        else max(1, int(workers_cfg))
    )
    worker_mode = 'auto' if workers_cfg is None else 'configured'
    print(f'correction: workers={num_workers} ({worker_mode})')
    print('correction: RAM worker caps are adjusted per correction data.')
    embedding_cfg  = dict(cfg.get('EMBEDDING', {}) or {})
    embedding_enabled = _cfg_bool(embedding_cfg.get('ENABLE', False))

    session_path  = str(cfg['SESSION_PATH'])
    num_objects   = 0  # inferred separately for each video
    video_path_in = str(cfg['TRACKING_VIDEO_PATH'])
    background_path = str(cfg.get('BACKGROUND_PATH', '')).strip()
    master_seed = normalize_seed(cfg.get('RANDOM_SEED', 0))
    print(f'[SEED] correction master={master_seed}')

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
        video_names = [
            os.path.splitext(f)[0]
            for f in os.listdir(video_path_in)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        ]
        video_paths = {
            os.path.splitext(f)[0]: os.path.join(video_path_in, f)
            for f in os.listdir(video_path_in)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        }
    else:
        video_names = [os.path.splitext(os.path.basename(video_path_in))[0]]
        video_paths = {video_names[0]: video_path_in}
    if not video_names:
        raise RuntimeError(f'No video files found at: {video_path_in}')

    specs = []
    for video_name in video_names:
        video_path = video_paths.get(video_name, '')
        flip_frame_threshold, flip_duration_sec, source_fps = _resolve_flip_frame_threshold(
            analysis, video_path,
        )
        for job in tracking_jobs:
            run_name = str(job['run_name'])
            out_dir = build_tracking_out_dir(session_path, model_name, dataset_name, run_name, video_name)
            specs.append({
                'out_dir': out_dir,
                'session_path': session_path,
                'model_name': model_name,
                'dataset_name': dataset_name,
                'run_name': run_name,
                'video_name': video_name,
                'video_path': video_path,
                'background_path': background_path,
                'num_objects': num_objects,
                'strong_iou': strong_iou,
                'strong_dir_deg': strong_dir_deg,
                'max_age': max_age,
                'flip_frame_threshold': flip_frame_threshold,
                'flip_duration_sec': flip_duration_sec,
                'source_fps': source_fps,
                'max_axis_error_deg': max_axis_error_deg,
                'num_workers': num_workers,
                'embedding': embedding_cfg,
                'embedding_enabled': embedding_enabled,
                'random_seed': derive_seed(master_seed, 'correction', video_name, run_name),
                'embedding_seed': derive_seed(master_seed, 'correction_embedding', video_name, run_name),
            })

    if not specs:
        return

    if embedding_enabled:
        # Embedding can use CUDA or CPU; keep it single-job to avoid concurrent
        # contrastive training jobs exhausting VRAM/RAM. No pre-embedding filled
        # phase is run: geometry correction happens after embedding.
        embedding_specs = [
            {**spec, 'phase': 'embedding', 'num_workers': num_workers}
            for spec in specs
        ]
        _run_correction_specs(
            embedding_specs,
            1,
            num_workers,
            'correction embedding-first phase',
        )
    else:
        outer_epoch, workers_per_epoch = _correction_worker_budget(
            num_workers,
            specs,
        )
        for spec in specs:
            spec['num_workers'] = workers_per_epoch
        _run_correction_specs(specs, outer_epoch, workers_per_epoch, 'correction')

# Bind the output helpers and job runner explicitly. Imported functions keep
# their defining module's globals, so rebinding local names alone is insufficient.
save_corrected_outputs = partial(fixed.save_corrected_outputs, save_outputs=save_tracking_outputs)
save_existing_corrected_final_result = partial(
    fixed.save_existing_corrected_final_result, save_final=save_final_result_csv)
_resolve_existing_correction_out_dir = partial(
    fixed._resolve_existing_correction_out_dir, path_builder=build_tracking_out_dir)
_run_correction_specs = partial(fixed._run_correction_specs, run_job=_run_single_correction_job)

if __name__ == '__main__':
    main()
