"""Candidate navigation should show the start of each classification run."""

from copy import deepcopy

from gui.refinement_candidates import (
    build_candidate_runs,
    candidate_run_edit_state,
    visible_candidate_runs,
)


def candidate(frame, tid=0, code=10, short="DIR-FLIP", label="direction flipped"):
    return {
        "frame": frame,
        "tid": tid,
        "assign_code": code,
        "label_short": short,
        "label": label,
    }


def test_consecutive_candidates_keep_first_frame_and_original_rows():
    items = [candidate(frame) for frame in range(10, 16)]
    original = deepcopy(items)

    runs = build_candidate_runs(items)

    assert len(runs) == 1
    assert runs[0].first_index == 0
    assert (runs[0].start_frame, runs[0].end_frame) == (10, 15)
    assert runs[0].member_indices == tuple(range(6))
    assert items == original


def test_interleaved_ids_and_classifications_group_independently():
    items = [
        candidate(10, 0), candidate(10, 1), candidate(10, 0, 11, "DIR-FIX"),
        candidate(11, 0), candidate(11, 1), candidate(11, 0, 11, "DIR-FIX"),
    ]

    runs = build_candidate_runs(items)

    assert [(run.tid, run.assign_code, run.start_frame, run.end_frame) for run in runs] == [
        (0, 10, 10, 11), (0, 11, 10, 11), (1, 10, 10, 11),
    ]


def test_gap_and_classification_change_create_new_representatives():
    items = [
        candidate(1), candidate(2),
        candidate(3, code=11, short="DIR-FIX"),
        candidate(4), candidate(5), candidate(7),
    ]

    assert [run.start_frame for run in build_candidate_runs(items)] == [1, 3, 4, 7]


def test_displayed_provenance_tags_split_a_run_but_long_explanation_does_not():
    items = [
        candidate(1, label="one source"), candidate(2, label="another source"),
        candidate(3, short="DIR-FLIP|ID-FIX"),
        candidate(4, short="DIR-FLIP|ID-FIX"), candidate(5),
    ]

    assert [(run.start_frame, run.end_frame) for run in build_candidate_runs(items)] == [
        (1, 2), (3, 4), (5, 5),
    ]


def test_distinct_assignment_codes_do_not_merge_under_identical_display_labels():
    runs = build_candidate_runs([candidate(1, code=10), candidate(2, code=11)])

    assert len(runs) == 2


def test_same_frame_duplicates_are_one_distinct_frame_for_edit_status():
    runs = build_candidate_runs([candidate(2), candidate(2), candidate(3)])

    assert len(runs) == 1
    assert runs[0].member_indices == (0, 1, 2)
    assert candidate_run_edit_state(runs[0], {(2, 0)}) == "partial"
    assert candidate_run_edit_state(runs[0], {(2, 0), (3, 0)}) == "yes"


def test_unsorted_rows_keep_original_indices_and_earliest_frame():
    items = [candidate(5), candidate(3), candidate(4), candidate(3)]

    runs = build_candidate_runs(items)

    assert len(runs) == 1
    assert runs[0].first_index == 1
    assert runs[0].member_indices == (1, 3, 2, 0)
    assert (runs[0].start_frame, runs[0].end_frame) == (3, 5)


def test_hide_edited_keeps_partial_run_at_its_original_start():
    runs = build_candidate_runs([candidate(2), candidate(3), candidate(4)])

    visible = visible_candidate_runs(runs, {10}, {(2, 0)}, show_only_unedited=True)

    assert [run.start_frame for run in visible] == [2]
    assert candidate_run_edit_state(visible[0], {(2, 0)}) == "partial"
    assert visible_candidate_runs(
        runs, {10}, {(2, 0), (3, 0), (4, 0)}, show_only_unedited=True,
    ) == []


def test_type_filter_applies_to_complete_runs():
    items = [
        candidate(1), candidate(2),
        candidate(1, code=11, short="DIR-FIX"), candidate(2, code=11, short="DIR-FIX"),
    ]
    runs = build_candidate_runs(items)

    assert [run.first_index for run in visible_candidate_runs(runs, {11}, set())] == [2]
    assert visible_candidate_runs(runs, set(), set()) == []


def test_remapped_id_change_splits_run_at_the_switch_frame():
    items = [candidate(1, 0), candidate(2, 0), candidate(3, 1), candidate(4, 1)]

    assert [(run.tid, run.start_frame, run.end_frame) for run in build_candidate_runs(items)] == [
        (0, 1, 2), (1, 3, 4),
    ]


def test_edit_keys_for_other_ids_do_not_hide_run():
    runs = build_candidate_runs([candidate(1, 0), candidate(2, 0)])
    edited = {(1, 1), (2, 1)}

    assert candidate_run_edit_state(runs[0], edited) == ""
    assert visible_candidate_runs(runs, {10}, edited, show_only_unedited=True) == runs


def test_unresolved_ids_keep_each_source_identity_in_a_separate_run():
    items = [
        {**candidate(frame, -1), "source_tid": source_tid}
        for frame in (1, 2)
        for source_tid in (0, 1)
    ]

    runs = build_candidate_runs(items)

    assert len(runs) == 2
    assert [run.member_indices for run in runs] == [(0, 2), (1, 3)]
    assert [run.first_index for run in runs] == [0, 1]


def test_empty_candidates_have_no_runs():
    assert build_candidate_runs([]) == []
