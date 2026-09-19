# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Shared assign-type codes and labels for tracking/correction/refinement.

multi_staged_association.py labels are exposed as S0-S6 in the paper and GUI:
  S0 - start-frame initialization
  S1 - strong IoU + heading-gated assignment
  S2 - strong IoU + axis-cost assignment
  S3 - loose IoU (IoU > 0) + heading-gated assignment
  S4 - loose IoU (IoU > 0) + axis-cost assignment
  S5 - fast-distance recovery of an active track after IoU stages fail
  S6 - long-gap recovery of a dormant track

refinement.py uses its own C1-C5 sequence after forward tracking:
  C1 - KF gap fill
  C2 - overlap gap fill
  C3 - final embedding identity resolution
  C4 - position-spike, direction-spike, and segment-flip repair
  C5 - missing-frame marking
"""

# Group 1: multi_staged_association.py paper/GUI stages.
ATYPE_INIT        =  0  # S0 - start-frame initialization
ATYPE_S1          =  1  # S1 - strong IoU + heading gate
ATYPE_S2          =  2  # S2 - strong IoU + axis cost
ATYPE_S3          =  3  # S3 - loose IoU (IoU > 0) + heading gate
ATYPE_S4          =  4  # S4 - loose IoU (IoU > 0) + axis cost

# Group 1b: multi_staged_association.py S5/S6 recovery and auxiliary direction states.
ATYPE_VEL_DIST    =  6  # S5 - active-track fast-distance recovery (IoU = 0)
ATYPE_BKFILL_DET  =  7  # Retroactive backfill: detected OBB matched in gap frame
ATYPE_GAP_DET     = 11  # S6 - dormant-track long-gap recovery by main detection
ATYPE_DIR_NAN     = 13  # Direction withheld (unreliable tracking stage; resolved by Viterbi)
ATYPE_DIR_FIX     = 14  # Direction changed by tracking repair or C4 direction-spike fix
ATYPE_VITERBI     = 15  # Direction corrected by Viterbi post-filter
ATYPE_TURN_ACCEPT = 16  # Direction committed after sharp-turn run acceptance
ATYPE_DIR_FLIP    = 17  # Direction flipped 180 degrees by tracking repair or C4 segment flip

# Group 2: refinement.py.
ATYPE_KF_FILL       = 20  # C1 - KF-gated linear-chord interpolation
ATYPE_OVERLAP_FILL  = 21  # C2 - overlapping ID copied into an interior gap
# C3 embedding correction outcomes are mutually exclusive at the point of
# decision (see assign_episode/refinement.py): ATYPE_EMBED_ABSTAIN is a
# targeted episode/sub-episode for which no reliable decision was reached
# (gate_abs_dist/gate_cost_ratio, missing fragments/centroids,
# invalid roster, or a swap decided but not applicable); ATYPE_EMBED_IDENTITY
# is only ever assigned when embedding assignment ran to completion and
# positively confirmed the current identity mapping (no swap); ATYPE_EMBED_RELABEL
# is only assigned when an ID swap was actually applied.
ATYPE_EMBED_ABSTAIN  = 22  # C3 - embedding targeted but no reliable decision (abstained)
ATYPE_EMBED_RELABEL  = 23  # C3 - embedding episode relabelled (ID swap applied)
ATYPE_MISSING        = 24  # C5 - unfilled interior frame for manual review
ATYPE_POS_FIX        = 25  # C4 - position spike corrected by OBB interpolation
ATYPE_EMBED_IDENTITY = 26  # C3 - embedding confirmed current identity (no swap)

# Group 3: GUI / post-processing (gui_refinement.py).
ATYPE_PRE_FILL    = 30  # Pre-correction interpolation
ATYPE_POST_FILL   = 31  # Post-correction fill
ATYPE_ANCHOR      = 32  # Anchor consistency filter
ATYPE_SPIKE       = 33  # Direction spike removed by post-filter
ATYPE_FILTER_FILL = 34  # OBB/direction filled by post-filter interpolation
ATYPE_ID_SWAP     = 35  # Long ID swap corrected by post-filter

ASSIGN_CODE_ORDER = (
    ATYPE_INIT,
    ATYPE_S1,
    ATYPE_S2,
    ATYPE_S3,
    ATYPE_S4,
    ATYPE_VEL_DIST,
    ATYPE_GAP_DET,
    ATYPE_BKFILL_DET,
    ATYPE_DIR_NAN,
    ATYPE_DIR_FIX,
    ATYPE_DIR_FLIP,
    ATYPE_VITERBI,
    ATYPE_TURN_ACCEPT,
    ATYPE_KF_FILL,
    ATYPE_OVERLAP_FILL,
    ATYPE_EMBED_ABSTAIN,
    ATYPE_EMBED_IDENTITY,
    ATYPE_EMBED_RELABEL,
    ATYPE_MISSING,
    ATYPE_POS_FIX,
    ATYPE_PRE_FILL,
    ATYPE_POST_FILL,
    ATYPE_ANCHOR,
    ATYPE_SPIKE,
    ATYPE_FILTER_FILL,
    ATYPE_ID_SWAP,
)

ASSIGN_CODE_LABELS = {
    ATYPE_INIT:        "S0 - start-frame initialization",
    ATYPE_S1:          "S1 - strong IoU + heading-gated assignment",
    ATYPE_S2:          "S2 - strong IoU + axis-cost assignment",
    ATYPE_S3:          "S3 - loose IoU (IoU > 0) + heading-gated assignment",
    ATYPE_S4:          "S4 - loose IoU (IoU > 0) + axis-cost assignment",
    ATYPE_VEL_DIST:    "S5 - fast-distance active-track recovery (IoU = 0)",
    ATYPE_GAP_DET:     "S6 - long-gap dormant-track recovery by main detection",
    ATYPE_BKFILL_DET:  "Backfill - retroactive detected-OBB match in a gap frame",
    ATYPE_DIR_NAN:     "Direction withheld for Viterbi resolution",
    ATYPE_DIR_FIX:     "Direction corrected (tracking repair or C4 direction-spike fix)",
    ATYPE_DIR_FLIP:    "Direction flipped 180 degrees (tracking repair or C4 segment flip)",
    ATYPE_VITERBI:     "Direction corrected by Viterbi post-filter",
    ATYPE_TURN_ACCEPT: "Direction committed after sharp-turn run acceptance",
    ATYPE_KF_FILL:     "C1 - OBB filled by KF-gated interpolation",
    ATYPE_OVERLAP_FILL: "C2 - OBB copied from overlapping ID to fill a gap",
    ATYPE_EMBED_ABSTAIN: "C3 - embedding targeted but no reliable decision (abstained)",
    ATYPE_EMBED_IDENTITY: "C3 - embedding confirmed current identity (no swap)",
    ATYPE_EMBED_RELABEL: "C3 - embedding contact episode relabelled (ID swap applied)",
    ATYPE_MISSING:     "C5 - OBB still missing after correction; manual review",
    ATYPE_POS_FIX:     "C4 - position spike corrected by OBB interpolation",
    ATYPE_PRE_FILL:    "GUI pre-correction interpolation",
    ATYPE_POST_FILL:   "GUI post-correction fill",
    ATYPE_ANCHOR:      "GUI direction corrected by anchor consistency filter",
    ATYPE_SPIKE:       "GUI short direction spike removed by post-filter",
    ATYPE_FILTER_FILL: "GUI OBB/direction filled by post-filter interpolation",
    ATYPE_ID_SWAP:     "GUI long ID swap corrected by post-filter",
}

ASSIGN_CODE_SHORT_LABELS = {
    ATYPE_INIT:        "S0",
    ATYPE_S1:          "S1",
    ATYPE_S2:          "S2",
    ATYPE_S3:          "S3",
    ATYPE_S4:          "S4",
    ATYPE_VEL_DIST:    "S5",
    ATYPE_GAP_DET:     "S6",
    ATYPE_BKFILL_DET:  "Backfill",
    ATYPE_DIR_NAN:     "Dir-NaN",
    ATYPE_DIR_FIX:     "C4-DIR-FIX",
    ATYPE_DIR_FLIP:    "DIR-FLIP",
    ATYPE_POS_FIX:     "C4-POS-FIX",
    ATYPE_VITERBI:     "Dir-DP",
    ATYPE_TURN_ACCEPT: "Turn-Accept",
    ATYPE_KF_FILL:     "C1-KF-INTERP",
    ATYPE_OVERLAP_FILL: "C2-OVERLAP-FILL",
    ATYPE_EMBED_ABSTAIN: "C3-EMBED-ABSTAIN",
    ATYPE_EMBED_IDENTITY: "C3-EMBED-IDENTITY",
    ATYPE_EMBED_RELABEL: "C3-EMBED-RELABEL",
    ATYPE_MISSING:     "C5-MISSING",
    ATYPE_PRE_FILL:    "GUI-PRE",
    ATYPE_POST_FILL:   "GUI-POST",
    ATYPE_ANCHOR:      "GUI-ANCHOR",
    ATYPE_SPIKE:       "GUI-SPIKE",
    ATYPE_FILTER_FILL: "GUI-INTERP",
    ATYPE_ID_SWAP:     "GUI-ID-SWAP",
}

REFINEMENT_EXCLUDED_ASSIGN_CODES = frozenset({
    ATYPE_INIT,
    ATYPE_S1,
})
REFINEMENT_ASSIGN_CODES = frozenset(
    code for code in ASSIGN_CODE_ORDER
    if code not in REFINEMENT_EXCLUDED_ASSIGN_CODES
)
