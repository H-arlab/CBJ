"""Tests for the H-Walker 9-token filename parser.

Canonical: {date}_{condition}_{terrain}_{speed}_{project}_{subject}_
            {f1}_{f2}_{trial}.csv
Optional source prefix: Loadcell_ or Motion_ (no prefix = Robot).

8 valid (f1, f2) condition pairs:
  (none, normal)  baseline 1: walking, no WB
  (none, WB)      baseline 2: WB only
  (prox, axial)   ─┐
  (prox, radial)   │  6 robot-assist conditions
  (mid,  axial)    │
  (mid,  radial)   │
  (dist, axial)    │
  (dist, radial)  ─┘
"""
from __future__ import annotations

from backend.services.filename_parser import (
    parse,
    is_canonical,
    VALID_CONDITION_PAIRS,
)


# ============================================================
# Happy paths
# ============================================================

def test_parses_robot_baseline_normal():
    p = parse("260427_TD_level_1.0_H-Walker_s01_none_normal_1.csv")
    assert p is not None
    assert p.source_prefix == "robot"
    assert p.date == "260427"
    assert p.condition == "TD"
    assert p.terrain == "level"
    assert p.speed_mps == 1.0
    assert p.project == "H-Walker"
    assert p.subject == "s01"
    assert p.feat1 == "none"
    assert p.feat2 == "normal"
    assert p.trial_idx == 1
    assert p.is_baseline is True
    assert p.is_assist is False
    assert p.condition_label == "s01_none_normal_1"


def test_parses_robot_baseline_WB():
    p = parse("260427_TD_level_1.0_H-Walker_s01_none_WB_2.csv")
    assert p is not None
    assert p.feat1 == "none"
    assert p.feat2 == "WB"
    assert p.is_baseline is True


def test_parses_robot_assist_all_six():
    """All 6 assist conditions must parse and be flagged is_assist."""
    for pos in ("prox", "mid", "dist"):
        for direction in ("axial", "radial"):
            name = f"260427_TD_level_1.0_H-Walker_s05_{pos}_{direction}_3.csv"
            p = parse(name)
            assert p is not None, name
            assert p.feat1 == pos
            assert p.feat2 == direction
            assert p.is_assist is True
            assert p.is_baseline is False


def test_parses_loadcell_prefix():
    p = parse("Loadcell_260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv")
    assert p is not None
    assert p.source_prefix == "loadcell"
    assert p.subject == "s01"


def test_parses_motion_prefix():
    p = parse("Motion_260427_TD_level_1.0_H-Walker_s10_dist_radial_3.csv")
    assert p is not None
    assert p.source_prefix == "motion"
    assert p.subject == "s10"


def test_strips_directory_path():
    p = parse("/home/user/data/s01/Motion_260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv")
    assert p is not None
    assert p.source_prefix == "motion"


# ============================================================
# Sad paths — must return None
# ============================================================

def test_returns_none_for_non_csv():
    assert parse("260427_TD_level_1.0_H-Walker_s01_prox_axial_1.txt") is None


def test_returns_none_for_legacy_8_token():
    """The legacy 'feat is one token' format (e.g. `_CBJ_Low_30`) is
    not recognized by the new 9-token canonical parser. The detector
    will fall back to column signature for these files."""
    assert parse("260427_TD_level_1.0_H-Walker_CBJ_Low_30.csv") is None


def test_returns_none_for_unknown_condition_pair():
    """Token shape OK but (feat1, feat2) not in the whitelist."""
    assert parse("260427_TD_level_1.0_H-Walker_s01_prox_normal_1.csv") is None
    assert parse("260427_TD_level_1.0_H-Walker_s01_dist_WB_1.csv") is None


def test_returns_none_for_garbage():
    assert parse("data.csv") is None
    assert parse("foobar_baz.csv") is None
    assert parse("") is None


# ============================================================
# Helpers
# ============================================================

def test_cell_key_pairs_three_sources():
    """The three source files of one trial share the same cell_key."""
    r = parse("260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv")
    l = parse("Loadcell_260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv")
    m = parse("Motion_260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv")
    assert r.cell_key == l.cell_key == m.cell_key
    assert r.cell_key == ("s01", "prox", "axial", 1)


def test_is_canonical():
    assert is_canonical("260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv")
    assert not is_canonical("data.csv")


def test_valid_condition_pairs_count():
    assert len(VALID_CONDITION_PAIRS) == 8


# ============================================================
# Zero-padding robustness — operator may export `s1` or `s01`,
# `_3.csv` or `_03.csv`, and the three sources of one trial must
# still pair up. These tests pin that contract.
# ============================================================

class TestZeroPaddingRobustness:
    def test_subject_s1_and_s01_collapse_to_canonical_form(self):
        a = parse("260427_TD_level_1.0_H-Walker_s1_prox_axial_3.csv")
        b = parse("260427_TD_level_1.0_H-Walker_s01_prox_axial_3.csv")
        assert a is not None and b is not None
        assert a.subject == b.subject == "s01"

    def test_trial_idx_3_and_03_collapse(self):
        a = parse("260427_TD_level_1.0_H-Walker_s01_prox_axial_3.csv")
        b = parse("260427_TD_level_1.0_H-Walker_s01_prox_axial_03.csv")
        assert a is not None and b is not None
        assert a.trial_idx == b.trial_idx == 3

    def test_cell_key_matches_across_padding(self):
        """The three source CSVs of one trial must share cell_key
        regardless of zero-padding in the subject or trial fields."""
        robot = parse("260427_TD_level_1.0_H-Walker_s1_prox_axial_3.csv")
        motion = parse("Motion_260427_TD_level_1.0_H-Walker_s01_prox_axial_03.csv")
        loadcell = parse("Loadcell_260427_TD_level_1.0_H-Walker_s01_prox_axial_3.csv")
        assert robot is not None and motion is not None and loadcell is not None
        assert robot.cell_key == motion.cell_key == loadcell.cell_key

    def test_capital_S_in_subject_normalizes(self):
        p = parse("260427_TD_level_1.0_H-Walker_s7_prox_axial_1.csv")
        assert p is not None
        assert p.subject == "s07"


# ============================================================
# Upload heuristic — `backend/routers/datasets.py:_parse_filename`
# is a SEPARATE parser used to *guess* (subject_id, condition, group)
# for arbitrary uploaded filenames that don't match the canonical
# 9-token shape. Regression for the operator's report: filenames
# like `robot_high_0.CSV` and `loadcell_low_30.CSV` were silently
# extracting the trial-index suffix as subject_id ("0", "30") and
# stamping that on the Dataset card.
# ============================================================

class TestUploadFilenameHeuristic:
    def test_robot_high_0_does_not_invent_subject(self):
        """The trailing `_0` is the trial index, not a subject."""
        from backend.routers.datasets import _parse_filename
        out = _parse_filename("robot_high_0.CSV")
        assert "subject_id" not in out, (
            f"robot_high_0.CSV should not auto-populate subject_id; "
            f"got {out!r}"
        )

    def test_loadcell_low_30_does_not_invent_subject(self):
        from backend.routers.datasets import _parse_filename
        out = _parse_filename("loadcell_low_30.CSV")
        assert "subject_id" not in out, (
            f"loadcell_low_30.CSV should not auto-populate subject_id; "
            f"got {out!r}"
        )

    def test_explicit_s_prefix_still_parses(self):
        from backend.routers.datasets import _parse_filename
        out = _parse_filename("s07_pre_2024_05_01.csv")
        assert out.get("subject_id") == "s07"
        assert out.get("condition") == "Pre"

    def test_subj_prefix_picks_up_bare_digits(self):
        from backend.routers.datasets import _parse_filename
        out = _parse_filename("pre_subj_07.csv")
        assert out.get("subject_id") == "07"

    def test_leading_digits_only_with_known_condition(self):
        from backend.routers.datasets import _parse_filename
        out = _parse_filename("001_pre.csv")
        assert out.get("subject_id") == "001"
        assert out.get("condition") == "Pre"

    def test_random_garbage_returns_empty(self):
        from backend.routers.datasets import _parse_filename
        assert _parse_filename("random_garbage.csv") == {}

    def test_robot_underscore_word_underscore_digit_returns_empty(self):
        """Generalization of the operator's case — any
        `<word>_<word>_<digit>.csv` shape must NOT misattribute the
        trailing digit as subject."""
        from backend.routers.datasets import _parse_filename
        for fn in (
            "Robot_high_3.csv",
            "Motion_low_15.csv",
            "loadcell_baseline_2.csv",
        ):
            out = _parse_filename(fn)
            assert "subject_id" not in out, (
                f"{fn} should not auto-populate subject_id; got {out!r}"
            )
