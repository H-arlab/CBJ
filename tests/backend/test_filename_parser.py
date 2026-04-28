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
