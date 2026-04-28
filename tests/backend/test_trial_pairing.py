"""Tests for trial_pairing — multi-source Trial assembly.

A Trial = (subject, feat1, feat2, trial_idx, sync_window_idx)
        + paired Robot/Motion/Loadcell datasets
        + matching SyncWindow per source.

These tests synthesize uploaded files (Robot + Motion ± Loadcell)
with deterministic sync windows and verify the pairing produces the
expected Trial objects with the right sources matched up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.services.trial_pairing import (
    pair_into_trials,
    summarize,
)


# ============================================================
# Helpers
# ============================================================

def _sync(n: int, fs: float, windows: list[tuple[float, float]]) -> np.ndarray:
    out = np.zeros(n, dtype=float)
    t = np.arange(n) / fs
    for r, f in windows:
        out[(t >= r) & (t < f)] = 1.0
    return out


def _df(fs: float, windows: list[tuple[float, float]],
        duration_s: float = 8.0,
        time_col: str = "Time_ms",
        extra: dict | None = None) -> pd.DataFrame:
    n = int(duration_s * fs)
    sync = _sync(n, fs, windows)
    if time_col == "Time_ms":
        time_vals = np.arange(n) * (1000.0 / fs)
    else:
        time_vals = np.arange(n) / fs
    base = {time_col: time_vals, "Sync": sync}
    if extra:
        base.update(extra)
    return pd.DataFrame(base)


# ============================================================
# Single-trial pairing (one sync window each)
# ============================================================

def test_single_trial_robot_plus_motion_pairs_complete():
    """One Robot + one Motion file with one sync window each → 1 Trial,
    is_complete_dual = True."""
    robot_df  = _df(111.0, [(1.0, 4.0)])
    motion_df = _df(1000.0, [(1.5, 4.5)], time_col="Time")
    uploads = [
        ("ds_r", "260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv", robot_df),
        ("ds_m", "Motion_260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv", motion_df),
    ]
    trials = pair_into_trials(uploads)
    assert len(trials) == 1
    t = trials[0]
    assert t.subject == "s01"
    assert t.condition_label == "prox_axial"
    assert t.trial_idx == 1
    assert t.sync_window_idx == 0
    assert t.is_complete_dual is True
    assert t.warnings == []


def test_loadcell_attaches_at_trial_set_level():
    robot_df    = _df(111.0, [(1.0, 4.0)])
    motion_df   = _df(1000.0, [(1.5, 4.5)], time_col="Time")
    loadcell_df = pd.DataFrame({"time": np.linspace(0, 5, 50),
                                "applied_N": np.linspace(0, 50, 50)})
    uploads = [
        ("ds_r",  "260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv", robot_df),
        ("ds_m",  "Motion_260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv", motion_df),
        ("ds_lc", "Loadcell_260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv", loadcell_df),
    ]
    trials = pair_into_trials(uploads)
    assert len(trials) == 1
    assert trials[0].has_loadcell


# ============================================================
# Multi-window pairing (one recording → multiple trials)
# ============================================================

def test_three_windows_in_one_recording_makes_three_trials():
    """One Robot CSV + one Motion CSV, each with 3 sync windows →
    3 Trial objects (one per window index)."""
    robot_df  = _df(111.0, [(1.0, 3.0), (5.0, 7.5), (10.0, 13.0)],
                     duration_s=15.0)
    motion_df = _df(1000.0, [(1.2, 3.2), (5.2, 7.7), (10.2, 13.2)],
                     duration_s=15.0, time_col="Time")
    uploads = [
        ("ds_r", "260427_TD_level_1.0_H-Walker_s01_mid_radial_2.csv", robot_df),
        ("ds_m", "Motion_260427_TD_level_1.0_H-Walker_s01_mid_radial_2.csv", motion_df),
    ]
    trials = pair_into_trials(uploads)
    assert len(trials) == 3
    assert [t.sync_window_idx for t in trials] == [0, 1, 2]
    for t in trials:
        assert t.is_complete_dual
        assert t.subject == "s01"
        assert t.condition_label == "mid_radial"
        assert t.trial_idx == 2


def test_window_count_mismatch_emits_warning():
    """Robot has 3 windows, Motion has 2 → pair the first 2,
    every Trial in this recording carries a mismatch warning."""
    robot_df  = _df(111.0, [(1.0, 3.0), (5.0, 7.0), (10.0, 13.0)],
                     duration_s=15.0)
    motion_df = _df(1000.0, [(1.2, 3.2), (5.2, 7.2)],
                     duration_s=15.0, time_col="Time")
    uploads = [
        ("ds_r", "260427_TD_level_1.0_H-Walker_s02_dist_axial_1.csv", robot_df),
        ("ds_m", "Motion_260427_TD_level_1.0_H-Walker_s02_dist_axial_1.csv", motion_df),
    ]
    trials = pair_into_trials(uploads)
    assert len(trials) == 2
    for t in trials:
        assert any("mismatch" in w for w in t.warnings)


# ============================================================
# Partial trials (missing one source)
# ============================================================

def test_missing_motion_flagged_but_trial_emitted():
    """Robot only — emit Trial, flag missing Motion."""
    robot_df = _df(111.0, [(1.0, 3.0)])
    uploads = [
        ("ds_r", "260427_TD_level_1.0_H-Walker_s03_prox_radial_1.csv", robot_df),
    ]
    trials = pair_into_trials(uploads)
    assert len(trials) == 1
    t = trials[0]
    assert t.has_robot and not t.has_motion
    assert t.is_complete_dual is False
    assert any("Motion" in w for w in t.warnings)


def test_missing_robot_flagged_but_trial_emitted():
    motion_df = _df(1000.0, [(1.5, 4.5)], time_col="Time")
    uploads = [
        ("ds_m", "Motion_260427_TD_level_1.0_H-Walker_s04_dist_radial_2.csv", motion_df),
    ]
    trials = pair_into_trials(uploads)
    assert len(trials) == 1
    t = trials[0]
    assert t.has_motion and not t.has_robot
    assert any("Robot" in w for w in t.warnings)


# ============================================================
# Multi-condition + multi-subject pairing
# ============================================================

def test_pairing_across_subjects_and_conditions():
    """Two subjects, two conditions each, all single-window — should
    produce 4 Trials with correct subject/condition tags."""
    r1 = _df(111.0, [(1.0, 4.0)])
    r2 = _df(111.0, [(1.0, 4.0)])
    r3 = _df(111.0, [(1.0, 4.0)])
    r4 = _df(111.0, [(1.0, 4.0)])
    m1 = _df(1000.0, [(1.5, 4.5)], time_col="Time")
    m2 = _df(1000.0, [(1.5, 4.5)], time_col="Time")
    m3 = _df(1000.0, [(1.5, 4.5)], time_col="Time")
    m4 = _df(1000.0, [(1.5, 4.5)], time_col="Time")
    uploads = [
        ("r1", "260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv", r1),
        ("m1", "Motion_260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv", m1),
        ("r2", "260427_TD_level_1.0_H-Walker_s01_dist_radial_1.csv", r2),
        ("m2", "Motion_260427_TD_level_1.0_H-Walker_s01_dist_radial_1.csv", m2),
        ("r3", "260428_TD_level_1.0_H-Walker_s02_prox_axial_1.csv", r3),
        ("m3", "Motion_260428_TD_level_1.0_H-Walker_s02_prox_axial_1.csv", m3),
        ("r4", "260428_TD_level_1.0_H-Walker_s02_none_normal_1.csv", r4),
        ("m4", "Motion_260428_TD_level_1.0_H-Walker_s02_none_normal_1.csv", m4),
    ]
    trials = pair_into_trials(uploads)
    assert len(trials) == 4
    by_key = {(t.subject, t.condition_label): t for t in trials}
    assert ("s01", "prox_axial") in by_key
    assert ("s01", "dist_radial") in by_key
    assert ("s02", "prox_axial") in by_key
    assert ("s02", "none_normal") in by_key
    for t in trials:
        assert t.is_complete_dual


# ============================================================
# Summary aggregation
# ============================================================

def test_summary_counts_complete_and_missing():
    r1 = _df(111.0, [(1.0, 4.0)])
    r2 = _df(111.0, [(1.0, 4.0)])
    m1 = _df(1000.0, [(1.5, 4.5)], time_col="Time")
    uploads = [
        ("r1", "260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv", r1),
        ("m1", "Motion_260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv", m1),
        ("r2", "260427_TD_level_1.0_H-Walker_s01_mid_axial_1.csv", r2),
        # mid_axial Motion missing on purpose
    ]
    trials = pair_into_trials(uploads)
    s = summarize(trials)
    assert s["n_trials_total"] == 2
    assert s["n_complete_dual"] == 1
    assert s["by_subject"]["s01"]["missing_motion"] == 1


# ============================================================
# Non-canonical filenames are silently skipped
# ============================================================

def test_non_canonical_filenames_are_ignored():
    """Files outside the 9-token convention don't generate Trials but
    must not crash the pairing of valid files."""
    good = _df(111.0, [(1.0, 4.0)])
    junk = _df(111.0, [(1.0, 4.0)])
    uploads = [
        ("g",  "260427_TD_level_1.0_H-Walker_s01_prox_axial_1.csv", good),
        ("j1", "random_file.csv", junk),
        ("j2", "another junk.csv", junk),
        ("j3", "260427_TD_level_1.0_H-Walker_CBJ_Low_30.csv", junk),  # legacy 8-token
    ]
    trials = pair_into_trials(uploads)
    assert len(trials) == 1
    assert trials[0].subject == "s01"
