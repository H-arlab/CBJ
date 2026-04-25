"""Tests for source-kind detection (Robot / Loadcell / Motion / Unknown).

User contract from the conversation:
    "Robot Data - Robot or robot CBJ_4.csv 이런식으로
     Loadcell Data - Loadcell or loadcell CBJ_5.csv
     Motion Data - Motion CBJ 이런 식으로 만들게"

Filename match is the primary cue. Column signature is the fallback
(third-party CSVs that don't follow the convention) and the
disagreement-resolver (filename says one thing, columns say another).
"""
from __future__ import annotations

from backend.services.source_kind import (
    detect_source_kind,
    detect_from_filename,
    detect_from_columns,
)


# ---------------------------------------------------------------
# Filename-only path
# ---------------------------------------------------------------

def test_filename_robot_variants():
    for name in (
        "Robot CBJ_4.csv",
        "robot_high_0.csv",
        "Robot.csv",
        "ROBOT_trial_1.csv",
    ):
        assert detect_from_filename(name) == "robot", name


def test_filename_loadcell_variants():
    for name in (
        "Loadcell CBJ_5.csv",
        "loadcell_low_30.csv",
        "Loadcell.csv",
    ):
        assert detect_from_filename(name) == "loadcell", name


def test_filename_motion_variants():
    for name in (
        "Motion CBJ.csv",
        "motion_trial_3.csv",
        "MOTION 04.csv",
    ):
        assert detect_from_filename(name) == "motion", name


def test_filename_no_match_is_none():
    for name in ("trial_001.csv", "session_a.csv", "data.csv"):
        assert detect_from_filename(name) is None


# ---------------------------------------------------------------
# Column-signature path
# ---------------------------------------------------------------

def test_columns_robot_signature():
    cols = ["Time_ms", "L_ActForce_N", "R_ActForce_N", "L_GCP", "R_GCP",
            "Sync", "L_Pitch", "R_Pitch"]
    kind, cues = detect_from_columns(cols)
    assert kind == "robot"
    assert any("L_ActForce_N" in c for c in cues)


def test_columns_motion_force_plate():
    cols = ["Frame", "Time", "FP1_Fx", "FP1_Fy", "FP1_Fz",
            "FP1_Mx", "FP1_My", "FP1_Mz", "FP1_COPx", "FP1_COPy"]
    kind, cues = detect_from_columns(cols)
    assert kind == "motion"
    assert any("force-plate" in c for c in cues)


def test_columns_motion_emg_only():
    cols = ["Time", "EMG_VL", "EMG_VM", "EMG_BF", "EMG_GAS"]
    kind, cues = detect_from_columns(cols)
    assert kind == "motion"
    assert any("EMG" in c for c in cues)


def test_columns_motion_markers():
    cols = ["Frame", "Time", "RHIP_X", "RHIP_Y", "RHIP_Z",
            "LKNEE_X", "LKNEE_Y", "LKNEE_Z", "RANK_X", "RANK_Y", "RANK_Z"]
    kind, cues = detect_from_columns(cols)
    assert kind == "motion"
    assert any("markers" in c for c in cues)


def test_columns_motion_joint_angles():
    cols = ["Time", "RHipAngle_X", "RKneeAngle_X", "RAnkleAngle_X"]
    kind, cues = detect_from_columns(cols)
    assert kind == "motion"


def test_columns_loadcell_calibration():
    cols = ["time", "applied_N", "robot_N"]
    kind, cues = detect_from_columns(cols)
    assert kind == "loadcell"


def test_columns_unknown_fallback():
    cols = ["foo", "bar", "baz"]
    kind, cues = detect_from_columns(cols)
    assert kind == "unknown"
    assert cues == []


# ---------------------------------------------------------------
# Combined path — confidence scoring
# ---------------------------------------------------------------

def test_filename_and_columns_agree_high_confidence():
    cols = ["Time_ms", "L_ActForce_N", "R_ActForce_N", "L_GCP"]
    d = detect_source_kind("Robot CBJ_4.csv", cols)
    assert d.kind == "robot"
    assert d.confidence == 1.0


def test_filename_only_medium_confidence():
    """User named it 'Robot ...' but columns are unrecognizable
    (e.g. only 'foo, bar'). Trust the user, mark medium confidence."""
    d = detect_source_kind("Robot CBJ_4.csv", ["foo", "bar"])
    assert d.kind == "robot"
    assert d.confidence == 0.6


def test_columns_override_misleading_filename():
    """Filename says Robot, columns are clearly motion → believe
    columns and surface the conflict in cues."""
    d = detect_source_kind(
        "Robot.csv",
        ["Time", "FP1_Fx", "FP1_Fy", "FP1_Fz", "EMG_VL"],
    )
    assert d.kind == "motion"
    assert d.confidence == 0.5
    assert any("filename suggests 'robot'" in c for c in d.matched_cues)
    assert any("columns suggest 'motion'" in c for c in d.matched_cues)


def test_columns_only_high_confidence_when_filename_neutral():
    cols = ["Time_ms", "L_ActForce_N", "R_ActForce_N", "L_GCP"]
    d = detect_source_kind("trial_xyz.csv", cols)
    assert d.kind == "robot"
    assert d.confidence == 0.7


def test_unknown_when_nothing_matches():
    d = detect_source_kind("data.csv", ["a", "b", "c"])
    assert d.kind == "unknown"
    assert d.confidence == 0.0
