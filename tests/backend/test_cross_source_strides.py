"""Tests for cross-source per-stride table builder.

Builds a synthetic Robot+Motion paired pair of aligned grids,
then verifies that:
  - strides are detected from Robot's GCP active mask
  - per-stride table has Robot + Motion columns side-by-side
  - sample counts and sequences are correct
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.services import cross_source_strides as css


def _gcp_sawtooth(n: int, fs: float, stride_s: float,
                   stance_frac: float = 0.6) -> np.ndarray:
    """Per-side GCP: 0 → 1 ramp during stance, 0 during swing."""
    out = np.zeros(n)
    stride_n = int(round(stride_s * fs))
    stance_n = int(round(stride_s * stance_frac * fs))
    for start in range(0, n - stride_n, stride_n):
        end = start + stance_n
        out[start:end] = np.linspace(0.01, 1.0, end - start)
    return out


def _aligned_grids(fs=1000.0, stride_s=1.0, n_strides=4):
    """Build two grids that share t_aligned axis. Robot has the
    GCP / force columns, Motion has GRF + a knee angle + EMG."""
    n = int((n_strides + 0.2) * stride_s * fs)
    t = np.arange(n) / fs

    # Robot
    L_GCP = _gcp_sawtooth(n, fs, stride_s)
    R_GCP = _gcp_sawtooth(n, fs, stride_s)   # treat L=R for simplicity
    L_DesForce_N = 30.0 * (L_GCP > 0.01)
    L_ActForce_N = L_DesForce_N + 1.0  # 1 N over-target
    L_Pitch = np.sin(2 * np.pi * t / stride_s) * 12.0   # 24° peak-to-peak

    robot = pd.DataFrame({
        "t_aligned": t,
        "L_GCP": L_GCP, "R_GCP": R_GCP,
        "L_DesForce_N": L_DesForce_N, "L_ActForce_N": L_ActForce_N,
        "L_Pitch": L_Pitch,
    })

    # Motion: Fz peaks during stance, knee flex pattern, EMG envelope
    Fz = 600.0 * (L_GCP > 0.01)
    knee = np.sin(2 * np.pi * t / stride_s) * 30.0 + 30.0
    emg_vl = np.abs(np.sin(2 * np.pi * t / stride_s)) * 0.5

    motion = pd.DataFrame({
        "t_aligned": t,
        "FP1_Fz": Fz,
        "FP1_COPx": np.linspace(0, 5, n),
        "FP1_COPy": np.linspace(0, 0, n),
        "RKneeAngle_X": knee,
        "EMG_VL": emg_vl,
    })

    return robot, motion, fs, stride_s, n_strides


# ============================================================
# Stride detection on the aligned grid
# ============================================================

def test_detect_strides_finds_expected_count():
    robot, _, fs, stride_s, n_strides = _aligned_grids(n_strides=5)
    strides = css.detect_strides_on_grid(robot, "L")
    assert len(strides) == n_strides - 1   # n HS → n-1 strides


def test_detect_strides_returns_empty_when_no_gcp():
    df = pd.DataFrame({"t_aligned": np.arange(100) / 100.0,
                        "L_ActForce_N": np.zeros(100)})
    assert css.detect_strides_on_grid(df, "L") == []


# ============================================================
# Cross-source table assembly
# ============================================================

def test_build_table_with_robot_and_motion_includes_both_columns():
    robot, motion, *_ = _aligned_grids(n_strides=4)
    res = css.build_stride_table(robot, motion, side="L")
    assert res.has_motion is True
    assert res.n_strides >= 1
    cols = list(res.table.columns)
    # Robot
    assert "cable_force_peak_L" in cols
    assert "cable_force_rmse_L" in cols
    assert "shank_pitch_rom_L" in cols
    # Motion (force plate prefix is "fp1" lowercased after our key build)
    fp_cols = [c for c in cols if c.startswith("grf_fz_peak.")]
    assert fp_cols, f"no GRF peak column in {cols}"
    knee_cols = [c for c in cols if "kneeangle" in c]
    assert knee_cols, f"no knee-angle column in {cols}"
    emg_cols = [c for c in cols if c.startswith("emg_rms.")]
    assert emg_cols, f"no EMG RMS column in {cols}"


def test_build_table_robot_only_omits_motion_columns():
    robot, _, *_ = _aligned_grids(n_strides=4)
    res = css.build_stride_table(robot, motion_grid=None, side="L")
    assert res.has_motion is False
    cols = list(res.table.columns)
    assert "cable_force_peak_L" in cols
    assert not any(c.startswith("grf_fz_peak") for c in cols)
    assert any("motion grid not provided" in n for n in res.notes)


def test_table_rows_equal_n_strides():
    robot, motion, *_ = _aligned_grids(n_strides=6)
    res = css.build_stride_table(robot, motion, side="L")
    assert len(res.table) == res.n_strides


def test_grf_peak_close_to_truth():
    """Synthetic GRF peak is 600 N during stance — table column should
    report ≈ 600 for every stride."""
    robot, motion, *_ = _aligned_grids(n_strides=4)
    res = css.build_stride_table(robot, motion, side="L")
    fp_col = next(c for c in res.table.columns
                  if c.startswith("grf_fz_peak."))
    peaks = res.table[fp_col].dropna().to_numpy()
    assert peaks.size >= 2
    assert np.allclose(peaks, 600.0, atol=1.0)


def test_knee_rom_close_to_truth():
    """Knee angle = 30 + 30·sin → ROM = 60° per cycle."""
    robot, motion, *_ = _aligned_grids(n_strides=4)
    res = css.build_stride_table(robot, motion, side="L")
    rom_col = next(c for c in res.table.columns
                   if c.startswith("joint_rom.") and "kneeangle" in c)
    roms = res.table[rom_col].dropna().to_numpy()
    assert roms.size >= 2
    assert np.all((roms > 50.0) & (roms < 65.0))
