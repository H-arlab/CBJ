"""Tests for Motion-source per-stride metric primitives.

Each primitive takes (df, stride_boundaries, column) and returns
one value per stride. Tests use synthetic signals where the answer
is analytically known so any regression produces a deterministic
failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.services import motion_metrics


# ============================================================
# Synthetic helpers
# ============================================================

def _grid(fs: float, duration_s: float, **cols) -> pd.DataFrame:
    n = int(duration_s * fs)
    base = {"t_aligned": np.arange(n) / fs}
    base.update({k: np.asarray(v, dtype=float) for k, v in cols.items()})
    return pd.DataFrame(base)


# ============================================================
# GRF
# ============================================================

def test_grf_fz_peak_picks_max_per_stride():
    fs = 100.0
    n = 500
    fz = np.zeros(n)
    fz[100:150] = 600.0   # stride 0 peak = 600
    fz[200:250] = 700.0   # stride 1 peak = 700
    df = _grid(fs, 5.0, FP1_Fz=fz)
    strides = [(50, 200), (200, 350)]
    peaks = motion_metrics.grf_fz_peak(df, strides, fz_col="FP1_Fz")
    assert peaks[0] == 600.0
    assert peaks[1] == 700.0


def test_grf_fz_impulse_trapezoidal():
    """For Fz constant at 100 N during [0, 1] s and 0 elsewhere,
    impulse over the [0, 1] window is exactly 100 N·s."""
    fs = 1000.0
    n = 2000
    fz = np.zeros(n)
    fz[0:1000] = 100.0
    df = _grid(fs, 2.0, FP1_Fz=fz)
    strides = [(0, 1000)]
    imp = motion_metrics.grf_fz_impulse(df, strides, fz_col="FP1_Fz")
    # Trapezoidal includes endpoints — 999 intervals × 0.001 s × 100 N
    assert 99.5 < imp[0] < 100.5


def test_grf_loading_rate_recovers_known_slope():
    """Linear ramp 0→100 N over 50 ms (fs=1000 Hz) → slope = 2000 N/s.
    Loading rate over the first 50 ms after HS should be ≈ 2000."""
    fs = 1000.0
    n = 200
    fz = np.zeros(n)
    fz[0:50] = np.linspace(0, 100, 50)
    df = _grid(fs, 0.2, FP1_Fz=fz)
    strides = [(0, 200)]
    rates = motion_metrics.grf_loading_rate(
        df, strides, fz_col="FP1_Fz", time_col="t_aligned", window_ms=50.0,
    )
    assert 1900 < rates[0] < 2100


def test_cop_path_length_simple_diagonal():
    """COP moves diagonally from (0,0) to (3,4) over a stride —
    Euclidean path length = √(3² + 4²) = 5 mm. With 5 samples on
    that diagonal the integral via summed segment norms is also 5."""
    n = 20
    copx = np.linspace(0, 3, n)
    copy = np.linspace(0, 4, n)
    df = _grid(100.0, n / 100.0, FP1_COPx=copx, FP1_COPy=copy)
    strides = [(0, n)]
    lengths = motion_metrics.cop_path_length(df, strides,
                                              copx_col="FP1_COPx",
                                              copy_col="FP1_COPy")
    assert 4.95 < lengths[0] < 5.05


# ============================================================
# Joint kinematics
# ============================================================

def test_joint_peak_per_stride_picks_max():
    angle = np.array([10.0, 30.0, 60.0, 50.0, 5.0,
                      -10.0, -5.0, 25.0, 70.0, 65.0])
    df = _grid(100.0, 0.1, RKneeAngle_X=angle)
    strides = [(0, 5), (5, 10)]
    peaks = motion_metrics.joint_peak_per_stride(
        df, strides, angle_col="RKneeAngle_X")
    assert peaks[0] == 60.0
    assert peaks[1] == 70.0


def test_joint_rom_per_stride_max_minus_min():
    angle = np.array([10.0, 30.0, 60.0, 50.0, 5.0,
                      -10.0, -5.0, 25.0, 70.0, 65.0])
    df = _grid(100.0, 0.1, RKneeAngle_X=angle)
    strides = [(0, 5), (5, 10)]
    roms = motion_metrics.joint_rom_per_stride(
        df, strides, angle_col="RKneeAngle_X")
    assert roms[0] == 60.0 - 5.0       # 55
    assert roms[1] == 70.0 - (-10.0)   # 80


def test_joint_moment_peak_preserves_sign():
    """For an extension moment that goes negative (flexion-positive
    convention) the absolute peak should preserve the sign."""
    moment = np.array([1.0, 2.0, -50.0, -45.0, -30.0, 5.0, 10.0])
    df = _grid(100.0, 0.07, RKneeMoment_X=moment)
    strides = [(0, 7)]
    peaks = motion_metrics.joint_moment_peak_per_stride(
        df, strides, moment_col="RKneeMoment_X")
    assert peaks[0] == -50.0


# ============================================================
# EMG
# ============================================================

def test_emg_rms_per_stride_known_value():
    """RMS of constant 3.0 mV signal over a stride = 3.0."""
    emg = np.full(100, 3.0)
    df = _grid(1000.0, 0.1, EMG_VL=emg)
    rms = motion_metrics.emg_rms_per_stride(df, [(0, 100)], emg_col="EMG_VL")
    assert abs(rms[0] - 3.0) < 1e-9


def test_emg_peak_returns_max_abs():
    emg = np.array([0.1, 0.5, -0.8, 0.3, -0.2])
    df = _grid(1000.0, 0.005, EMG_VL=emg)
    peak = motion_metrics.emg_peak_per_stride(df, [(0, 5)], emg_col="EMG_VL")
    assert peak[0] == 0.8


def test_cocontraction_index_perfect_overlap_is_one():
    """Identical activation in agonist and antagonist → CCI = 1."""
    a = np.full(100, 0.4)
    b = np.full(100, 0.4)
    df = _grid(1000.0, 0.1, EMG_VL=a, EMG_BF=b)
    cci = motion_metrics.cocontraction_index(
        df, [(0, 100)], agonist_col="EMG_VL", antagonist_col="EMG_BF")
    assert abs(cci[0] - 1.0) < 1e-9


def test_cocontraction_index_no_overlap_is_zero():
    """One muscle silent throughout → CCI = 0."""
    a = np.full(100, 0.5)
    b = np.zeros(100)
    df = _grid(1000.0, 0.1, EMG_VL=a, EMG_BF=b)
    cci = motion_metrics.cocontraction_index(
        df, [(0, 100)], agonist_col="EMG_VL", antagonist_col="EMG_BF")
    assert abs(cci[0]) < 1e-9


# ============================================================
# Edge cases
# ============================================================

def test_returns_nan_for_missing_column():
    df = _grid(100.0, 1.0, foo=np.zeros(100))
    out = motion_metrics.grf_fz_peak(df, [(0, 100)], fz_col="FP1_Fz")
    assert np.isnan(out[0])


def test_returns_nan_for_too_short_stride():
    df = _grid(100.0, 1.0, FP1_Fz=np.full(100, 500.0))
    out = motion_metrics.grf_fz_peak(df, [(0, 3)], fz_col="FP1_Fz")  # 3 samples
    assert np.isnan(out[0])
