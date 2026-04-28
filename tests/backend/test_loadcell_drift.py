"""Tests for loadcell_drift — calibration regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.services import loadcell_drift


# ============================================================
# Happy paths
# ============================================================

def test_perfect_calibration_yields_slope_one_intercept_zero():
    """robot = applied (no drift) → β=1, α=0, R²=1, drift_at_50N=0."""
    applied = np.linspace(0.0, 100.0, 21)
    df = pd.DataFrame({"applied_N": applied, "robot_reported_N": applied})
    res = loadcell_drift.fit_drift(df)
    assert abs(res.slope - 1.0) < 1e-9
    assert abs(res.intercept) < 1e-9
    assert res.r_squared > 0.999
    assert abs(res.drift_at_50N_n) < 1e-9
    assert res.slope_ok and res.intercept_ok


def test_2_percent_drift_detected():
    """robot = 1.02 · applied → β=1.02, slope_ok=False (>5% threshold)
    is False (2% is within 5%) so this should still pass slope_ok,
    but drift_at_50N = 1.0 N."""
    applied = np.linspace(0.0, 100.0, 21)
    df = pd.DataFrame({"applied_N": applied, "robot_N": applied * 1.02})
    res = loadcell_drift.fit_drift(df)
    assert abs(res.slope - 1.02) < 1e-6
    assert res.slope_ok is True   # within 5%
    assert abs(res.drift_at_50N_n - 1.0) < 1e-6


def test_10_percent_drift_fails_slope_ok():
    applied = np.linspace(0.0, 100.0, 21)
    df = pd.DataFrame({"applied_N": applied, "robot_N": applied * 1.10})
    res = loadcell_drift.fit_drift(df)
    assert abs(res.slope - 1.10) < 1e-6
    assert res.slope_ok is False
    assert res.drift_at_50N_n == pytest.approx(5.0, abs=1e-6)


def test_offset_only_yields_intercept_and_unit_slope():
    """robot = applied + 3 N → β=1, α=3."""
    applied = np.linspace(0.0, 100.0, 21)
    df = pd.DataFrame({"applied_N": applied, "robot_N": applied + 3.0})
    res = loadcell_drift.fit_drift(df)
    assert abs(res.slope - 1.0) < 1e-9
    assert abs(res.intercept - 3.0) < 1e-9
    assert res.intercept_ok is False    # |α| > 1 N


def test_noisy_calibration_t_test_significant_when_drift_real():
    """Add ε noise; β=1.05 should be detected as significantly
    different from 1 when n is large enough."""
    rng = np.random.default_rng(seed=42)
    applied = np.tile(np.linspace(0.0, 100.0, 21), 5)   # n=105
    noise = rng.normal(0, 1.0, applied.size)
    df = pd.DataFrame({
        "applied_N": applied,
        "robot_N": applied * 1.05 + noise,
    })
    res = loadcell_drift.fit_drift(df)
    assert 1.04 < res.slope < 1.06
    assert res.slope_p_vs_1 < 0.001     # clearly different from 1


# ============================================================
# Auto-detection of column names
# ============================================================

def test_auto_detects_alternative_column_names():
    applied = np.linspace(0.0, 100.0, 11)
    df = pd.DataFrame({"reference_N": applied, "L_ActForce_N": applied * 1.02})
    res = loadcell_drift.fit_drift(df)
    assert res.applied_col == "reference_N"
    assert res.robot_col == "L_ActForce_N"
    assert abs(res.slope - 1.02) < 1e-6


def test_explicit_column_override_wins():
    applied = np.linspace(0.0, 100.0, 11)
    df = pd.DataFrame({
        "applied_N": applied,
        "robot_N": applied * 1.02,
        "extra_N": applied * 0.5,    # ambiguous extra column
    })
    res = loadcell_drift.fit_drift(df, applied_col="applied_N",
                                   robot_col="robot_N")
    assert abs(res.slope - 1.02) < 1e-6


# ============================================================
# Sad paths
# ============================================================

def test_raises_when_too_few_points():
    df = pd.DataFrame({"applied_N": [0, 50, 100],
                       "robot_N":   [0, 50, 100]})
    with pytest.raises(ValueError, match="≥4"):
        loadcell_drift.fit_drift(df)


def test_raises_when_no_applied_variance():
    df = pd.DataFrame({"applied_N": [50] * 10,
                       "robot_N":   list(range(10))})
    with pytest.raises(ValueError, match="variance"):
        loadcell_drift.fit_drift(df)


def test_raises_when_columns_missing():
    df = pd.DataFrame({"foo": [1, 2, 3, 4], "bar": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="calibration columns"):
        loadcell_drift.fit_drift(df)
