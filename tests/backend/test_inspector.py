"""Tests for the per-sync inspector — sync window detection + slicing.

Sync contract (user-confirmed):
    Rising edge = sync STARTS (operator pressed → trial begins)
    Falling edge = sync ENDS  (operator released → trial ends)
    Window = [rising, falling] half-open interval
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.routers import inspector


# ============================================================
# Synthetic helpers
# ============================================================

def _sync_pulses(n: int, fs: float,
                  windows: list[tuple[float, float]]) -> np.ndarray:
    """Sync = LOW by default, HIGH inside each (rise, fall) interval."""
    out = np.zeros(n, dtype=float)
    t = np.arange(n) / fs
    for r, f in windows:
        out[(t >= r) & (t < f)] = 1.0
    return out


# ============================================================
# Window detection
# ============================================================

def test_detect_three_windows():
    """3 s of recording with 3 rising-falling pulses → 3 windows."""
    fs = 100.0
    n = 1500  # 15 s total
    t = np.arange(n) / fs
    sync = _sync_pulses(n, fs, [(1.0, 3.0), (5.0, 7.5), (9.0, 13.0)])
    windows = inspector._detect_sync_windows(sync, t)
    assert len(windows) == 3
    expected = [(1.0, 3.0), (5.0, 7.5), (9.0, 13.0)]
    for (rs, fs_e), (er, ef) in zip(windows, expected):
        assert abs(rs - er) < 0.02
        assert abs(fs_e - ef) < 0.02


def test_detect_no_windows_when_constant():
    fs = 100.0
    t = np.arange(200) / fs
    assert inspector._detect_sync_windows(np.zeros(200), t) == []
    assert inspector._detect_sync_windows(np.ones(200), t) == []


def test_detect_drops_unclosed_trailing_window():
    """Sync rises at 4 s but never falls before EOF → that pulse is
    incomplete and must not be returned."""
    fs = 100.0
    n = 500
    t = np.arange(n) / fs
    sync = np.zeros(n)
    sync[100:200] = 1.0   # complete window: 1.0–2.0 s
    sync[400:] = 1.0      # rises at 4 s, never falls
    windows = inspector._detect_sync_windows(sync, t)
    assert len(windows) == 1


def test_detect_handles_analog_threshold():
    """Analog TTL (5 V high) — threshold midpoint must work."""
    fs = 1000.0
    n = 5000
    t = np.arange(n) / fs
    sync = np.zeros(n)
    for start in (0.5, 1.5, 2.5):
        sync[(t >= start) & (t < start + 0.4)] = 5.0
    windows = inspector._detect_sync_windows(sync, t)
    assert len(windows) == 3


# ============================================================
# Endpoint helpers
# ============================================================

def _register(monkeypatch, df: pd.DataFrame, ds_id: str = "ds_test") -> str:
    """Patch inspector._read_df to return our synthetic df."""
    monkeypatch.setattr(inspector, "_read_df", lambda _id: df)
    return ds_id


@pytest.fixture
def windowed_df(monkeypatch):
    fs = 100.0
    n = 800
    t = np.arange(n) / fs
    df = pd.DataFrame({
        "Time_s": t,
        "Sync": _sync_pulses(n, fs, [(1.0, 3.0), (5.0, 7.0)]),
        "L_ActForce_N": np.sin(2 * np.pi * t) * 30 + 50,
        "L_Pitch": np.cos(2 * np.pi * t) * 12,
    })
    return df, _register(monkeypatch, df)


def test_list_syncs_returns_two_windows(windowed_df):
    df, ds_id = windowed_df
    resp = inspector.list_syncs(ds_id)
    assert resp.column == "Sync"
    assert len(resp.cycles) == 2
    assert resp.cycles[0].t_start < resp.cycles[0].t_end
    assert resp.cycles[1].index == 1


# ============================================================
# /window (zoom-data fetch) is unchanged in behavior
# ============================================================

def test_fetch_window_slices_by_time(windowed_df):
    _, ds_id = windowed_df
    req = inspector.WindowRequest(
        columns=["L_ActForce_N", "L_Pitch", "missing"],
        t_start=0.5, t_end=4.0,
        max_points=80,
    )
    resp = inspector.fetch_window(ds_id, req)
    assert "missing" in resp.columns_missing
    names = {s.name for s in resp.series}
    assert names == {"L_ActForce_N", "L_Pitch"}
    assert resp.n_returned <= 80
    for s in resp.series:
        assert len(s.y) == len(resp.t)


def test_fetch_window_rejects_inverted_range(windowed_df):
    from fastapi import HTTPException
    _, ds_id = windowed_df
    with pytest.raises(HTTPException):
        inspector.fetch_window(ds_id, inspector.WindowRequest(
            columns=["L_Pitch"], t_start=2.0, t_end=1.0,
        ))


# ============================================================
# Downsampling helper
# ============================================================

def test_downsample_indices_below_max_returns_full_range():
    out = inspector._downsample_indices(100, max_points=4000)
    assert np.array_equal(out, np.arange(100))


def test_downsample_indices_above_max_strides_evenly():
    out = inspector._downsample_indices(10000, max_points=1000)
    assert len(out) == 1000
    assert out[0] == 0
    assert out[-1] == 9999
