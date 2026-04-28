"""Per-stride metric primitives for Motion-source data (Visual 3D /
Anybody / EMG).

Each primitive takes:
  - a `df` (DataFrame slice already restricted to one trial / sync
    window — caller is responsible for windowing)
  - a list of `stride_boundaries` = [(start_idx, end_idx), ...] in
    sample indices of `df` (caller computed these from Robot's GCP
    on the common time grid; see `cross_source_strides.py`)
  - the relevant column name(s)

and returns a list of per-stride scalar values aligned with
`stride_boundaries`. NaN is used when a stride is too short or the
column is missing — caller decides what to do (drop / impute).

All metrics are biomechanically defined, not user-tuned. Any
constant (50 ms loading-rate window, GCP threshold 0.01) is cited
in the docstring.

References:
  · Cavanagh & Lafortune (1980): GRF & loading rate definitions
  · Hof (1996): EMG normalization conventions
  · Kellis & Baltzopoulos (1998): co-contraction index formulation
  · Sutherland (2002): clinical gait parameters
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# Helper: sample-rate inference from a uniform grid
# ============================================================

def _grid_fs(df: pd.DataFrame, time_col: str) -> float:
    """Infer sample rate (Hz) from a uniform time column."""
    if time_col not in df.columns or len(df) < 2:
        raise ValueError(f"need ≥2 samples and column '{time_col}'")
    t = df[time_col].to_numpy(dtype=np.float64)
    dt = np.median(np.diff(t))
    if dt <= 0:
        raise ValueError(f"non-monotonic '{time_col}' axis (dt={dt})")
    return float(1.0 / dt)


def _slice(df: pd.DataFrame, start: int, end: int,
           col: str) -> Optional[np.ndarray]:
    """Numeric column slice [start:end]; None when the column is
    missing or the slice has fewer than 5 finite samples."""
    if col not in df.columns:
        return None
    if start < 0 or end > len(df) or end <= start:
        return None
    arr = df[col].to_numpy(dtype=np.float64)[start:end]
    finite = arr[np.isfinite(arr)]
    if finite.size < 5:
        return None
    return arr


# ============================================================
# Ground reaction force (real, from force plate — Motion source)
# ============================================================

def grf_fz_peak(df: pd.DataFrame,
                strides: list[tuple[int, int]],
                fz_col: str = "FP1_Fz") -> list[float]:
    """Peak vertical GRF per stride (N).

    Definition (Cavanagh & Lafortune 1980): max of vertical
    component of force plate during the stance portion of the
    stride. We use the entire stride window — swing-phase Fz is
    near zero so max is dominated by stance peak.
    """
    out: list[float] = []
    for s, e in strides:
        a = _slice(df, s, e, fz_col)
        if a is None:
            out.append(float("nan"))
            continue
        out.append(float(np.nanmax(a)))
    return out


def grf_fz_impulse(df: pd.DataFrame,
                    strides: list[tuple[int, int]],
                    fz_col: str = "FP1_Fz",
                    time_col: str = "t_aligned") -> list[float]:
    """Vertical GRF impulse per stride (N·s) — trapezoidal integration."""
    out: list[float] = []
    if time_col not in df.columns:
        return [float("nan")] * len(strides)
    t = df[time_col].to_numpy(dtype=np.float64)
    for s, e in strides:
        a = _slice(df, s, e, fz_col)
        if a is None:
            out.append(float("nan"))
            continue
        ts = t[s:e]
        mask = np.isfinite(a) & np.isfinite(ts)
        if mask.sum() < 5:
            out.append(float("nan"))
            continue
        out.append(float(np.trapezoid(a[mask], ts[mask])))
    return out


def grf_loading_rate(df: pd.DataFrame,
                      strides: list[tuple[int, int]],
                      fz_col: str = "FP1_Fz",
                      time_col: str = "t_aligned",
                      window_ms: float = 50.0) -> list[float]:
    """Loading rate per stride (N/s) — slope of GRF rise during
    the first `window_ms` after heel strike.

    Definition: linear regression of Fz vs t over [HS, HS+50 ms]
    (Cavanagh & Lafortune 1980; window from Crowell & Davis 2011).
    """
    out: list[float] = []
    if time_col not in df.columns:
        return [float("nan")] * len(strides)
    t = df[time_col].to_numpy(dtype=np.float64)
    fs = _grid_fs(df, time_col)
    n_window = max(int(round(window_ms / 1000.0 * fs)), 3)
    for s, e in strides:
        a = _slice(df, s, e, fz_col)
        if a is None:
            out.append(float("nan"))
            continue
        end_load = min(s + n_window, e)
        a_w = df[fz_col].to_numpy(dtype=np.float64)[s:end_load]
        t_w = t[s:end_load]
        mask = np.isfinite(a_w) & np.isfinite(t_w)
        if mask.sum() < 3:
            out.append(float("nan"))
            continue
        # Slope via least squares
        slope = np.polyfit(t_w[mask], a_w[mask], 1)[0]
        out.append(float(slope))
    return out


def cop_path_length(df: pd.DataFrame,
                     strides: list[tuple[int, int]],
                     copx_col: str = "FP1_COPx",
                     copy_col: str = "FP1_COPy") -> list[float]:
    """COP path length per stride (mm) — Σ‖ΔCOP‖ over stance.

    Definition: cumulative Euclidean displacement of the centre-of-
    pressure during stance phase. Caller should pre-filter to stance
    samples (CoP undefined during swing); for now we compute over
    all finite samples in the stride window.
    """
    out: list[float] = []
    for s, e in strides:
        x = _slice(df, s, e, copx_col)
        y = _slice(df, s, e, copy_col)
        if x is None or y is None:
            out.append(float("nan"))
            continue
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size < 3:
            out.append(float("nan"))
            continue
        dx = np.diff(x)
        dy = np.diff(y)
        out.append(float(np.sum(np.sqrt(dx**2 + dy**2))))
    return out


# ============================================================
# Joint kinematics (V3D output)
# ============================================================

def joint_peak_per_stride(df: pd.DataFrame,
                           strides: list[tuple[int, int]],
                           angle_col: str) -> list[float]:
    """Peak (max) angle per stride (degrees)."""
    out: list[float] = []
    for s, e in strides:
        a = _slice(df, s, e, angle_col)
        if a is None:
            out.append(float("nan"))
            continue
        out.append(float(np.nanmax(a)))
    return out


def joint_rom_per_stride(df: pd.DataFrame,
                          strides: list[tuple[int, int]],
                          angle_col: str) -> list[float]:
    """Range of motion (max − min) per stride (degrees)."""
    out: list[float] = []
    for s, e in strides:
        a = _slice(df, s, e, angle_col)
        if a is None:
            out.append(float("nan"))
            continue
        out.append(float(np.nanmax(a) - np.nanmin(a)))
    return out


def joint_moment_peak_per_stride(df: pd.DataFrame,
                                  strides: list[tuple[int, int]],
                                  moment_col: str) -> list[float]:
    """Peak (absolute) joint moment per stride (N·m). Sign is
    preserved via the position of the absolute extremum (so a strong
    extension moment shows up negative in flexion-positive convention)."""
    out: list[float] = []
    for s, e in strides:
        a = _slice(df, s, e, moment_col)
        if a is None:
            out.append(float("nan"))
            continue
        finite = a[np.isfinite(a)]
        if finite.size < 3:
            out.append(float("nan"))
            continue
        idx = np.nanargmax(np.abs(finite))
        out.append(float(finite[idx]))
    return out


# ============================================================
# EMG (per-muscle envelope on the aligned grid)
# ============================================================

def emg_rms_per_stride(df: pd.DataFrame,
                        strides: list[tuple[int, int]],
                        emg_col: str) -> list[float]:
    """RMS amplitude per stride (mV or %MVC, depending on input).

    Definition: sqrt(mean(emg²)) over the stride window. Caller is
    responsible for upstream EMG processing (band-pass 20–450 Hz,
    rectification, low-pass envelope) — this primitive only computes
    the per-stride RMS scalar.
    """
    out: list[float] = []
    for s, e in strides:
        a = _slice(df, s, e, emg_col)
        if a is None:
            out.append(float("nan"))
            continue
        finite = a[np.isfinite(a)]
        if finite.size < 5:
            out.append(float("nan"))
            continue
        out.append(float(np.sqrt(np.mean(finite ** 2))))
    return out


def emg_peak_per_stride(df: pd.DataFrame,
                         strides: list[tuple[int, int]],
                         emg_col: str) -> list[float]:
    """Peak rectified EMG amplitude per stride."""
    out: list[float] = []
    for s, e in strides:
        a = _slice(df, s, e, emg_col)
        if a is None:
            out.append(float("nan"))
            continue
        finite = a[np.isfinite(a)]
        if finite.size < 5:
            out.append(float("nan"))
            continue
        out.append(float(np.max(np.abs(finite))))
    return out


def cocontraction_index(df: pd.DataFrame,
                         strides: list[tuple[int, int]],
                         agonist_col: str,
                         antagonist_col: str) -> list[float]:
    """Co-contraction index per stride (Kellis & Baltzopoulos 1998).

        CCI = 2 · Σ min(EMG_a, EMG_b) / Σ (EMG_a + EMG_b)

    Range [0, 1]: 0 = no co-activation (one muscle silent),
    1 = perfectly equal antagonist activation. Inputs should already
    be normalized to %MVC (or at least to the same scale per muscle).
    Negative samples are clipped to 0 before the integral.
    """
    out: list[float] = []
    for s, e in strides:
        a = _slice(df, s, e, agonist_col)
        b = _slice(df, s, e, antagonist_col)
        if a is None or b is None:
            out.append(float("nan"))
            continue
        finite = np.isfinite(a) & np.isfinite(b)
        a_f = np.maximum(a[finite], 0.0)
        b_f = np.maximum(b[finite], 0.0)
        denom = float(np.sum(a_f + b_f))
        if denom < 1e-9:
            out.append(float("nan"))
            continue
        cci = 2.0 * float(np.sum(np.minimum(a_f, b_f))) / denom
        out.append(cci)
    return out
