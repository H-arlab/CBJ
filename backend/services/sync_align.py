"""Multi-source time alignment + upsampling onto a common grid.

User-stated requirement (verbatim from the conversation):
    "sync가 안 맞는다니까 이거를 꼭 맞춰야 해 알지? upsampling
     해야지 시간으로 맞추면"

i.e. **never** trust nominal sample rates to line up across sources;
detect the first sync falling-edge in each CSV, set that to t = 0,
then linearly interpolate every channel onto a single high-rate
grid so per-stride windows can be sliced consistently across
Robot / Motion / Loadcell sources.

Key shapes:

    find_first_sync_falling_t(df, time_col=auto, sync_col=auto)
        → float seconds (or None if no sync cycle found)

    align_to_t0(df, t0, time_col=auto)
        → DataFrame with `t_aligned` column added (Time − t0, in seconds)

    resample_to_grid(df, target_fs, t_min, t_max,
                     time_col='t_aligned', value_cols=None)
        → DataFrame with one row per grid sample, NaN-aware linear
          interpolation, original column order preserved.

    align_pair(df_a, df_b, target_fs)  → (a_grid, b_grid) on the
        same uniform time axis, ready for cross-source per-stride
        analysis.

Uses the same falling-edge → rising-edge → falling-edge cycle
definition as backend/routers/inspector.py (CLAUDE.md: "디지털/
아날로그 sync 신호의 한 사이클").
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Column auto-detection helpers
# ---------------------------------------------------------------

_TIME_CANDIDATES_S    = ("Time_s", "time_s", "Timestamp", "Time", "time", "T", "t")
_TIME_CANDIDATES_MS   = ("Time_ms", "time_ms")
_SYNC_CANDIDATES      = ("Sync", "sync", "Trigger", "trigger", "TTL", "ttl")


def _find_time_column(df: pd.DataFrame) -> tuple[Optional[str], float]:
    """Return (column_name, scale_to_seconds). For Time_ms scale=1e-3."""
    for c in _TIME_CANDIDATES_S:
        if c in df.columns:
            return c, 1.0
    for c in _TIME_CANDIDATES_MS:
        if c in df.columns:
            return c, 1e-3
    return None, 1.0


def _find_sync_column(df: pd.DataFrame) -> Optional[str]:
    for c in _SYNC_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _seconds_axis(df: pd.DataFrame) -> np.ndarray:
    """Always-positive monotonically-increasing seconds axis. Falls
    back to a sample-index axis at 1 kHz when no time column exists
    (caller is responsible for using `align_to_t0` to anchor t=0)."""
    col, scale = _find_time_column(df)
    if col is None:
        return np.arange(len(df), dtype=np.float64) * 1e-3
    t = df[col].to_numpy(dtype=np.float64) * scale
    # Some firmwares ship a wrapping ms counter — undo wraps so t is
    # globally monotone.
    if t.size > 1:
        d = np.diff(t)
        if np.any(d < 0):
            wraps = np.cumsum(np.where(d < 0, -d.min() if d.min() < 0 else 0, 0.0))
            t = np.concatenate([[t[0]], t[1:] + wraps])
    return t


# ---------------------------------------------------------------
# Sync cycle detection (falling-pair)
# ---------------------------------------------------------------

def find_first_sync_falling_t(df: pd.DataFrame,
                               sync_col: Optional[str] = None,
                               ) -> Optional[float]:
    """Time (in seconds) of the first falling edge of the sync signal,
    or None if no falling edge exists.

    "First falling edge" anchors t = 0 across sources per the user's
    `sync` definition. Threshold midpoint between min and max so the
    routine handles both digital (0/1) and analog (TTL ramp) traces.
    """
    sync_col = sync_col or _find_sync_column(df)
    if sync_col is None:
        return None
    t = _seconds_axis(df)
    s = df[sync_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(s)
    if finite.sum() < 4:
        return None
    s = s[finite]
    t = t[finite]
    lo, hi = float(np.min(s)), float(np.max(s))
    if hi - lo < 1e-9:
        return None  # constant — no edges
    threshold = (lo + hi) / 2.0
    high = (s > threshold).astype(np.int8)
    diff = np.diff(high)
    falling_idx = np.where(diff == -1)[0] + 1
    if falling_idx.size == 0:
        return None
    return float(t[falling_idx[0]])


# ---------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------

def align_to_t0(df: pd.DataFrame,
                 t0: Optional[float] = None,
                 ) -> pd.DataFrame:
    """Add a `t_aligned` column to df (in seconds), where t_aligned = 0
    at the first sync falling edge. If t0 is None, detect from this
    DataFrame's own sync signal. If still None (no sync) we just
    return df with t_aligned = raw seconds axis (caller will see
    that the offset wasn't determined and can warn the user)."""
    if t0 is None:
        t0 = find_first_sync_falling_t(df)
        if t0 is None:
            t0 = 0.0
    out = df.copy()
    out["t_aligned"] = _seconds_axis(df) - t0
    return out


def resample_to_grid(df: pd.DataFrame,
                      target_fs: float,
                      t_min: float,
                      t_max: float,
                      time_col: str = "t_aligned",
                      value_cols: Optional[list[str]] = None,
                      ) -> pd.DataFrame:
    """Linear-interpolate `value_cols` (default: every numeric column
    except the time axis) onto a uniform grid at `target_fs`."""
    if target_fs <= 0:
        raise ValueError("target_fs must be > 0")
    if t_max <= t_min:
        raise ValueError("t_max must be > t_min")
    if time_col not in df.columns:
        raise KeyError(f"time column '{time_col}' missing from DataFrame")

    t_src = df[time_col].to_numpy(dtype=np.float64)
    n_grid = int(np.floor((t_max - t_min) * target_fs)) + 1
    t_grid = t_min + np.arange(n_grid) / target_fs

    if value_cols is None:
        value_cols = [c for c in df.columns
                      if c != time_col and pd.api.types.is_numeric_dtype(df[c])]

    out: dict[str, np.ndarray] = {time_col: t_grid}
    # Sort source by time once — interp requires monotone xp.
    order = np.argsort(t_src)
    t_sorted = t_src[order]
    # Drop duplicate timestamps (linear interp would zigzag otherwise).
    uniq_mask = np.concatenate([[True], np.diff(t_sorted) > 0])
    t_uniq = t_sorted[uniq_mask]

    for col in value_cols:
        y = df[col].to_numpy(dtype=np.float64)[order][uniq_mask]
        # Interp returns left/right boundary value outside [t_uniq];
        # we want NaN there so per-stride windows don't pull garbage
        # from before/after the recorded trial.
        gridded = np.interp(t_grid, t_uniq, y, left=np.nan, right=np.nan)
        out[col] = gridded

    return pd.DataFrame(out)


def align_pair(df_a: pd.DataFrame, df_b: pd.DataFrame,
                target_fs: float = 1000.0,
                ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Align two CSVs onto a single uniform time grid at `target_fs`.

    Each DataFrame is anchored at its own first sync-falling edge
    (so two sessions recorded with independent clocks line up at
    that physical event). The grid spans the maximum overlap window
    of the two sources.

    Returns (a_gridded, b_gridded, info) where `info` carries the
    per-source t0 in original seconds, the chosen grid range, and
    any warnings (e.g. "no sync edge found in B — alignment may be
    off by the source's recording offset").
    """
    info: dict = {"warnings": []}

    t0_a = find_first_sync_falling_t(df_a)
    t0_b = find_first_sync_falling_t(df_b)
    if t0_a is None:
        info["warnings"].append("no sync falling-edge in source A — anchored at 0")
    if t0_b is None:
        info["warnings"].append("no sync falling-edge in source B — anchored at 0")

    a = align_to_t0(df_a, t0_a)
    b = align_to_t0(df_b, t0_b)

    info["t0_a_s"] = float(t0_a) if t0_a is not None else None
    info["t0_b_s"] = float(t0_b) if t0_b is not None else None

    # Common overlap window (after alignment both axes share t=0 at sync).
    a_t = a["t_aligned"].to_numpy()
    b_t = b["t_aligned"].to_numpy()
    t_min = float(max(a_t.min(), b_t.min()))
    t_max = float(min(a_t.max(), b_t.max()))
    if t_max <= t_min:
        raise ValueError(
            f"sources have no temporal overlap after sync alignment: "
            f"A=[{a_t.min():.3f}, {a_t.max():.3f}], "
            f"B=[{b_t.min():.3f}, {b_t.max():.3f}]"
        )
    info["t_min_s"] = t_min
    info["t_max_s"] = t_max
    info["target_fs_hz"] = target_fs

    a_grid = resample_to_grid(a, target_fs, t_min, t_max)
    b_grid = resample_to_grid(b, target_fs, t_min, t_max)
    info["n_grid_samples"] = len(a_grid)
    return a_grid, b_grid, info
