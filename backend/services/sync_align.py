"""Multi-source sync alignment + window extraction + upsampling.

============================================================
SYNC SIGNAL CONTRACT (user-confirmed 2026-04-25)
============================================================

For an H-Walker recording, the operator presses a hardware sync
button at the start of each task period and releases it at the end.
The sync TTL is fanned out to:

  - Robot DAQ → recorded as the `Sync` column
  - Motion-capture system (Qualisys) → recorded as analog/trigger
  - (Loadcell calibration is unsynced, recorded separately)

Definition of one sync window:

      ___       ▔▔▔▔▔▔▔▔       ___       ▔▔▔▔▔▔       ___
         ┕━ rise           fall ┙   ┕━ rise       fall ┙
         ┕━━ window 1 (trial)  ━┙   ┕━ window 2 ━━━━━━┙

  - Rising edge = sync STARTS  (button pressed → trial begins)
  - Falling edge = sync ENDS   (button released → trial ends)
  - Window = the half-open interval [t_rise, t_fall]

Analysis happens **inside** the window only. Data before the first
rising edge (subject prep) or between windows (resting) is discarded.

A single recording can contain N sync windows = N independent trials.
============================================================

Module shape:

    SyncWindow                       — dataclass: rising_t_s, falling_t_s, ...
    find_sync_windows(df)            — every [rise, fall] in the source
    extract_window_slice(df, win)    — DataFrame slice for one window
    align_to_window_start(df, win)   — rebase t_aligned so window start = 0
    resample_to_grid(df, fs, t_min, t_max)
                                     — NaN-aware linear upsampling
    align_sources_on_window(...)     — N sources → common time grid

Backwards-compat shims `find_first_sync_falling_t` and `align_pair`
have been removed; callers must migrate to the window API. The earlier
falling-edge anchoring was based on a misinterpretation of the user's
spec and produced incorrect alignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Column auto-detection
# ------------------------------------------------------------

_TIME_CANDIDATES_S  = ("Time_s", "time_s", "Timestamp", "Time", "time", "T", "t")
_TIME_CANDIDATES_MS = ("Time_ms", "time_ms")
_SYNC_CANDIDATES    = ("Sync", "sync", "Trigger", "trigger", "TTL", "ttl")


def _find_time_column(df: pd.DataFrame) -> tuple[Optional[str], float]:
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
    """Monotonically-increasing seconds axis. Falls back to a 1 kHz
    sample-index axis when no time column exists."""
    col, scale = _find_time_column(df)
    if col is None:
        return np.arange(len(df), dtype=np.float64) * 1e-3
    t = df[col].to_numpy(dtype=np.float64) * scale
    if t.size > 1 and np.any(np.diff(t) < 0):
        # ms-counter wrap correction (rare but recoverable).
        d = np.diff(t)
        wraps = np.cumsum(np.where(d < 0, -d, 0.0))
        t = np.concatenate([[t[0]], t[1:] + wraps])
    return t


# ------------------------------------------------------------
# SyncWindow dataclass
# ------------------------------------------------------------

@dataclass(frozen=True)
class SyncWindow:
    """One [rising-edge, falling-edge] sync window in a recording.

    All times are in seconds, in the source's own time axis (i.e.
    the values straight out of the time column, before any cross-
    source alignment).
    """
    index: int                 # 0-based ordering within the recording
    rising_t_s: float
    falling_t_s: float
    sample_rising: int         # row index (0-based) of the rising edge
    sample_falling: int        # row index of the falling edge

    @property
    def duration_s(self) -> float:
        return self.falling_t_s - self.rising_t_s

    @property
    def n_samples(self) -> int:
        return self.sample_falling - self.sample_rising

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "rising_t_s": self.rising_t_s,
            "falling_t_s": self.falling_t_s,
            "duration_s": self.duration_s,
            "n_samples": self.n_samples,
        }


# ------------------------------------------------------------
# Window detection
# ------------------------------------------------------------

"""Default minimum trial duration (seconds).

Pulses shorter than this are treated as **phantom pulses** rather than
real operator-driven trials. The H-Walker DAQ shares a sync GPIO line
with the recording software's New-File / Save-File handlers, and those
file-IO events can briefly toggle the line for a few milliseconds. A
real trial is at least a few gait cycles (a stride is ~1 s), so 0.5 s
cleanly separates the two regimes:

    duration < 0.5 s  → phantom pulse, dropped
    duration ≥ 0.5 s  → real trial, kept

Override per-call via the `min_duration_s` argument when you need to
see *every* edge (e.g. debugging the recording line itself).
"""
DEFAULT_MIN_WINDOW_DURATION_S = 0.5


def find_sync_windows(df: pd.DataFrame,
                      sync_col: Optional[str] = None,
                      min_duration_s: float = DEFAULT_MIN_WINDOW_DURATION_S,
                      ) -> list[SyncWindow]:
    """Return every rising→falling sync window in the DataFrame.

    Algorithm:
      1. Threshold the sync signal at midpoint(min, max) so digital
         (0/1) and analog (TTL ramp) signals both work.
      2. Find all rising-edge sample indices.
      3. For each rising edge, find the **next** falling-edge sample.
      4. Emit the [rising, falling) pair as one SyncWindow.
      5. Drop any window narrower than `min_duration_s` — these are
         phantom pulses caused by file-IO toggling the sync line, not
         real operator-driven trials. Default 0.5 s.

    A trailing rising edge with no closing falling edge is dropped
    (incomplete window — the operator never released the button or
    the recording stopped mid-trial).

    Pass `min_duration_s=0.0` to keep every detected pulse, including
    millisecond glitches — useful when you want to see the raw line
    behaviour for debugging, but never the right setting for analysis.
    """
    sync_col = sync_col or _find_sync_column(df)
    if sync_col is None or sync_col not in df.columns:
        return []

    t = _seconds_axis(df)
    s = df[sync_col].to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(s)
    if finite_mask.sum() < 4:
        return []

    s_finite = s[finite_mask]
    t_finite = t[finite_mask]
    finite_idx = np.where(finite_mask)[0]      # mapping back to original rows

    lo, hi = float(np.min(s_finite)), float(np.max(s_finite))
    if hi - lo < 1e-9:
        return []   # constant signal → no edges

    threshold = (lo + hi) / 2.0
    high = (s_finite > threshold).astype(np.int8)
    diff = np.diff(high)
    rising_local  = np.where(diff ==  1)[0] + 1
    falling_local = np.where(diff == -1)[0] + 1
    if rising_local.size == 0 or falling_local.size == 0:
        return []

    raw_windows: list[SyncWindow] = []
    used_falling = 0
    for r_local in rising_local:
        # First falling edge AFTER this rising edge that we haven't
        # already paired with a previous rising edge.
        cands = falling_local[(falling_local > r_local)
                              & (np.arange(falling_local.size) >= used_falling)]
        if cands.size == 0:
            break
        f_local = cands[0]
        used_falling = int(np.where(falling_local == f_local)[0][0]) + 1
        raw_windows.append(SyncWindow(
            index=len(raw_windows),
            rising_t_s=float(t_finite[r_local]),
            falling_t_s=float(t_finite[f_local]),
            sample_rising=int(finite_idx[r_local]),
            sample_falling=int(finite_idx[f_local]),
        ))

    # Phantom-pulse rejection: drop any window narrower than the
    # configured minimum and re-index so the surviving windows are
    # numbered 0..N-1 contiguously (downstream code uses .index as a
    # consecutive trial id).
    if min_duration_s <= 0:
        return raw_windows
    survivors = [w for w in raw_windows if w.duration_s >= min_duration_s]
    return [
        SyncWindow(
            index=i,
            rising_t_s=w.rising_t_s,
            falling_t_s=w.falling_t_s,
            sample_rising=w.sample_rising,
            sample_falling=w.sample_falling,
        )
        for i, w in enumerate(survivors)
    ]


def find_sync_windows_raw(df: pd.DataFrame,
                          sync_col: Optional[str] = None,
                          ) -> list[SyncWindow]:
    """Return every rising→falling pair, including sub-`min_duration`
    phantom pulses. Use this when the caller needs to *see* the
    glitches (e.g. the inspector UI showing the user that 3 short
    spikes were dropped before analysis)."""
    return find_sync_windows(df, sync_col, min_duration_s=0.0)


# ------------------------------------------------------------
# Window slicing + rebasing
# ------------------------------------------------------------

def extract_window_slice(df: pd.DataFrame,
                          window: SyncWindow,
                          ) -> pd.DataFrame:
    """DataFrame slice corresponding to one sync window. Adds a
    `t_window_s` column rebased so window start = 0."""
    sub = df.iloc[window.sample_rising:window.sample_falling].copy()
    if sub.empty:
        return sub
    raw_t = _seconds_axis(df)[window.sample_rising:window.sample_falling]
    sub["t_window_s"] = raw_t - window.rising_t_s
    return sub.reset_index(drop=True)


def align_to_window_start(df: pd.DataFrame,
                           window: SyncWindow,
                           ) -> pd.DataFrame:
    """Add a `t_aligned` column to df, where t_aligned = 0 at the
    window's rising edge. The full df is preserved (not sliced).
    Use this when you need to keep pre/post-window samples for
    visualization or quality-check overlays."""
    out = df.copy()
    out["t_aligned"] = _seconds_axis(df) - window.rising_t_s
    return out


# ------------------------------------------------------------
# Uniform-grid resampling (NaN-aware linear interpolation)
# ------------------------------------------------------------

def resample_to_grid(df: pd.DataFrame,
                      target_fs: float,
                      t_min: float,
                      t_max: float,
                      time_col: str = "t_aligned",
                      value_cols: Optional[list[str]] = None,
                      ) -> pd.DataFrame:
    """Linearly interpolate `value_cols` onto a uniform grid at
    `target_fs` Hz. Returns NaN for grid samples outside the source's
    own range so per-stride logic upstream can skip the gap."""
    if target_fs <= 0:
        raise ValueError("target_fs must be > 0")
    if t_max <= t_min:
        raise ValueError("t_max must be > t_min")
    if time_col not in df.columns:
        raise KeyError(f"time column '{time_col}' missing")

    t_src = df[time_col].to_numpy(dtype=np.float64)
    n_grid = int(np.floor((t_max - t_min) * target_fs)) + 1
    t_grid = t_min + np.arange(n_grid) / target_fs

    if value_cols is None:
        value_cols = [c for c in df.columns
                      if c != time_col and pd.api.types.is_numeric_dtype(df[c])]

    order = np.argsort(t_src)
    t_sorted = t_src[order]
    uniq_mask = np.concatenate([[True], np.diff(t_sorted) > 0])
    t_uniq = t_sorted[uniq_mask]

    out: dict[str, np.ndarray] = {time_col: t_grid}
    for col in value_cols:
        y = df[col].to_numpy(dtype=np.float64)[order][uniq_mask]
        out[col] = np.interp(t_grid, t_uniq, y, left=np.nan, right=np.nan)
    return pd.DataFrame(out)


# ------------------------------------------------------------
# Multi-source alignment on a chosen sync window
# ------------------------------------------------------------

@dataclass
class AlignedSources:
    """Result of aligning N sources on a single sync-window index."""
    target_fs_hz: float
    t_min_s: float
    t_max_s: float
    n_grid_samples: int
    grids: dict[str, pd.DataFrame] = field(default_factory=dict)
    """{source_id: gridded_dataframe with t_aligned + numeric channels}"""
    window_per_source: dict[str, Optional[SyncWindow]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def align_sources_on_window(
    sources: dict[str, pd.DataFrame],
    window_idx: int = 0,
    target_fs: float = 1000.0,
    columns_per_source: Optional[dict[str, list[str]]] = None,
    min_duration_s: float = DEFAULT_MIN_WINDOW_DURATION_S,
) -> AlignedSources:
    """Align several sources on the **same sync-window index** within
    each source.

    Per-source pipeline:
      1. Detect that source's sync windows.
      2. Pick the window at `window_idx` (0 = first trial in recording).
      3. Anchor t_aligned = 0 at that window's rising edge.
      4. Resample to a common grid spanning the overlap of all
         source's t_aligned ranges.

    A source missing the requested window (e.g. Loadcell calibration
    has no Sync) is skipped with a warning; its grid is omitted from
    the result. Callers should check `warnings` and either retry with
    a different window or treat that source as unaligned.

    `min_duration_s` is forwarded to `find_sync_windows` so phantom
    pulses are filtered identically across sources.
    """
    if not sources:
        raise ValueError("at least one source required")

    warnings: list[str] = []
    aligned: dict[str, pd.DataFrame] = {}
    win_per_src: dict[str, Optional[SyncWindow]] = {}

    for src_id, df in sources.items():
        windows = find_sync_windows(df, min_duration_s=min_duration_s)
        if not windows:
            warnings.append(
                f"{src_id}: no sync window found — source skipped from alignment"
            )
            win_per_src[src_id] = None
            continue
        if window_idx >= len(windows):
            warnings.append(
                f"{src_id}: requested window {window_idx} but only "
                f"{len(windows)} present — source skipped"
            )
            win_per_src[src_id] = None
            continue
        win = windows[window_idx]
        aligned[src_id] = align_to_window_start(df, win)
        win_per_src[src_id] = win

    if not aligned:
        raise ValueError(
            "no source had the requested sync window — "
            "check sync TTL routing across the lab"
        )

    # Grid range = inside the sync window only (t = 0 at rising edge,
    # t = min(window_duration) at the earliest falling edge across
    # sources). Pre/post-window data is preserved in `aligned` and is
    # available for visualization but not part of the analysis grid.
    t_min = 0.0
    durations = [w.duration_s for w in win_per_src.values() if w is not None]
    if not durations:
        raise ValueError("no aligned source has a valid sync window")
    t_max = float(min(durations))
    if t_max <= t_min:
        raise ValueError(
            f"sync window has zero or negative duration ({t_max:.3f}s)"
        )

    grids: dict[str, pd.DataFrame] = {}
    for src_id, df_aligned in aligned.items():
        cols = (columns_per_source or {}).get(src_id)
        grids[src_id] = resample_to_grid(
            df_aligned, target_fs, t_min, t_max,
            time_col="t_aligned", value_cols=cols,
        )

    return AlignedSources(
        target_fs_hz=target_fs,
        t_min_s=float(t_min),
        t_max_s=float(t_max),
        n_grid_samples=len(next(iter(grids.values()))),
        grids=grids,
        window_per_source=win_per_src,
        warnings=warnings,
    )
